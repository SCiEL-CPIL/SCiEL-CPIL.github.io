"""
Strategy: Parametric Block-Pulse Representation with Differential Evolution.

This strategy fundamentally changes the search space representation. Instead of
searching in the high-dimensional binary space (2^400) or evolving it via CA,
we represent the continuous signal $x(t)$ as a linear combination of $K$
rectangular blocks (top-hat functions):
$$ x(t) = \sum_{j=1}^K A_j \cdot \mathbb{1}_{[P_j, P_j+W_j]}(t) $$
where $A_j$ is amplitude, $P_j$ is start position, and $W_j$ is width.

1. We define a low-dimensional parameter vector $\theta$ containing $\{A_j, P_j, W_j\}_{j=1}^K$.
2. We use Differential Evolution (DE) to optimize $\theta$ to minimize the
   autoconvolution peak sidelobe metric of the resulting *continuous* signal.
   This allows the optimizer to find "spectral nulls" in the continuous domain
   which are hard to reach in the discrete binary domain.
3. The optimal continuous signal is discretized and projected to signs
   ($x_i = \text{sign}(x_{cont}(i))$).
4. A final greedy polish is applied to the resulting binary sequence to
   recover any losses from quantization.

This approach leverages the smooth landscape of the continuous approximation to
navigate towards good basins of attraction in the discrete space.
"""

import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

# EVOLVE-BLOCK-START

def _autoconv_full(x):
    """Full linear autoconvolution of x (length 2N-1)."""
    n = len(x)
    if n == 0:
        return np.array([0.0])
    size = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, n=size)
    conv = np.fft.irfft(F * F, n=size)
    return conv[:2 * n - 1]

def _metric_continuous(params, K):
    """
    Compute the metric for a continuous block-pulse signal.
    params: flattened array of [A1, P1, W1, A2, P2, W2, ...]
    K: number of blocks
    """
    # Reconstruct signal on a fine grid for accurate convolution?
    # Or just evaluate on the discrete grid N=400?
    # The task asks for discrete sequence. But the optimization is on continuous params.
    # We evaluate the metric on the discrete samples of the continuous signal.
    
    # Extract params
    A = params[0::3]
    P = params[1::3]
    W = params[2::3]
    
    # Ensure widths are positive
    W = np.abs(W)
    
    # Create discrete signal
    t = np.arange(N, dtype=np.float64)
    x = np.zeros(N, dtype=np.float64)
    
    for j in range(K):
        a = A[j]
        p = P[j]
        w = W[j]
        
        # Block from p to p+w
        # Clip to [0, N)
        start = int(np.floor(p))
        end = int(np.floor(p + w))
        
        # Clip indices
        s_idx = max(0, start)
        e_idx = min(N, end)
        
        if s_idx < e_idx:
            x[s_idx:e_idx] += a
            
    # If signal is all zeros, return bad score
    integral = np.sum(x)
    if abs(integral) < 1e-9:
        return 1e12
        
    # Compute autoconvolution
    conv = _autoconv_full(x)
    if len(conv) <= 1:
        return 0.0
        
    max_abs_offdiag = np.max(np.abs(conv[1:]))
    score = max_abs_offdiag / (integral * integral)
    
    return score

def _objective_de(params):
    """Objective function for Differential Evolution."""
    return _metric_continuous(params, K_BLOCKS)

def _greedy_polish(x, deadline):
    """
    Greedy local search: flip bit i if it reduces the exact score.
    Optimized for speed using FFT updates if possible, but N=400 is small.
    """
    n = len(x)
    current = x.copy()
    current_score = _metric_continuous(current, 0) # Hack: use a dummy metric function? No, use specific one.
    
    # We need a metric function that works on the discrete array directly.
    def _metric_discrete(arr):
        integral = np.sum(arr)
        if abs(integral) < 1e-12:
            return 1e12
        conv = _autoconv_full(arr)
        if len(conv) <= 1:
            return 0.0
        max_abs_offdiag = np.max(np.abs(conv[1:]))
        return max_abs_offdiag / (integral * integral)
        
    current_score = _metric_discrete(current)
    best_score = current_score
    best_x = current.copy()
    
    size = 1 << (2 * n - 1).bit_length()
    
    no_improve = 0
    max_no_improve = 2000
    indices = np.arange(n)
    
    while time.time() < deadline and no_improve < max_no_improve:
        if no_improve > 100:
            np.random.shuffle(indices)
            
        improved = False
        for i in indices:
            if time.time() > deadline - 1.0:
                break
                
            old_val = current[i]
            current[i] = -old_val
            
            s = np.sum(current)
            if abs(s) < 1e-12:
                new_score = np.inf
            else:
                H = np.fft.rfft(current, size)
                ac = np.fft.irfft(H * H, size)[:2 * n - 1]
                max_abs = np.max(np.abs(ac[1:]))
                new_score = max_abs / (s * s)
                
            if new_score < current_score:
                current_score = new_score
                no_improve = 0
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_x = current.copy()
            else:
                current[i] = old_val
                no_improve += 1
                
        if not improved:
            if no_improve > 1000:
                break

    return best_x, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    rng = np.random.default_rng(42)
    
    # Parameters for the block-pulse representation
    # We use a moderate number of blocks to balance complexity and searchability.
    # 10 blocks * 3 params = 30 dimensions. DE can handle this.
    K_BLOCKS = 12
    
    # Bounds for each parameter:
    # A_j (Amplitude): [-2.0, 2.0]
    # P_j (Position): [0.0, N]
    # W_j (Width): [1.0, N]
    
    bounds = []
    for _ in range(K_BLOCKS):
        bounds.append((-2.0, 2.0))  # Amplitude
        bounds.append((0.0, float(N)))  # Position
        bounds.append((1.0, float(N)))  # Width
        
    # Differential Evolution
    # maxiter: limit to ensure we finish in time.
    # popsize: larger for better exploration, but slower.
    # tol: convergence tolerance.
    # strategy: 'best1bin' is standard. 'best2bin' might be more aggressive.
    
    # We want to spend about 60-80% of the time on DE, rest on polishing.
    # DE on 30 dims with popsize 20 might be slow. Let's monitor time.
    
    best_seq = None
    best_score = 1e12
    
    # Initial random guess to ensure we have a fallback
    init_seq = rng.choice([-1.0, 1.0], size=N)
    init_score = _metric_continuous(init_seq, 0) # Placeholder, will recompute
    
    def _metric_discrete(arr):
        integral = np.sum(arr)
        if abs(integral) < 1e-12:
            return 1e12
        conv = _autoconv_full(arr)
        if len(conv) <= 1:
            return 0.0
        max_abs_offdiag = np.max(np.abs(conv[1:]))
        return max_abs_offdiag / (integral * integral)
        
    init_score = _metric_discrete(init_seq)
    best_seq = init_seq.copy()
    best_score = init_score
    
    # Run Differential Evolution
    # We wrap the objective to track time and abort if necessary.
    de_deadline = deadline - 40.0 # Leave 40s for polishing and fallbacks
    
    try:
        # DE is stochastic, so we use seed for reproducibility in evaluation?
        # The task doesn't require deterministic output, but it's good practice.
        # However, DE seed might not be exposed in all versions or easy to control.
        # We'll rely on the internal RNG of DE or just let it run.
        
        # To handle time limits inside DE, we can't easily interrupt it.
        # So we must estimate maxiter.
        # One function eval takes ~0.1ms (FFT of 400 is fast).
        # Popsize 20, 100 iterations -> 2000 evals -> 0.2s.
        # We can afford many more.
        # Let's set maxiter to 200 and see.
        
        result = differential_evolution(
            _objective_de,
            bounds,
            maxiter=300,
            popsize=20,
            tol=1e-4,
            seed=42,
            polish=False, # We do our own polish
            init='latinhypercube'
        )
        
        if result.success or result.fun < best_score:
            # Get the continuous signal from the best params
            best_params = result.x
            A = best_params[0::3]
            P = best_params[1::3]
            W = np.abs(best_params[2::3])
            
            t = np.arange(N, dtype=np.float64)
            cont_signal = np.zeros(N, dtype=np.float64)
            
            for j in range(K_BLOCKS):
                a = A[j]
                p = P[j]
                w = W[j]
                
                start = int(np.floor(p))
                end = int(np.floor(p + w))
                
                s_idx = max(0, start)
                e_idx = min(N, end)
                
                if s_idx < e_idx:
                    cont_signal[s_idx:e_idx] += a
            
            # Project to signs
            # If signal is 0, assign random sign or 1?
            # sign(0) is 0 in numpy, which is bad for our +-1 constraint.
            # We map 0 to 1 (or random). Let's use 1 for stability, polish will fix.
            bin_signal = np.sign(cont_signal)
            bin_signal[bin_signal == 0] = 1.0
            
            # Evaluate the projected signal
            proj_score = _metric_discrete(bin_signal)
            
            if proj_score < best_score:
                best_score = proj_score
                best_seq = bin_signal.copy()
                
    except Exception:
        pass # Fallback to initial random
    
    # Polish the best candidate
    if time.time() < deadline - 5.0:
        best_seq, best_score = _greedy_polish(best_seq, deadline - 5.0)
        
    # Try a few other diverse candidates if time permits
    # 1. Random restart with polish
    if time.time() < deadline - 20.0:
        rand_seq = rng.choice([-1.0, 1.0], size=N)
        rand_polished, rand_score = _greedy_polish(rand_seq, deadline - 10.0)
        if rand_score < best_score:
            best_score = rand_score
            best_seq = rand_polished
            
    # 2. M-sequence (if time permits)
    def _m_sequence(n, k):
        polys = {
            6: [5, 3, 0],
            7: [6, 5, 3, 0],
            8: [7, 5, 3, 2, 0],
        }
        if k not in polys:
            return np.sign(np.sin(np.arange(n) * 2 * np.pi / n * 3))[:n]
        
        taps = polys[k]
        state = [0] * k
        state[0] = 1
        seq = []
        for _ in range(n):
            fb = 0
            for t in taps:
                fb ^= state[t]
            for i in range(k - 1, 0, -1):
                state[i] = state[i - 1]
            state[0] = fb
            seq.append(1.0 if state[0] else -1.0)
        return np.array(seq, dtype=np.float64)
        
    if time.time() < deadline - 10.0:
        m_seq = _m_sequence(N, 8)
        m_polished, m_score = _greedy_polish(m_seq, deadline - 5.0)
        if m_score < best_score:
            best_score = m_score
            best_seq = m_polished

    # Final polish on the absolute best
    if time.time() < deadline - 1.0:
        best_seq, best_score = _greedy_polish(best_seq, deadline - 1.0)
        
    return best_seq

# EVOLVE-BLOCK-END
