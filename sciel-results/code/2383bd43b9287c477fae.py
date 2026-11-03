"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Block Coordinate Descent exploiting the convexity of the objective
in each individual variable h_i, combined with random restarts.
"""
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(10):
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            diff -= add
            if abs(diff) > 1e-12:
                mask = h < 1.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            diff += sub
            if abs(diff) > 1e-12:
                mask = h > 0.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
                    
    return h


def _optimize_single_coordinate(h, i):
    """
    Optimize h[i] while keeping other h[j] fixed.
    The objective max_k C_k is convex in h[i].
    """
    n = len(h)
    # Pre-compute parts of C_k that do not depend on h[i]
    # C_k = (2/n) * (n/2 - sum_j h_j h_{j-k})
    # Terms involving h[i]:
    # 1. j=i: h[i] * h[(i-k)%n]  (Note: correlation is circular? 
    #    The problem statement says "shifted 1-h". 
    #    The seed code uses `corr[:n]` from `irfft` of zero-padded h.
    #    This implies linear correlation (zero-padding), not circular.
    #    Let's re-verify the seed code's `_linear_overlap_values`.
    #    h_pad is zero padded. So correlation is linear.
    #    r[k] = sum_{j} h[j] h[j+k] ? 
    #    irfft(FFT(h)*conj(FFT(h))) gives circular autocorrelation of the padded signal.
    #    Since h is zero padded to 2N, the circular autocorrelation of length 2N
    #    restricted to first N lags is the linear autocorrelation.
    #    r[k] = sum_{j=0}^{N-1-k} h[j] h[j+k] for k>=0.
    #    Wait, the seed code does:
    #    r = corr[:n]
    #    cs = (2.0 / n) * (n / 2.0 - r)
    #    This r corresponds to lags k=0..N-1.
    #    For a specific lag k, r[k] = sum_{j} h[j] h[j+k] (with h[j+k]=0 if out of bounds).
    #    
    #    So, for a fixed k, the term in the sum is h[j] * h[j+k].
    #    h[i] appears if j=i (term h[i]*h[i+k]) or j+k=i => j=i-k (term h[i-k]*h[i]).
    #    So h[i] appears in r[k] with coefficient:
    #    coeff_k = h[i+k] (if i+k < N) + h[i-k] (if i-k >= 0)
    #    Let A_k = coeff_k.
    #    Let B_k = r[k] - h[i] * A_k. (The rest of the sum).
    #    Then C_k = (2/n) * (n/2 - (B_k + A_k * h[i]))
    #    C_k = - (2/n) * A_k * h[i] + (2/n) * (n/2 - B_k)
    #    This is linear in h[i].
    
    # Calculate A and B vectors for all k
    A = np.zeros(n)
    B = np.zeros(n)
    
    # Current r values
    r_current = _linear_overlap_values(h)
    
    # For each k, determine contribution of h[i]
    # r[k] = sum_{j=0}^{N-1-k} h[j] h[j+k]
    # h[i] is in the sum if:
    # 1. j = i and i+k < N. Term: h[i] h[i+k].
    # 2. j+k = i and i-k >= 0. Term: h[i-k] h[i].
    
    for k in range(n):
        coeff = 0.0
        # Case 1: j = i
        if i + k < n:
            coeff += h[i + k]
        # Case 2: j = i - k
        if i - k >= 0:
            coeff += h[i - k]
            
        A[k] = coeff
        # B[k] is the sum excluding the h[i] terms
        # r[k] = B[k] + A[k] * h[i]
        B[k] = r_current[k] - A[k] * h[i]
        
    # Objective: max_k C_k(h_i)
    # C_k(x) = (2.0 / n) * (n / 2.0 - (B[k] + A[k] * x))
    # C_k(x) = - (2.0 / n) * A[k] * x + (2.0 / n) * (n / 2.0 - B[k])
    
    def obj_1d(x):
        cs = (2.0 / n) * (n / 2.0 - (B + A * x))
        return np.max(cs)
        
    # Minimize obj_1d(x) for x in [0, 1]
    # Since it's convex, we can use minimize_scalar with method='bounded'
    res = minimize_scalar(obj_1d, bounds=(0.0, 1.0), method='bounded')
    
    # Check if the new value is better
    new_val = res.fun
    old_val = obj_1d(h[i])
    
    if new_val < old_val:
        h[i] = res.x
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    
    # 1. Analytical Starts
    starts = []
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * (1.0 - 2.0 * np.abs((2.0 * i / N) - 1.0))))
    starts.append(np.full(N, 0.5))
    starts.append(_fix_sum(0.5 + 0.5 * (2.0 * (i / N) - 1.0)))
    
    # 2. Random Starts
    for _ in range(5):
        if time.time() - t0 > _DEADLINE - 30:
            break
        h_rand = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_rand))

    # Phase 1: Coordinate Descent from all starts
    for h_start in starts:
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        _update_best(h_start)
        h_current = h_start.copy()
        
        # Run Coordinate Descent
        # Random order to avoid cyclic biases
        order = np.random.permutation(N)
        max_sweeps = 100
        prev_score = _score(h_current)
        
        for sweep in range(max_sweeps):
            if time.time() - t0 > _DEADLINE - 5:
                break
                
            for idx in order:
                if time.time() - t0 > _DEADLINE - 5:
                    break
                h_current = _optimize_single_coordinate(h_current, idx)
                
            # Check for convergence
            curr_score = _score(h_current)
            if abs(prev_score - curr_score) < 1e-8:
                break
            prev_score = curr_score
            
        _update_best(h_current)

    # Phase 2: Refinement with small random perturbations (Simulated Annealing lite)
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_current = best_h.copy()
        current_score = best_score
        
        # Simple hill climbing with random restarts on coordinates
        for step in range(2000):
            if time.time() - t0 > _DEADLINE - 5:
                break
                
            # Pick a random coordinate to optimize
            idx = np.random.randint(N)
            h_new = h_current.copy()
            h_new = _optimize_single_coordinate(h_new, idx)
            
            new_score = _score(h_new)
            if new_score < current_score:
                h_current = h_new
                current_score = new_score
                _update_best(h_current)
                
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
