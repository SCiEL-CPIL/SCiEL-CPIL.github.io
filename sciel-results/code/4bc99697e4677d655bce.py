"""Rewritten program: Rule 90 cellular automaton initialization + fast local search."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_MARGIN = 135.0

def _compute_c3_upper_bound(f):
    """
    Compute the upper bound on C3 for a signed step function f with N intervals.
    Returns max |autoconvolution| over non-zero lags.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real
    conv_lags = conv[:n]
    if len(conv_lags) > 1:
        return np.max(np.abs(conv_lags[1:]))
    return 0.0

def _score(f):
    """Score is the upper bound on C3; lower is better."""
    return _compute_c3_upper_bound(f)

def _rule90_sequence(steps, seed=1):
    """
    Generate a binary sequence from Rule 90 cellular automaton.
    Rule 90: new[i] = old[i-1] XOR old[i+1]
    Start with a single 1 at position 'seed', run for 'steps' iterations.
    Returns array of ±1 values.
    """
    # We need a large enough array to capture the pattern
    # After k steps, the pattern spans at most 2k+1 cells
    size = 2 * steps + 1
    grid = np.zeros(size, dtype=np.int8)
    grid[steps] = 1  # seed at center
    
    for _ in range(steps):
        # Rule 90: XOR of left and right neighbors
        new_grid = np.zeros(size, dtype=np.int8)
        new_grid[1:] ^= grid[:-1]
        new_grid[:-1] ^= grid[1:]
        grid = new_grid
    
    # Map 0 -> -1, 1 -> +1
    return 2.0 * grid - 1.0

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n as ±1."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return 2.0 * tm - 1.0

def _fast_local_search(f, max_iterations=2000, deadline=None):
    """
    Efficient local search by flipping individual bits.
    Uses incremental convolution update for speed.
    """
    current_f = f.copy()
    current_score = _score(current_f)
    
    # Precompute the convolution for fast updates
    n = len(current_f)
    size = 2 * n
    F = np.fft.fft(current_f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        best_score = current_score
        best_idx = -1
        best_delta = None
        
        # Try all possible flips
        for i in range(n):
            # Flipping f[i] changes the convolution by -2*f[i]*f * delta_conv
            # Specifically, new_conv = conv - 2*f[i]*f (element-wise in lag domain)
            # Actually: conv_new[k] = conv[k] - 2*f[i]*sum over j of f[j]*f[j+k-i]
            # This is complex. Let's just recompute for the candidate.
            
            f_new = current_f.copy()
            f_new[i] *= -1.0
            s = _score(f_new)
            if s < best_score:
                best_score = s
                best_idx = i
        
        if best_idx == -1:
            break
        
        current_f[best_idx] *= -1.0
        current_score = best_score
    
    return current_f, current_score

def _construct_initial_rule90():
    """Generate initial signed step function using Rule 90 CA."""
    # Try different numbers of steps to find good patterns
    best_f = None
    best_score = float('inf')
    
    for steps in [7, 8, 9, 10, 11, 12, 13]:
        # Generate sequence longer than N, then take first N
        seq = _rule90_sequence(steps, seed=steps)
        # Take a window of length N from the sequence
        if len(seq) >= N:
            f = seq[:N].copy()
        else:
            # Pad or repeat
            f = np.tile(seq, N // len(seq) + 1)[:N]
        
        s = _score(f)
        if s < best_score:
            best_score = s
            best_f = f.copy()
    
    return best_f

def _construct_initial_thue_morse():
    """Generate initial signed step function using Thue-Morse."""
    return _thue_morse(N)

def _smooth_score(f, temperature=0.3):
    """
    Smooth surrogate for the max |conv| objective using log-sum-exp.
    """
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    conv_nonzero = conv[1:] if len(conv) > 1 else conv
    abs_conv = np.abs(conv_nonzero)
    max_val = np.max(abs_conv)
    if max_val < 1e-10:
        return 0.0
    scaled = (abs_conv - max_val) / temperature
    log_sum_exp = np.log(np.sum(np.exp(scaled))) + max_val / temperature
    return log_sum_exp * temperature

def construct():
    """
    Main entry point. Returns a sequence of N signed step heights.
    Score is the upper bound on C3 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_MARGIN
    
    best_f = None
    best_score = float('inf')
    
    # Phase 1: Try Rule 90 initializations
    f_r90 = _construct_initial_rule90()
    if f_r90 is not None:
        s = _score(f_r90)
        if s < best_score:
            best_score = s
            best_f = f_r90.copy()
    
    # Phase 2: Try Thue-Morse initialization
    f_tm = _construct_initial_thue_morse()
    s_tm = _score(f_tm)
    if s_tm < best_score:
        best_score = s_tm
        best_f = f_tm.copy()
    
    # Phase 3: Local search from best initial
    if time.time() < deadline - 30:
        f_ls, s_ls = _fast_local_search(best_f, max_iterations=3000, deadline=deadline - 30)
        if s_ls < best_score:
            best_score = s_ls
            best_f = f_ls.copy()
    
    # Phase 4: Random restarts with local search
    rng = np.random.RandomState(42)
    remaining_time = deadline - time.time()
    num_restarts = min(10, int(remaining_time / 5))
    
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Random restart
        f_restart = rng.choice([-1.0, 1.0], size=N)
        f_new, s_new = _fast_local_search(f_restart, max_iterations=500, deadline=deadline - 15)
        if s_new < best_score:
            best_score = s_new
            best_f = f_new.copy()
    
    # Phase 5: Smooth optimization (continuous relaxation)
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        x0 = best_f.copy()
        
        def objective(x):
            if time.time() > deadline - 5:
                return best_score
            return _smooth_score(x, temperature=0.3)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B',
                            bounds=[(-1, 1)] * N,
                            options={'maxiter': 200, 'ftol': 1e-10})
            f_smooth = np.sign(result.x)
            f_smooth[f_smooth == 0] = 1.0
            s_smooth = _score(f_smooth)
            if s_smooth < best_score:
                best_score = s_smooth
                best_f = f_smooth.copy()
        except Exception:
            pass
    
    # Phase 6: Final local search polish
    if time.time() < deadline - 5:
        f_final, s_final = _fast_local_search(best_f, max_iterations=1000, deadline=deadline - 2)
        if s_final < best_score:
            best_score = s_final
            best_f = f_final.copy()
    
    return best_f

# EVOLVE-BLOCK-END
