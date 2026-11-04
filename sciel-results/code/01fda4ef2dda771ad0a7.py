import numpy as np
import time

N = 400

def _score(h):
    # FFT autocorrelation
    n = len(h)
    if n == 0:
        return 0.0
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    acf = np.fft.irfft(np.abs(H) ** 2, n=fft_size)
    # Return max abs acf at lags 1..n-1
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:n])))

def _optimize_signs(h, time_limit=1.0):
    """Greedy sign flipping to minimize peak autocorrelation."""
    current_h = h.copy()
    current_score = _score(current_h)
    start = time.time()
    n = len(current_h)
    nonzero_idx = np.nonzero(current_h)[0]
    
    if len(nonzero_idx) == 0:
        return current_h, current_score

    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Random order to avoid bias
        order = np.random.permutation(len(nonzero_idx))
        for idx_pos in order:
            i = nonzero_idx[idx_pos]
            trial = current_h.copy()
            trial[i] = -trial[i]
            s = _score(trial)
            if s < current_score - 1e-12:
                current_h = trial
                current_score = s
                improved = True
                # Restart order? Or continue? Restarting is more thorough.
                break
    return current_h, current_score

def _optimize_support(h, time_limit=10.0, T_start=5.0):
    """
    Optimizes the support (indices with non-zero values) and signs.
    Moves:
    1. Remove a random non-zero element.
    2. Add a random zero element with random sign.
    3. Swap a non-zero element with a zero element.
    After each structural move, performs a few sign optimizations.
    """
    current_h = h.copy()
    current_score = _score(current_h)
    best_h = current_h.copy()
    best_score = current_score
    
    start = time.time()
    T = T_start
    cooling = 0.99
    
    n = len(current_h)
    
    while T > 0.01 and time.time() - start < time_limit:
        # Generate a trial move
        trial_h = current_h.copy()
        
        nonzero_idx = np.nonzero(current_h)[0]
        zero_idx = np.nonzero(current_h == 0)[0]
        
        move_type = np.random.random()
        
        if len(nonzero_idx) == 0:
            # Must add
            if len(zero_idx) > 0:
                i = zero_idx[np.random.randint(len(zero_idx))]
                trial_h[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                break
        elif len(zero_idx) == 0:
            # Must remove
            i = nonzero_idx[np.random.randint(len(nonzero_idx))]
            trial_h[i] = 0.0
        else:
            if move_type < 0.3:
                # Remove
                i = nonzero_idx[np.random.randint(len(nonzero_idx))]
                trial_h[i] = 0.0
            elif move_type < 0.6:
                # Add
                i = zero_idx[np.random.randint(len(zero_idx))]
                trial_h[i] = 1.0 if np.random.random() < 0.5 else -1.0
            else:
                # Swap
                i_out = nonzero_idx[np.random.randint(len(nonzero_idx))]
                i_in = zero_idx[np.random.randint(len(zero_idx))]
                val = current_h[i_out]
                trial_h[i_out] = 0.0
                trial_h[i_in] = val # Keep sign, or randomize? Randomize to explore.
                if np.random.random() < 0.5:
                    trial_h[i_in] = -trial_h[i_in]
        
        # Quick sign optimization on the trial
        # Limit time for this sub-step to keep outer loop fast
        sub_time = min(0.1, time_limit - (time.time() - start))
        if sub_time > 0:
            trial_h, trial_score = _optimize_signs(trial_h, time_limit=sub_time)
        else:
            trial_score = _score(trial_h)
            
        delta = trial_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_h = trial_h
            current_score = trial_score
            if trial_score < best_score:
                best_score = trial_score
                best_h = trial_h.copy()
                
        T *= cooling
        
    return best_h, best_score

def construct():
    deadline = time.time() + 140
    best_h = None
    best_score = float('inf')
    
    # Strategy: Sparse Combinatorial Design
    # Try various sparsity levels
    # The optimal sparsity for minimizing peak autocorrelation is likely low-moderate.
    # Random full sequences have peak ~ sqrt(N) ~ 20.
    # Sparse sequences can have lower peaks if signs are well-balanced.
    
    sparsities = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
    
    for k in sparsities:
        if time.time() > deadline - 10:
            break
            
        # Multiple restarts for each sparsity
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Initialize sparse random support
            h = np.zeros(N)
            if k < N:
                indices = np.random.choice(N, size=k, replace=False)
                h[indices] = np.random.choice([-1, 1], size=k)
            else:
                h = np.random.choice([-1, 1], size=N).astype(float)
                
            # Phase 1: Optimize signs for fixed support
            h_opt, score = _optimize_signs(h, time_limit=2.0)
            
            # Phase 2: Optimize support and signs together
            if time.time() < deadline - 5:
                time_left_for_sa = min(10.0, deadline - time.time() - 2.0)
                if time_left_for_sa > 1.0:
                    h_opt, score = _optimize_support(h_opt, time_limit=time_left_for_sa)
                    
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
                
    # Final polish on the best solution found
    if best_h is not None and time.time() < deadline - 5:
        time_left = deadline - time.time() - 1.0
        if time_left > 1.0:
            best_h, best_score = _optimize_signs(best_h, time_limit=time_left * 0.5)
            if time.time() < deadline - 2.0:
                best_h, best_score = _optimize_support(best_h, time_limit=min(5.0, deadline - time.time() - 1.0))

    # Fallback
    if best_h is None:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        best_score = _score(best_h)
        
    return best_h
