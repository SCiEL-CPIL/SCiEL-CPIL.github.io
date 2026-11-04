import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    # Initialize with random signs
    f = np.random.choice([1.0, -1.0], N)
    
    # Helper to compute metrics
    def evaluate(f_vec):
        # Aperiodic autocorrelation
        # np.correlate(f, f, 'full') returns 2N-1 values
        # Center is N-1
        corr = np.correlate(f_vec, f_vec, mode='full')
        # Extract non-zero lags
        # Indices 0 to N-2 are lags -N+1 to -1
        # Indices N to 2N-2 are lags 1 to N-1
        # We want max abs of these.
        # corr[0:N-1] and corr[N:2N-1]
        lags = np.concatenate((corr[:N-1], corr[N:]))
        max_abs = np.max(np.abs(lags))
        sum_sq = np.sum(lags * lags)
        return max_abs, sum_sq

    curr_max, curr_sq = evaluate(f)
    best_max = curr_max
    best_f = f.copy()
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    # Pre-allocate arrays if needed? No, np.correlate is fast enough.
    
    while time.time() < deadline:
        # Pick random index
        i = np.random.randint(N)
        
        # Flip
        old_val = f[i]
        f[i] = -old_val
        
        new_max, new_sq = evaluate(f)
        
        # Accept if better or with probability
        if new_sq < curr_sq:
            curr_max = new_max
            curr_sq = new_sq
            if curr_max < best_max:
                best_max = curr_max
                best_f = f.copy()
        else:
            delta = new_sq - curr_sq
            # Avoid overflow in exp
            if delta < 1000 * T:
                if np.random.rand() < np.exp(-delta / T):
                    curr_max = new_max
                    curr_sq = new_sq
                    if curr_max < best_max:
                        best_max = curr_max
                        best_f = f.copy()
                else:
                    # Revert
                    f[i] = old_val
            else:
                # Revert
                f[i] = old_val
        
        T *= cooling
        if T < min_T:
            T = min_T

    return best_f
