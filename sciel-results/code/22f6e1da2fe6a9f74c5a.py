import numpy as np
import time

N = 400

def _init_corr(x):
    # x is +/- 1
    # c has length 2N-1
    # c[k + N - 1] = sum x_j x_{j+k}
    c = np.correlate(x, x, mode='full')
    return c

def _max_sidelobe(c):
    # c index N-1 is DC
    # Exclude it
    # c[:N-1] and c[N:]
    if N == 1:
        return 0
    return np.max(np.abs(np.concatenate([c[:N-1], c[N:]])))

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_x = None
    best_val = float('inf')
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        x = np.random.choice([1, -1], N)
        c = _init_corr(x)
        current_val = _max_sidelobe(c)
        
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline:
                    break
                
                xi = x[i]
                
                # Determine range of k affected
                # k such that 0 <= i+k < N and 0 <= i-k < N
                # -i <= k <= N-1-i
                k_min = max(-i, -(N-1))
                k_max = min(N-1-i, N-1)
                
                if k_min > k_max:
                    continue
                    
                # Indices in c
                # c_idx = k + N - 1
                idx_min = k_min + N - 1
                idx_max = k_max + N - 1
                
                # Calculate delta for these k
                # Delta c_k = -2 * xi * (x[i+k] + x[i-k])
                
                # We need to extract x[i+k] and x[i-k] for k in [k_min, k_max]
                # i+k ranges from i+k_min to i+k_max
                # i-k ranges from i-k_min to i-k_max (reversed)
                
                # Let's just loop over k for simplicity, N is small
                old_val = current_val
                # We will update c in place
                # But we need to know if the max changed.
                
                # Optimization: Just update c and check max.
                
                for k in range(k_min, k_max + 1):
                    term1 = x[i + k] if 0 <= i + k < N else 0
                    term2 = x[i - k] if 0 <= i - k < N else 0
                    delta = -2 * xi * (term1 + term2)
                    c[k + N - 1] += delta
                    
                # Update x
                x[i] = -x[i]
                
                new_val = _max_sidelobe(c)
                
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                else:
                    # Revert
                    x[i] = xi
                    for k in range(k_min, k_max + 1):
                        term1 = x[i + k] if 0 <= i + k < N else 0
                        term2 = x[i - k] if 0 <= i - k < N else 0
                        delta = -2 * xi * (term1 + term2)
                        c[k + N - 1] -= delta

    if best_x is None:
        return np.ones(N)
        
    return best_x
