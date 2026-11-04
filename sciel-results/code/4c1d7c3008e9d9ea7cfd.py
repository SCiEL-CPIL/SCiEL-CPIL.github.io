import numpy as np

N = 400

def construct():
    # Use a random support set to allow for sign cancellation
    # M=25 is a reasonable size, slightly larger than sqrt(N)
    M = 25
    
    # Random support
    support = np.random.choice(N, M, replace=False)
    
    # Initialize signs randomly
    signs = np.random.choice([-1, 1], M)
    
    x = np.zeros(N)
    x[support] = signs / np.sqrt(M)
    
    # Local search to minimize peak sidelobe
    # We track the max absolute value of y[k] for k != 0
    best_peak = np.inf
    best_x = x.copy()
    
    for _ in range(1000): # Iterations
        # Compute autoconvolution
        y = np.correlate(x, x, mode='full')
        # y is centered. y[N-1] is lag 0.
        # We want max |y[k]| for k != N-1
        y_sidelobes = np.delete(y, N-1)
        peak_idx = np.argmax(np.abs(y_sidelobes))
        # Map back to original index
        if peak_idx < N-1:
            current_lag = peak_idx
        else:
            current_lag = peak_idx + 1
            
        current_peak_val = np.abs(y[current_lag])
        
        if current_peak_val < best_peak:
            best_peak = current_peak_val
            best_x = x.copy()
            
        # Try to flip signs to reduce this specific lag
        # The lag 'current_lag' corresponds to difference d.
        # If current_lag < N, d = current_lag.
        # If current_lag >= N, d = current_lag - N.
        d = current_lag if current_lag < N else current_lag - N
        
        # Find pairs (i, j) in support such that support[i] - support[j] = d
        # Or support[j] - support[i] = d
        # We need to check contributions to y[d] and y[-d]
        
        # Efficiently find indices in support with difference d
        # support is an array of indices.
        # We can use a set for fast lookup.
        supp_set = set(support)
        
        pairs = []
        for idx, s in enumerate(support):
            target = s - d
            if target in supp_set:
                # Find index of target
                # This is O(M) per check, total O(M^2) per iteration
                # M=25, so 625 ops. 1000 iters = 625k ops. Fast.
                # But we need the index of target in support array
                # Let's precompute a map
                pass
        
        # Precompute map outside loop? No, support is fixed.
        # Let's just iterate.
        
        # Actually, let's just try flipping each sign and check if peak decreases.
        # This is O(M) flips per iteration, each O(N) conv? No, too slow.
        # Update conv incrementally?
        # Flipping sign of x[i] changes y[k] by -2 * x[i] * x[i-k] (roughly)
        
        # Simpler: Just try all M flips, compute new peak for the specific lag?
        # No, flipping one sign affects all lags.
        
        # Heuristic: Flip the sign of the element 's' in the pair (s, s-d) that has the same sign as the current sum contribution?
        
        # Let's identify the pairs contributing to lag d.
        contribs = []
        for i, s in enumerate(support):
            t = s - d
            if t in supp_set:
                # Find index of t
                for j, st in enumerate(support):
                    if st == t:
                        contribs.append((i, j))
                        break
        
        if not contribs:
            continue
            
        # Try flipping each sign in contribs
        improved = False
        for i, j in contribs:
            # Flip sign i
            x[support[i]] *= -1
            y_new = np.correlate(x, x, mode='full')
            peak_new = np.max(np.abs(np.delete(y_new, N-1)))
            if peak_new < current_peak_val:
                improved = True
                break
            else:
                x[support[i]] *= -1 # Revert
        
        if not improved:
            # Try flipping any sign
            for i in range(M):
                x[support[i]] *= -1
                y_new = np.correlate(x, x, mode='full')
                peak_new = np.max(np.abs(np.delete(y_new, N-1)))
                if peak_new < current_peak_val:
                    improved = True
                    break
                else:
                    x[support[i]] *= -1 # Revert
                    
        if not improved:
            # Random restart or break
            if np.random.rand() < 0.1:
                signs = np.random.choice([-1, 1], M)
                x[support] = signs / np.sqrt(M)
            else:
                break
                
    return best_x
