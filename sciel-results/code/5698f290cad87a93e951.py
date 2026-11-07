import numpy as np

N = 100

def construct():
    # Initialize with 50 zeros and 50 ones
    h = np.zeros(N)
    h[:N//2] = 1.0
    
    # Shuffle with a fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(h)
    
    # Function to compute max correlation
    def get_max_corr(arr):
        # arr is h
        # 1-h is the other vector
        corr = np.correlate(arr, 1.0 - arr, mode="full")
        return np.max(corr)
    
    current_max = get_max_corr(h)
    
    # Greedy local search
    # We try to swap any 1 and 0
    # Indices of 1s and 0s
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    improved = True
    while improved:
        improved = False
        # To avoid checking all 2500 pairs every time which might be slow if many iterations,
        # we can just check a subset or all if fast enough.
        # 2500 pairs * 199 length corr * 100 ops = ~500k ops per iteration.
        # Python might take ~0.1s per iteration. 120s allows ~1000 iterations.
        
        best_swap = None
        best_val = current_max
        
        # Iterate through all possible swaps of a 1 and a 0
        for i in ones:
            for j in zeros:
                # Perform swap
                h[i], h[j] = h[j], h[i]
                
                # Compute new max
                # Optimization: only recompute if necessary? 
                # np.correlate is fast enough for N=100.
                new_corr = np.correlate(h, 1.0 - h, mode="full")
                new_max = np.max(new_corr)
                
                # Undo swap
                h[i], h[j] = h[j], h[i]
                
                if new_max < best_val:
                    best_val = new_max
                    best_swap = (i, j)
        
        if best_swap:
            i, j = best_swap
            h[i], h[j] = h[j], h[i]
            current_max = best_val
            # Update lists of ones and zeros
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
            improved = True
            
    return h
