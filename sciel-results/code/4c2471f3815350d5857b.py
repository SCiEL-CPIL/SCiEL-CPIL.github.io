import numpy as np

N = 100

def construct():
    # Objective function
    def get_value(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode="full")
        return np.max(c) / N * 2.0

    best_h = None
    best_val = float('inf')
    
    # Number of restarts
    restarts = 50
    
    for _ in range(restarts):
        # Initialize with random binary sequence (50 zeros, 50 ones)
        h = np.zeros(N)
        h[:50] = 1.0
        np.random.shuffle(h)
        
        current_val = get_value(h)
        
        # Local Search: Greedy Swap
        # We swap a 1 and a 0 to maintain the sum.
        # Indices of 1s and 0s
        ones = np.where(h == 1.0)[0]
        zeros = np.where(h == 0.0)[0]
        
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            
            # Iterate over all pairs of (zero_idx, one_idx)
            # To speed up, we can just iterate over all i, j
            # But N=100, so 10000 pairs is fine.
            
            # Pre-compute correlation? No, just calculate new value for best candidate.
            # To be efficient, we can estimate the change.
            # C_k = sum h_i (1-h_{i+k})
            # Let's just brute force the new value for the most promising swaps?
            # Actually, calculating full value for 10000 swaps * 100 steps is 1M correlate calls.
            # Each correlate is fast, but overhead might be high.
            
            # Better: Calculate the change in correlation vector C.
            # If we swap h[i]=0 to 1 and h[j]=1 to 0:
            # h_new[i] = 1, h_new[j] = 0.
            # Delta h: +1 at i, -1 at j.
            # g = 1-h. Delta g: -1 at i, +1 at j.
            # C = h * g (correlation).
            # This is getting complicated to update incrementally for 'max'.
            
            # Let's just do a limited number of random swaps or check all if fast enough.
            # 10000 swaps * 100 iterations * 50 restarts = 50M evaluations.
            # np.correlate(100, 100) is very fast, maybe ~1us?
            # 50M * 1us = 50s. Too slow.
            
            # Optimization: Only check a subset of swaps?
            # Or use a smarter heuristic.
            # Let's try checking swaps that are "far" apart or "near" apart?
            # Actually, the objective is dominated by the max lag.
            
            # Alternative: Use the analytical derivative?
            # f(h) = max_k C_k.
            # Gradient with respect to h is hard due to max.
            
            # Let's reduce restarts or steps.
            # Or, just pick a random swap that improves?
            
            # Let's try: Pick a random swap. If it improves, accept.
            # This is Simulated Annealing with T=0.
            # But we want a distinct approach.
            
            # Let's try a "Best Improvement" search but limit the candidates.
            # Sort indices by value? They are binary.
            
            # Let's try: For each 0 index, find the 1 index that best improves the max correlation.
            # We can approximate the change in C_k.
            # C_k = sum h_i g_{i+k}.
            # Change at i: h_i 0->1. g_i 1->0.
            # Change at j: h_j 1->0. g_j 0->1.
            
            # Let's just do a random walk with acceptance if better (Greedy Random Walk).
            # This is fast.
            
            for step in range(1000):
                # Pick random i from zeros, j from ones
                if len(zeros) == 0 or len(ones) == 0:
                    break
                
                i = zeros[np.random.randint(len(zeros))]
                j = ones[np.random.randint(len(ones))]
                
                # Perform swap
                h[i], h[j] = 1.0, 0.0
                
                new_val = get_value(h)
                
                if new_val < current_val:
                    current_val = new_val
                    # Update indices
                    # i was 0, now 1. j was 1, now 0.
                    # Remove i from zeros, add to ones.
                    # Remove j from ones, add to zeros.
                    # Using lists for efficiency
                    zeros = list(np.where(h == 0.0)[0])
                    ones = list(np.where(h == 1.0)[0])
                    improved = True
                else:
                    # Revert
                    h[i], h[j] = 0.0, 1.0
            
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
            
    return best_h
