# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # safety margin before 60s kill

    # Precompute indices for each step d
    # For a given step d, we have d progressions starting at 0, 1, ..., d-1
    # We want to evaluate max absolute sum over all contiguous subarrays of these progressions.
    # Sum over subarray [i, j] in progression p is P[j+1] - P[i].
    # Max absolute sum is max(P) - min(P).
    
    def evaluate(x):
        """
        Compute max absolute sum over all arithmetic progressions in x.
        x is a numpy array of length n with values -1, 1.
        """
        max_abs = 1 # singletons
        # Iterate over steps 1 to n-1
        # For each step d, the progressions are x[0:d:n], x[1:d:n], ..., x[d-1:d:n]
        # We can process them all at once if we reshape?
        # Actually, slicing x[s::d] for s in range(d) creates views.
        # We can stack them? They have different lengths.
        # Better: loop over d, but vectorize over starts s?
        
        # For a fixed d, the lengths are roughly n/d.
        # We can create a 2D array where each row is a progression?
        # But lengths vary.
        
        # Alternative: Loop over d. For each d, loop over s in range(d).
        # This is O(n^2) Python loops if not careful.
        # Let's try to vectorize over s for a fixed d.
        
        # For fixed d:
        # max_len = max length of any progression with step d.
        # max_len = (n-1)//d + 1 (for s=0)
        # We can pad all progressions to max_len?
        # Or just loop over s? d is up to 255. Total iterations sum_{d=1}^{255} d = 32640.
        # Inside, we do numpy ops on arrays of length ~ n/d.
        # This should be fast enough.
        
        current_max = 1
        for d in range(1, n):
            # For this step d, consider all starts s in 0..d-1
            # We can process all starts for this d in a vectorized way?
            # Let's create an array of all elements involved in step d progressions.
            # It's easier to just loop s, but use numpy for the cumsum part.
            
            # Optimization:
            # For a fixed d, the set of indices is {s + k*d | s < d, k >= 0, s+k*d < n}
            # We can iterate s.
            # To speed up, we can precompute the max length for this d.
            # Actually, let's just use the loop but ensure numpy is doing the heavy lifting.
            
            # A faster way for fixed d:
            # Reshape x? No, x is 1D.
            # Let's stick to the loop over s but it's only 32k iterations total.
            # Each iteration does np.cumsum on an array of size ~ n/d.
            # Total operations ~ sum_{d=1}^{n} d * (n/d) = n^2 = 65k.
            # This is very fast in numpy.
            
            for s in range(d):
                # Extract progression
                # indices: s, s+d, s+2d, ...
                # Use slicing for speed
                vals = x[s::d]
                if len(vals) == 0:
                    continue
                # Prefix sum
                P = np.cumsum(vals)
                # We need max(P) - min(P) but also considering starting from 0?
                # The sum of a subarray is P[j] - P[i-1].
                # If we include 0 at the start: P_full = [0, P[0], P[1], ...]
                # Then max subarray sum is max(P_full) - min(P_full).
                # Note: P_full[0]=0.
                # So we just need to check if 0 is the min or max.
                # min_val = min(0, P.min())
                # max_val = max(0, P.max())
                # local_max = max_val - min_val
                
                # Efficiently:
                # p_min = P.min()
                # p_max = P.max()
                # if p_min > 0: local_max = p_max
                # elif p_max < 0: local_max = -p_min
                # else: local_max = p_max - p_min
                
                # Even simpler:
                # P_full_min = min(0, P.min())
                # P_full_max = max(0, P.max())
                # local_max = P_full_max - P_full_min
                
                p_min = P.min()
                p_max = P.max()
                if p_min > 0:
                    local_max = p_max
                elif p_max < 0:
                    local_max = -p_min
                else:
                    local_max = p_max - p_min
                
                if local_max > current_max:
                    current_max = local_max
                    # Early exit? No, we need the true max to guide search.
                    # But if current_max is already very high, maybe break?
                    # No, we need accurate gradient.
                    
        return int(current_max)

    def optimize(x_init, max_time=10.0):
        """
        Perform local search with simulated annealing.
        """
        x = x_init.copy()
        current_val = evaluate(x)
        best_x = x.copy()
        best_val = current_val
        
        t_start = time.time()
        T = 2.0 # Initial temperature
        cooling = 0.9999
        
        # Greedy phase first
        improved = True
        while improved and (time.time() - t_start) < max_time * 0.5:
            improved = False
            # Shuffle order to avoid bias
            indices = np.arange(n)
            np.random.shuffle(indices)
            for i in indices:
                if time.time() > t_start + max_time * 0.5:
                    break
                old = x[i]
                x[i] = -old
                new_val = evaluate(x)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                else:
                    x[i] = old
            
        # Simulated Annealing phase
        while (time.time() - t_start) < max_time:
            i = np.random.randint(n)
            old = x[i]
            x[i] = -old
            new_val = evaluate(x)
            
            delta = new_val - current_val
            if delta < 0:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
            else:
                # Accept with probability exp(-delta/T)
                # Since delta is integer, and T is float
                # If T is small, this is rare.
                if np.random.random() < np.exp(-delta / T):
                    current_val = new_val
                else:
                    x[i] = old
            
            T *= cooling
            if T < 0.01:
                break
        
        return best_x, best_val

    # Strategy: Multiple restarts with different initializations
    # 1. Thue-Morse
    # 2. Random
    # 3. Alternating
    # 4. Block Alternating
    
    best_overall = None
    best_overall_val = 999
    
    rng = np.random.RandomState(42)
    
    # Allocation of time
    # Total time ~ 55s
    # Let's do 5 restarts, each with 10s optimization.
    
    # 1. Thue-Morse
    def thue_morse(n):
        seq = [1]
        while len(seq) < n:
            seq = seq + [-s for s in seq]
        return np.array(seq[:n], dtype=np.int8)
    
    x_tm = thue_morse(n)
    x_opt, val_opt = optimize(x_tm, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
    
    # 2. Random
    x_rand = np.array(rng.choice([-1, 1], n), dtype=np.int8)
    x_opt, val_opt = optimize(x_rand, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 3. Alternating
    x_alt = np.array([(-1)**i for i in range(n)], dtype=np.int8)
    x_opt, val_opt = optimize(x_alt, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 4. Random again
    x_rand2 = np.array(rng.choice([-1, 1], n), dtype=np.int8)
    x_opt, val_opt = optimize(x_rand2, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 5. Block Alternating (period 4)
    x_block = np.array([(-1)**(i//4) for i in range(n)], dtype=np.int8)
    x_opt, val_opt = optimize(x_block, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt

    # If time remains, do a final polish on the best solution
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        x_opt, val_opt = optimize(best_overall, max_time=remaining)
        if val_opt < best_overall_val:
            best_overall_val = val_opt
            best_overall = x_opt

    return [int(v) for v in best_overall]

# EVOLVE-BLOCK-END
