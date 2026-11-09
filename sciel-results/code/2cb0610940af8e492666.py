# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin before 60s kill

    def evaluate(x):
        """
        Compute max absolute sum over all arithmetic progressions in x.
        x is a numpy array of length n with values -1, 1.
        """
        max_abs = 1 # singletons
        # Iterate over steps 1 to n-1
        for d in range(1, n):
            # For this step d, consider all starts s in 0..d-1
            # We want to compute max subarray sum for each progression x[s::d]
            
            # Determine the maximum length of any progression with step d
            # The progression starting at s has length L_s = (n - 1 - s) // d + 1 if s < n
            # The longest is for s=0: L_max = (n - 1) // d + 1
            
            L_max = (n - 1) // d + 1
            
            # Create a 2D array of shape (d, L_max)
            # Initialize with zeros to handle padding
            # Note: We only care about valid elements. Zeros at the end don't affect 
            # the max-min difference if we are careful, but actually they DO if the 
            # prefix sum goes negative/positive and then stays there.
            # However, the prefix sum of zeros is 0. 
            # If the sequence ends, the prefix sum stays at its final value.
            # So padding with 0 is WRONG for prefix sums if we just cumsum.
            # We need to mask or handle lengths correctly.
            
            # Alternative: 
            # Construct the matrix M of shape (d, L_max) where M[s, k] = x[s + k*d] if s+k*d < n else 0.
            # Then P = cumsum(M, axis=1).
            # The valid prefix sums are P[s, :L_s].
            # The max subarray sum for row s is max(P[s, :L_s]) - min(P[s, :L_s])? 
            # No, we must include 0 as a starting point.
            # So we need min(0, min(P_valid)) and max(0, max(P_valid)).
            
            # Let's build M efficiently.
            # Indices for M:
            # Rows s in 0..d-1
            # Cols k in 0..L_max-1
            # M[s, k] = x[s + k*d]
            
            # We can use advanced indexing or broadcasting.
            # Let's create indices.
            # k_indices = np.arange(L_max)
            # s_indices = np.arange(d)
            # flat_indices = s_indices[:, None] + k_indices[None, :] * d
            # valid_mask = flat_indices < n
            # M = np.zeros((d, L_max), dtype=np.int8)
            # M[valid_mask] = x[flat_indices[valid_mask]]
            
            # This creates large arrays for small d. 
            # For d=1, L_max=256, size 256x256 = 65k.
            # For d=128, L_max=2, size 128x2 = 256.
            # Total memory/time is dominated by small d.
            
            # Let's try this vectorized approach.
            k_indices = np.arange(L_max)
            s_indices = np.arange(d)
            flat_indices = s_indices[:, None] + k_indices[None, :] * d
            valid_mask = flat_indices < n
            
            # Only fetch valid values
            # x is 1D, flat_indices is 2D.
            # We need to be careful with indexing.
            # M = np.zeros((d, L_max), dtype=np.int8)
            # M[valid_mask] = x[flat_indices[valid_mask]]
            
            # To avoid creating the full M if possible, or just do it.
            # 65k elements is small.
            M = np.zeros((d, L_max), dtype=np.int8)
            M[valid_mask] = x[flat_indices[valid_mask]]
            
            # Compute cumulative sums along columns (axis=1)
            P = np.cumsum(M, axis=1)
            
            # We need max(P) - min(P) for each row, but considering 0 is also a value.
            # p_min_row = np.minimum(0, P.min(axis=1))
            # p_max_row = np.maximum(0, P.max(axis=1))
            # row_maxes = p_max_row - p_min_row
            # local_max = np.max(row_maxes)
            
            # Optimization:
            # P.min(axis=1) and P.max(axis=1)
            # Note: P includes the cumulative sum of the padded zeros, which is 0.
            # So P[s, L_s-1] is the total sum. P[s, L_s:] are 0.
            # So min(P) will be <= 0 if the sum ever goes negative, or 0 if it stays positive?
            # No, if the sequence is all +1, P is [1, 2, ..., L]. min(P)=1.
            # But we must include the "empty prefix" 0.
            # So we need min(0, P.min()) and max(0, P.max()).
            
            p_mins = P.min(axis=1)
            p_maxs = P.max(axis=1)
            
            # If p_min > 0, min is 0. If p_min < 0, min is p_min.
            # If p_max < 0, max is 0. If p_max > 0, max is p_max.
            
            # We can compute:
            # effective_min = np.minimum(0, p_mins)
            # effective_max = np.maximum(0, p_maxs)
            # diffs = effective_max - effective_min
            # local_max = np.max(diffs)
            
            # Let's do it in one go
            # Since x are +/- 1, P starts at 0 effectively.
            # Actually, P[0] is x[0]. The "prefix before start" is 0.
            # So we compare 0 with P values.
            
            # Vectorized min/max with 0
            # min_val = np.where(p_mins < 0, p_mins, 0)
            # max_val = np.where(p_maxs > 0, p_maxs, 0)
            # local_max = np.max(max_val - min_val)
            
            # Even faster:
            # max_diff = np.max(np.maximum(0, p_maxs) - np.minimum(0, p_mins))
            
            local_max = np.max(np.maximum(0, p_maxs) - np.minimum(0, p_mins))
            
            if local_max > max_abs:
                max_abs = int(local_max)
                # Early exit if we know it's bad? 
                # If max_abs is already > some threshold, we can't really exit unless we have a known bound.
                # But for search, we need the exact value.
                
        return int(max_abs)

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
        
        # Greedy phase: First improvement
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
                if np.random.random() < np.exp(-delta / T):
                    current_val = new_val
                else:
                    x[i] = old
            
            T *= cooling
            if T < 0.01:
                break
        
        return best_x, best_val

    best_overall = None
    best_overall_val = 999
    
    rng = np.random.RandomState(42)
    
    # Allocation of time
    # Total time ~ 55s
    # We can afford more restarts now.
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
