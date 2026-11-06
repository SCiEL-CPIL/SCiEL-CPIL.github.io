"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np

def construct():
    n_max = 4096
    target_size = 80
    max_sum = 3 * n_max
    
    # We will use a vectorized greedy approach with batch processing.
    # This allows us to run many trials quickly.
    
    best_set = []
    best_size = 0
    num_trials = 100  # Increase trials since each is faster
    
    for trial in range(num_trials):
        if best_size == target_size:
            break
        
        # Random ordering of candidates
        rng = np.random.default_rng(seed=trial)
        candidates = rng.permutation(np.arange(1, n_max + 1))
        
        S = []
        # Boolean array to track which sums are already taken
        # Size 3 * 4096 = 12288
        sum_present = np.zeros(max_sum + 1, dtype=bool)
        
        # We process candidates in batches to reduce Python loop overhead
        # and leverage NumPy vectorization for conflict checking.
        
        i = 0
        while i < len(candidates) and len(S) < target_size:
            # Take a batch of candidates
            batch_size = 50
            batch = candidates[i : i + batch_size]
            i += batch_size
            
            if len(S) == 0:
                # First element can be anything
                x = batch[0]
                S.append(x)
                sum_present[3*x] = True
                continue
            
            # Convert current set S to numpy array for vectorized operations
            S_arr = np.array(S)
            k = len(S_arr)
            
            # We need to check for each candidate x in batch:
            # 1. 3x not in sum_present
            # 2. 2x + S_arr not in sum_present
            # 3. x + pair_sums(S_arr) not in sum_present
            
            valid_indices = []
            
            # Pre-compute pair sums of S_arr
            # Number of pairs is k*(k+1)/2
            if k > 1:
                i_idx, j_idx = np.triu_indices(k)
                pair_sums = S_arr[i_idx] + S_arr[j_idx]
            else:
                pair_sums = np.array([])
            
            # Vectorized check for the batch
            # batch is 1D array of shape (B,)
            # We want to find indices where no conflict exists.
            
            # 1. Check 3*x
            sums_3x = 3 * batch
            mask_3x = ~sum_present[sums_3x]
            
            # 2. Check 2*x + S
            # Shape: (B, K)
            # sums_2x_s = 2*batch[:, None] + S_arr[None, :]
            # We need to check if ANY of these are in sum_present.
            # sum_present is 1D. We index sum_present with sums_2x_s.
            # result: (B, K) boolean array.
            # If any is True, there is a conflict.
            sums_2x_s = 2 * batch[:, None] + S_arr[None, :]
            # Ensure indices are within bounds? Max sum is 3*4096.
            # 2*4096 + 4096 = 12288. Bounds are safe.
            mask_2x_s = ~np.any(sum_present[sums_2x_s], axis=1)
            
            # 3. Check x + pair_sums
            # Shape: (B, P) where P = k*(k+1)/2
            if len(pair_sums) > 0:
                sums_x_p = batch[:, None] + pair_sums[None, :]
                mask_x_p = ~np.any(sum_present[sums_x_p], axis=1)
            else:
                mask_x_p = np.ones(len(batch), dtype=bool)
            
            # Combine masks
            valid_mask = mask_3x & mask_2x_s & mask_x_p
            
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) > 0:
                # Pick the first valid candidate (or random one from valid)
                # Picking the first maintains a deterministic-ish order within the random batch
                # Picking random might be better for diversity?
                # Let's pick the first to be fast.
                chosen_idx = valid_indices[0]
                x = batch[chosen_idx]
                
                S.append(x)
                
                # Update sum_present
                sum_present[3*x] = True
                if k > 0:
                    new_sums_2x = 2*x + S_arr
                    sum_present[new_sums_2x] = True
                if k > 1:
                    new_sums_x_p = x + pair_sums
                    sum_present[new_sums_x_p] = True
            # If no valid candidate in batch, just continue to next batch
            # The while loop will handle i increment
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
# EVOLVE-BLOCK-END
