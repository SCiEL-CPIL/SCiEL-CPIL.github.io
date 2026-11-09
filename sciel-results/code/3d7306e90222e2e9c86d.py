import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    N = 17
    MIN_D = 6
    max_val = 1 << N
    
    # Precompute weights using vectorized approach
    # max_val is 2**17 = 131072
    # Create an array of indices 0..max_val-1
    idx = np.arange(max_val, dtype=np.uint32)
    # Use numpy bit operations to count bits
    # This might be memory heavy if not careful, but 131k is small.
    # Alternative: simple loop is ~0.1s, which is fine.
    # But let's stick to a safe integer type. uint8 is enough for weights <= 17.
    weights = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        weights[i] = weights[i >> 1] + (i & 1)
        
    best_code = []
    best_size = 0
    
    # Try different linear code sizes
    # k=8 is the max linear size with d>=6
    # We try k=8, 7, 6, 5
    for k in [8, 7, 6, 5]:
        if time.time() > deadline - 10:
            break
            
        # 1. Construct Linear Code of size 2^k
        basis = []
        current_lin = [0]
        
        # Greedy linear construction
        # We scan 1 to max_val-1
        # Optimization: only scan if we have time
        for g in range(1, max_val):
            if len(basis) >= k:
                break
            if time.time() > deadline - 10:
                break
                
            # Check if g is independent and maintains distance
            # For linear code, we just need to check distance to current_lin
            # But since it's linear, we can just check against basis?
            # No, distance to any codeword in the span.
            # current_lin contains all codewords in the span of basis.
            
            valid = True
            # Vectorized check
            # current_lin is a list of ints. Convert to array?
            # For small sizes (<=256), loop is fine.
            for c in current_lin:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
                    
            if valid:
                basis.append(g)
                old_code = current_lin
                current_lin = []
                for c in old_code:
                    current_lin.append(c)
                    current_lin.append(c ^ g)
                    
        if len(current_lin) < (1 << k):
            # Failed to build full code, skip or use what we have?
            # If we can't build 2^k, we probably can't build a good code from it.
            # But let's try with whatever size we got.
            pass
            
        # 2. Greedy Expansion
        # current_lin is the seed
        current_code = list(current_lin)
        existing_set = set(current_code)
        
        # Calculate min_dist to current_lin for all other words
        # This is expensive: 131k * 256 ops.
        # We can do it in chunks or just loop.
        
        candidates = []
        # Optimization: 
        # We only care about candidates that are "far" from current_lin.
        # If min_dist < MIN_D, they are invalid *initially*.
        # But wait, if we start with a smaller code, some words might be valid.
        
        # Let's just compute min_dist for all x not in existing_set.
        # To speed up, we can use NumPy if current_lin is large.
        # But here current_lin is the linear code.
        
        # We will iterate x.
        # If time is tight, we might skip this for large k.
        
        if time.time() > deadline - 5:
            continue
            
        # Compute min_dist_to_lin
        # We can do this by iterating x and checking against current_lin.
        # 131072 * 256 is ~33M ops. In Python, this is ~1-2 seconds.
        # In NumPy, it's instant.
        
        # Let's use a simple loop first, it might be fast enough.
        # Or we can use the fact that we just want to sort by min_dist.
        
        # Actually, for k=8, we know min_dist <= 3 for all x. So candidates list will be empty.
        # So for k=8, we skip expansion.
        
        if len(current_lin) == 256:
            # We know no expansion is possible
            if len(current_lin) > best_size:
                best_code = current_lin
                best_size = len(current_lin)
            continue
            
        # For k < 8, we try to expand.
        
        # Compute min_dist for all x not in existing_set
        # Vectorized approach:
        # Convert current_lin to a numpy array of uint32
        lin_arr = np.array(current_lin, dtype=np.uint32)
        
        # We need to check all x from 0 to max_val-1
        # This is a big matrix multiplication if we do it all at once (131k x 256).
        # 131072 * 256 = 33M entries. 33M * 4 bytes = 132 MB. This is acceptable.
        
        # However, creating the XOR matrix is expensive.
        # Let's do it in chunks to save memory and allow early exit if possible?
        # Or just do it for all valid x.
        
        # Identify valid x indices
        valid_mask = np.ones(max_val, dtype=bool)
        valid_mask[list(existing_set)] = False
        valid_indices = np.where(valid_mask)[0]
        
        # If no valid indices, skip
        if len(valid_indices) == 0:
            continue
            
        # Process in chunks to manage memory and time
        chunk_size = 10000
        all_candidate_scores = []
        
        for i in range(0, len(valid_indices), chunk_size):
            chunk_idx = valid_indices[i:i+chunk_size]
            chunk_vals = chunk_idx.astype(np.uint32)
            
            # Compute XOR with all lin_arr elements
            # chunk_vals shape (N,), lin_arr shape (M,)
            # We want (N, M) matrix of XORs
            xor_matrix = chunk_vals[:, np.newaxis] ^ lin_arr[np.newaxis, :]
            
            # Compute weights for this matrix
            # We can use the precomputed weights array
            dist_matrix = weights[xor_matrix]
            
            # Find min distance for each candidate in this chunk
            min_dists_chunk = np.min(dist_matrix, axis=1)
            
            # Filter candidates with min_dist >= MIN_D
            valid_in_chunk = min_dists_chunk >= MIN_D
            if np.any(valid_in_chunk):
                valid_chunk_idx = chunk_idx[valid_in_chunk]
                valid_chunk_scores = min_dists_chunk[valid_in_chunk]
                all_candidate_scores.extend(zip(valid_chunk_scores, valid_chunk_idx))
                
            if time.time() > deadline - 5:
                break

        # Convert to list of tuples (score, x)
        # all_candidate_scores is already list of tuples (score, x)
        candidates = all_candidate_scores
                
        if not candidates:
            if len(current_lin) > best_size:
                best_code = current_lin
                best_size = len(current_lin)
            continue
            
        # Sort candidates
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Greedy Add
        # Sort candidates by score (min_dist) descending
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        # Maintain a numpy array of current_code for vectorized checks
        # This is faster than looping in Python for each candidate
        code_arr = np.array(current_code, dtype=np.uint32)
        
        for min_d, x in candidates:
            if time.time() > deadline - 2:
                break
                
            # Check x against all codewords in code_arr
            xor_vals = x ^ code_arr
            dists = weights[xor_vals]
            
            if np.min(dists) >= MIN_D:
                current_code.append(x)
                existing_set.add(x)
                # Update code_arr
                # Appending to numpy array is expensive. 
                # We can batch updates or just convert at the end?
                # But we need it for the next check.
                # Let's just convert if it grows significantly?
                # Or maintain a list and convert periodically?
                # For simplicity and speed, let's convert every time we add? 
                # No, that's O(N) per add. 
                # Better: maintain list, and only vectorize if list is large?
                # Actually, the number of additions is small (maybe 2-3 for 258).
                # So converting every time is fine.
                code_arr = np.array(current_code, dtype=np.uint32)
                
        if len(current_code) > best_size:
            best_code = current_code
            best_size = len(current_code)
            
    # If we didn't find anything (should not happen as k=5+ works), return zero
    if not best_code:
        best_code = [0]
        
    return best_code
