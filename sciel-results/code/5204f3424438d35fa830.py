import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    
    # 1. Initialize with the [17, 8, 6] linear code
    # This part reuses the algebraic construction logic but serves as initialization for the heuristic
    qr = [1, 2, 4, 8, 9, 13, 15, 16]
    h = np.zeros(N, dtype=np.uint8)
    for r in qr:
        h[r] = 1
    
    G = np.zeros((8, N), dtype=np.uint8)
    for i in range(8):
        G[i] = np.roll(h, -i)
    
    codewords = []
    for mask in range(256):
        c = np.zeros(N, dtype=np.uint8)
        for i in range(8):
            if mask & (1 << i):
                c = (c + G[i]) % 2
        val = 0
        for j in range(N):
            if c[j]:
                val |= (1 << j)
        codewords.append(val)
        
    # Convert to numpy array for fast operations
    code_arr = np.array(codewords, dtype=np.uint8)
    # Reshape to (K, 1) for broadcasting? Or just keep as list for now and convert when checking.
    # Let's keep a list of ints for easy management, and convert to array for batch checks.
    
    code_list = list(codewords)
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Helper to convert int to numpy bit vector
    def int_to_vec(val):
        vec = np.zeros(N, dtype=np.uint8)
        for j in range(N):
            if val & (1 << j):
                vec[j] = 1
        return vec

    # Helper to check if a batch of candidates can be added
    def try_add_candidates(candidates):
        # candidates: list of ints
        if not candidates:
            return 0
        # Convert candidates to array (M, N)
        cand_arr = np.array([int_to_vec(c) for c in candidates], dtype=np.uint8)
        
        # Current code array (K, N)
        code_arr = np.array([int_to_vec(c) for c in code_list], dtype=np.uint8)
        
        # Calculate distances
        # XOR: (M, 1, N) ^ (1, K, N) -> (M, K, N)
        # This creates a large intermediate array. 
        # M=1000, K=256, N=17. 1000*256*17 ~ 4.3 million elements. 
        # This is fine for NumPy.
        
        # To save memory/time, we can do it in chunks if M is huge.
        # Let's assume batch size is reasonable (e.g., 500).
        
        xor_mat = cand_arr[:, None, :] ^ code_arr[None, :, :]
        weights = np.sum(xor_mat, axis=2)
        min_weights = np.min(weights, axis=1)
        
        valid_indices = np.where(min_weights >= MIN_DIST)[0]
        
        added = 0
        for idx in valid_indices:
            new_word = candidates[idx]
            # Double check? No, vectorized check is exact.
            code_list.append(new_word)
            added += 1
            
        return added

    # Main Loop
    # Try to add random words
    np.random.seed(42) # Reproducibility? Maybe not needed, but good for debugging.
    
    while time.time() - start_time < time_limit:
        # Generate a batch of random candidates
        batch_size = 1000
        # Generate random 17-bit integers
        candidates = np.random.randint(0, 2**N, size=batch_size).tolist()
        
        # Optional: Filter out candidates that are too close to 0 or 1? 
        # Not necessary, the distance check handles it.
        
        added = try_add_candidates(candidates)
        
        if added > 0:
            continue
            
        # If no additions, try a swap?
        # Simple swap: Remove one, try to add two.
        # This is expensive. Let's just keep trying random additions.
        # If the code is maximal, we will never add.
        # In that case, we should break early to save time?
        # But maybe a different random seed would work? No, if it's maximal, no word fits.
        
        # To avoid infinite loop if maximal, we can limit the number of failed batches.
        # But 60s is short. If we fail 10 times, it's likely maximal.
        
        # Let's add a simple local search move:
        # Pick a random word to remove, generate 200 candidates, see if any 2 fit.
        # This is O(K^2) per pair? No, we check against K-1 words.
        
        # Given the complexity, let's stick to simple greedy addition first.
        # If we are at 256 and can't add, we return 256.
        
        # To improve chances, let's vary the batch size or strategy?
        # Actually, if the linear code is NOT maximal, greedy will find words.
        # If it IS maximal, we need swaps.
        
        # Let's implement a basic swap heuristic if stuck.
        
        if len(code_list) == 256:
             # Try to swap
             # Pick a random index to remove
             idx_to_remove = np.random.randint(0, len(code_list))
             w_out = code_list[idx_to_remove]
             
             # Generate candidates
             swap_candidates = np.random.randint(0, 2**N, size=200).tolist()
             
             # We need to find 2 candidates c1, c2 from swap_candidates such that:
             # 1. dist(c1, c2) >= 6
             # 2. dist(c1, w) >= 6 for all w in code_list \ {w_out}
             # 3. dist(c2, w) >= 6 for all w in code_list \ {w_out}
             
             # This is a sub-problem.
             # We can filter swap_candidates against the rest of the code first.
             rest_code = [w for i, w in enumerate(code_list) if i != idx_to_remove]
             rest_arr = np.array([int_to_vec(w) for w in rest_code], dtype=np.uint8)
             cand_arr = np.array([int_to_vec(c) for c in swap_candidates], dtype=np.uint8)
             
             # Check distances to rest_code
             xor_mat = cand_arr[:, None, :] ^ rest_arr[None, :, :]
             weights = np.sum(xor_mat, axis=2)
             min_weights = np.min(weights, axis=1)
             
             valid_cands = swap_candidates[np.where(min_weights >= MIN_DIST)[0]]
             
             if len(valid_cands) >= 2:
                 # Check pairwise distance among valid_cands
                 # Find a pair with distance >= 6
                 found_pair = False
                 for i in range(len(valid_cands)):
                     for j in range(i+1, len(valid_cands)):
                         c1 = valid_cands[i]
                         c2 = valid_cands[j]
                         if bin(c1 ^ c2).count('1') >= MIN_DIST:
                             # Success!
                             # Remove w_out, add c1, c2
                             code_list.remove(w_out)
                             code_list.append(c1)
                             code_list.append(c2)
                             found_pair = True
                             break
                     if found_pair:
                         break
                         
    return code_list
