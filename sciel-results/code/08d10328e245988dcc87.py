import numpy as np

def construct():
    N = 17
    MIN_DIST = 6
    
    # 1. Generate the base linear code (256 words)
    qr = [1, 2, 4, 8, 9, 13, 15, 16]
    h = np.zeros(N, dtype=np.uint8)
    for r in qr:
        h[r] = 1
        
    G = np.zeros((8, N), dtype=np.uint8)
    for i in range(8):
        G[i] = np.roll(h, -i)
        
    base_codewords = []
    for mask in range(256):
        c = np.zeros(N, dtype=np.uint8)
        for i in range(8):
            if mask & (1 << i):
                c = (c + G[i]) % 2
        val = 0
        for j in range(N):
            val |= (c[j] << j)
        base_codewords.append(val)
        
    # Convert to numpy array of bits
    base_bits = np.zeros((256, N), dtype=np.uint8)
    for i in range(256):
        for j in range(N):
            base_bits[i, j] = (base_codewords[i] >> j) & 1
            
    # 2. Identify candidates
    all_numbers = np.arange(2**N, dtype=np.int64)
    is_base = np.zeros(2**N, dtype=bool)
    is_base[base_codewords] = True
    candidates = all_numbers[~is_base]
    
    # Prepare candidate bits
    cand_bits = np.zeros((len(candidates), N), dtype=np.uint8)
    for i in range(len(candidates)):
        val = candidates[i]
        for j in range(N):
            cand_bits[i, j] = (val >> j) & 1
            
    best_size = len(base_codewords)
    best_code = base_codewords.copy()
    
    num_trials = 20 # Increase trials to find better packing
    block_size = 1000
    
    for _ in range(num_trials):
        shuffled_indices = np.random.permutation(len(candidates))
        
        current_code_bits = base_bits.copy()
        current_code_vals = base_codewords.copy()
        
        # Process in blocks
        for start in range(0, len(shuffled_indices), block_size):
            end = min(start + block_size, len(shuffled_indices))
            block_indices = shuffled_indices[start:end]
            block_bits = cand_bits[block_indices]
            block_vals = candidates[block_indices]
            
            # Calculate min distance from each candidate in block to current code
            # block_bits: (B, N)
            # current_code_bits: (K, N)
            # We need (B, K) distance matrix
            # XOR: (B, 1, N) ^ (1, K, N) -> (B, K, N)
            # Sum over N -> (B, K)
            
            # This creates a large intermediate array. 
            # B=1000, K~250. 1000*250*17 = 4.25 million elements. 
            # uint8 is 4.25 MB. This is fine.
            
            xor = block_bits[:, np.newaxis, :] ^ current_code_bits[np.newaxis, :, :]
            dists = np.sum(xor, axis=2)
            min_dists = np.min(dists, axis=1)
            
            # Find candidates in this block that are valid
            valid_mask = min_dists >= MIN_DIST
            valid_indices_in_block = np.where(valid_mask)[0]
            
            if len(valid_indices_in_block) == 0:
                continue
                
            # Process valid candidates one by one to update the code
            # We should process them in an order that might maximize future additions?
            # Random order is fine.
            # But we need to re-check validity against the *updated* code.
            
            for idx_in_block in valid_indices_in_block:
                bit = block_bits[idx_in_block]
                val = block_vals[idx_in_block]
                
                # Re-check against current_code_bits (which might have grown)
                # If current_code_bits is large, this is expensive.
                # But valid_indices_in_block is small.
                
                xor_single = current_code_bits ^ bit
                dists_single = np.sum(xor_single, axis=1)
                if np.min(dists_single) >= MIN_DIST:
                    current_code_bits = np.vstack([current_code_bits, bit])
                    current_code_vals.append(val)
                    
        if len(current_code_vals) > best_size:
            best_size = len(current_code_vals)
            best_code = current_code_vals.copy()
            
    return best_code
