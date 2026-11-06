import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 17
    MIN_DIST = 6
    
    # 1. Generate initial RM code (32 words)
    rm_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            rm_words.append(word)
            
    # Convert to NumPy array (K, 17)
    # We need to handle the 17th bit. The RM code is on 16 bits.
    # The task says "extended by two zero bits" in the seed comment, but the seed only uses 16 bits for the word construction?
    # "First-order Reed-Muller code on 16 coordinates, extended by two zero bits."
    # The seed code:
    # word |= bit << point (point in 0..15)
    # So bits 16 and 17 are 0.
    # So the RM words are 17-bit words with the last two bits 0.
    
    codewords = np.array(rm_words, dtype=np.uint32)
    
    # 2. Generate all candidates
    # We can generate all 2^17 integers
    all_candidates = np.arange(2**N, dtype=np.uint32)
    
    # We need to compare these with codewords.
    # To compute Hamming distance, we need to convert to bits or use XOR and popcount.
    # NumPy doesn't have a direct popcount for uint32 in all versions, but we can use a lookup table or bit manipulation.
    # Alternatively, convert to bits (K, 17) and (M, 17).
    
    # Let's convert codewords to bits
    def int_to_bits(arr, N):
        # arr: (K,) uint32
        # return: (K, N) uint8
        bits = np.zeros((len(arr), N), dtype=np.uint8)
        for j in range(N):
            bits[:, j] = (arr >> j) & 1
        return bits
        
    codeword_bits = int_to_bits(codewords, N)
    
    # Active candidates
    active_indices = np.arange(len(all_candidates))
    
    # We need to check if a candidate is in codewords?
    # The RM words are in all_candidates. We should remove them from active or handle them.
    # Since we start with RM words, we should remove them from active to avoid duplicates.
    # Create a set of current codewords for fast lookup?
    # Or just mark them as inactive.
    
    # Let's maintain a list of codewords and a boolean mask for active candidates.
    # To speed up distance calculation, we will keep codewords as bits.
    
    current_codewords_bits = codeword_bits
    num_codewords = len(current_codewords_bits)
    
    # Remove initial codewords from active
    # all_candidates contains all 0..2^17-1
    # We can create a set of current codewords (as ints)
    current_set = set(rm_words)
    active_mask = np.array([c not in current_set for c in all_candidates], dtype=bool)
    active_indices = np.where(active_mask)[0]
    
    while time.time() < deadline:
        if len(active_indices) == 0:
            break
            
        # Pick a candidate. 
        # Strategy: Pick the one with the largest minimum distance to current codewords?
        # Or just the first one?
        # To be efficient, we can sample a few or pick the first valid one.
        # Let's pick the first one in active_indices that is valid.
        # But we need to check validity.
        
        # To find a valid candidate, we can iterate through active_indices.
        # However, checking one candidate against all codewords is fast.
        # Checking all active candidates against all codewords is slow.
        
        # Let's try to add the first active candidate.
        # If it's valid, add it. If not, remove it from active.
        # This is a simple greedy.
        
        cand_idx = active_indices[0]
        cand_int = all_candidates[cand_idx]
        
        # Check distance to all current codewords
        cand_bit = (cand_int >> np.arange(N)) & 1
        # current_codewords_bits is (K, N)
        # dists = np.bitwise_xor(cand_bit[None, :], current_codewords_bits).sum(axis=1)
        
        # Vectorized check for this candidate
        # cand_bit is (N,)
        # xor = np.bitwise_xor(cand_bit, current_codewords_bits) -> (K, N)
        dists = np.bitwise_xor(cand_bit, current_codewords_bits).sum(axis=1)
        
        if np.min(dists) >= MIN_DIST:
            # Valid! Add to codewords
            current_codewords_bits = np.vstack([current_codewords_bits, cand_bit[None, :]])
            current_set.add(cand_int)
            
            # Remove from active all candidates that are too close to this new codeword
            # Compute distance from cand_int to all active candidates
            # active_cands = all_candidates[active_indices]
            # This is a large array. 
            # dists_active = np.bitwise_xor(cand_bit[None, :], int_to_bits(active_cands, N)).sum(axis=1)
            # This is expensive if we convert bits every time.
            
            # Better: Keep active candidates as integers.
            # We need a fast way to compute distance between one int and many ints.
            # We can use a precomputed table?
            # Or just use the bit conversion for the batch.
            
            # Let's optimize the removal step.
            # We have `active_indices`.
            # We want to keep only those with dist >= MIN_DIST.
            # We can do this in chunks to avoid memory blowup.
            
            new_active_indices = []
            chunk_size = 10000
            for i in range(0, len(active_indices), chunk_size):
                chunk_idx = active_indices[i:i+chunk_size]
                chunk_cands = all_candidates[chunk_idx]
                # Convert chunk to bits
                chunk_bits = int_to_bits(chunk_cands, N)
                # Compute distances to cand_bit
                dists_chunk = np.bitwise_xor(cand_bit[None, :], chunk_bits).sum(axis=1)
                valid_mask = dists_chunk >= MIN_DIST
                if np.any(valid_mask):
                    new_active_indices.append(chunk_idx[valid_mask])
                    
            if new_active_indices:
                active_indices = np.concatenate(new_active_indices)
            else:
                active_indices = np.array([], dtype=int)
                
        else:
            # Invalid, remove from active
            active_indices = active_indices[1:]
            
    # Convert back to list of ints
    # current_codewords_bits is (K, N)
    result = []
    for i in range(len(current_codewords_bits)):
        w = 0
        for j in range(N):
            if current_codewords_bits[i, j]:
                w |= (1 << j)
        result.append(w)
        
    return result
