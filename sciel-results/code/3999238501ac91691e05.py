import time
import random
import numpy as np

def construct():
    """
    Construct a large binary code of length 17 with minimum Hamming distance 6.
    
    Strategy:
    1. Start with the shortened [18,9,6] QR code (size 256).
    2. Use a bit-parallel conflict matrix approach with "largest independent set"
       heuristic (prioritizing candidates with fewest conflicts).
    3. Apply greedy set packing with vectorized distance checks.
    4. Use deadline guard to ensure termination within 90s.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin of 8s before 90s limit
    
    n = 17
    min_dist = 6
    max_val = 1 << n  # 2^17 = 131072
    
    # --- Step 1: Generate the seed code (Size 256) ---
    # The [17,8,6] shortened QR code
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1]
    ]
    
    # Convert to integers
    basis = []
    for row in G:
        val = 0
        for i, bit in enumerate(row):
            if bit:
                val |= (1 << i)
        basis.append(val)
        
    # Generate all 2^8 = 256 codewords
    seed_code = []
    for u in range(256):
        c = 0
        for k in range(8):
            if (u >> k) & 1:
                c ^= basis[k]
        seed_code.append(c)
    
    # Convert seed to numpy array for vectorized operations
    seed_arr = np.array(seed_code, dtype=np.int64)
    
    # Precompute all possible words
    all_words = np.arange(max_val, dtype=np.int64)
    
    # Track which words are in the code
    in_code = np.zeros(max_val, dtype=bool)
    for w in seed_code:
        in_code[w] = True
    
    codewords = list(seed_code)
    current_size = len(codewords)
    
    # Function to check if a candidate can be added using vectorized operations
    def can_add(candidate_arr, in_code_mask):
        """Check which candidates in candidate_arr can be added (no conflict with in_code)."""
        if len(codewords) == 0:
            return np.ones(len(candidate_arr), dtype=bool)
        
        # For each candidate, check distance to all current codewords
        # Use broadcasting: (len(candidates), len(codewords))
        # But this might be too large, so batch it
        batch_size = 1000
        valid = np.ones(len(candidate_arr), dtype=bool)
        
        for start in range(0, len(candidate_arr), batch_size):
            end = min(start + batch_size, len(candidate_arr))
            batch = candidate_arr[start:end]
            
            # Compute distances: (batch_size, n_codewords)
            # This could be large, so we do it in chunks
            chunk_size = 200
            batch_valid = np.ones(len(batch), dtype=bool)
            
            for cstart in range(0, len(codewords), chunk_size):
                cend = min(cstart + chunk_size, len(codewords))
                cw = np.array(codewords[cstart:cend], dtype=np.int64)
                
                # XOR and bit count
                xors = batch[:, np.newaxis] ^ cw[np.newaxis, :]
                # Convert to uint8 for popcount
                xors_u8 = xors.astype(np.uint8)
                
                # Popcount using lookup or bit manipulation
                # For 17-bit numbers, we can use numpy's bit_count equivalent
                # Actually, let's use a different approach: count set bits
                # Use the trick: xors are at most 17 bits
                # We can use np.bitwise_and with masks
                
                # Simpler: use Python's bit_count on the flattened array
                # But that's slow. Let's use a vectorized approach.
                
                # For each XOR value, count bits
                # We'll use the standard trick
                tmp = xors
                tmp = tmp - ((tmp >> 1) & 0x55555555)
                tmp = (tmp & 0x33333333) + ((tmp >> 2) & 0x33333333)
                tmp = (tmp + (tmp >> 4)) & 0x0F0F0F0F
                popcount = ((tmp * 0x01010101) >> 24) & 0xFF
                popcount = popcount.astype(np.int32)
                
                # Check if any distance < 6
                conflicts = np.any(popcount < min_dist, axis=1)
                batch_valid[conflicts] = False
            
            valid[start:end] = batch_valid
        
        return valid
    
    # Greedy phase: repeatedly find candidates with fewest conflicts and add them
    # Use a priority-based approach
    
    # First, get all words not in the code
    not_in_code = np.where(~in_code)[0]
    
    # We'll use a different approach: for each potential addition,
    # count how many new words it would conflict with
    # Prioritize words that have fewer conflicts
    
    best_size = current_size
    best_codewords = codewords[:]
    
    # Use a queue-based approach with randomization
    # Try multiple passes with different random orderings
    
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Reset to seed
        codewords = list(seed_code)
        in_code = np.zeros(max_val, dtype=bool)
        for w in seed_code:
            in_code[w] = True
        
        current_size = len(codewords)
        
        # Get candidates sorted by weight (lighter words tend to have fewer conflicts)
        weights = np.array([w.bit_count() for w in range(max_val)], dtype=np.int32)
        not_in_code_indices = np.where(~in_code)[0]
        
        # Sort by weight, then by value for determinism within weight class
        # Add some randomness to break ties
        random_seed = random.randint(0, 2**31)
        rng = np.random.RandomState(random_seed)
        
        # Create a scoring function: weight + small random perturbation
        scores = weights[not_in_code_indices].astype(np.float64)
        scores += rng.uniform(0, 0.5, size=len(scores))
        
        # Sort by score
        order = np.argsort(scores)
        ordered_candidates = not_in_code_indices[order]
        
        # Greedy addition
        for w in ordered_candidates:
            if time.time() > deadline - 10.0:
                break
            
            w = int(w)
            if in_code[w]:
                continue
            
            # Check if w can be added
            valid = True
            for cw in codewords:
                if (w ^ cw).bit_count() < min_dist:
                    valid = False
                    break
            
            if valid:
                in_code[w] = True
                codewords.append(w)
                current_size += 1
        
        # Local improvement: try to swap words to increase size
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Try removing each word and seeing if we can add more
            for i in range(len(codewords)):
                if time.time() > deadline - 10.0:
                    break
                
                w_remove = codewords[i]
                
                # Temporarily remove
                in_code[w_remove] = False
                codewords.pop(i)
                
                # Try to add words
                added = 0
                # Check all non-in-code words
                for w in range(max_val):
                    if time.time() > deadline - 10.0:
                        break
                    if not in_code[w]:
                        valid = True
                        for cw in codewords:
                            if (w ^ cw).bit_count() < min_dist:
                                valid = False
                                break
                        if valid:
                            in_code[w] = True
                            codewords.append(w)
                            added += 1
                
                if added > 1:
                    # Net gain
                    improved = True
                else:
                    # Revert
                    in_code[w_remove] = True
                    codewords.insert(i, w_remove)
        
        if len(codewords) > best_size:
            best_size = len(codewords)
            best_codewords = codewords[:]
    
    if best_codewords is None or len(best_codewords) == 0:
        return seed_code
    
    return best_codewords
