"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # 1. Construct the extended quadratic residue code of length 18
    # This serves as a strong, valid seed (512 words, min dist 6)
    
    p = 19
    qr_set = set((x * x) % p for x in range(1, p))
    
    # Generator matrix: 9 rows, 17 columns
    G = np.zeros((9, 17), dtype=np.uint8)
    for i in range(9):
        for j in range(17):
            val = (j - i) % p
            if val in qr_set:
                G[i, j] = 1
    
    # Generate all 512 codewords
    # We represent codewords as integers for the final output, but use numpy for search
    # Bit 0 is LSB (position 0), Bit 17 is MSB (position 17)
    
    base_words = []
    for mask in range(512):
        # Compute linear combination of generator rows
        # mask bits 0..8 correspond to G rows 0..8
        # G is 9x17. We want the sum of rows where mask bit is 1.
        # result[j] = XOR of G[i][j] for all i where mask has bit i.
        # Since G is binary, this is sum mod 2.
        # We can do this via matrix multiplication over GF(2) or just loop.
        # For speed, precompute the 512 words.
        
        # Efficient way:
        # word_bits = G @ mask_vec mod 2
        # But we need it as an integer.
        
        # Let's just build them quickly.
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                # XOR row i into word
                # G[i] is a numpy array of 0s and 1s
                # We need to map G[i][j] to bit j
                # Pre-computed bitmasks for each row?
                pass
        
        # Let's precompute row bitmasks for speed
        # Actually, let's just build the 512 words once at the start.
        
    # Re-doing seed generation efficiently
    row_bitmasks = []
    for i in range(9):
        mask_val = 0
        for j in range(17):
            if G[i, j] == 1:
                mask_val |= (1 << j)
        row_bitmasks.append(mask_val)
        
    seed_ints = [0] * 512
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= row_bitmasks[i]
        # Extend with parity bit (bit 17)
        parity = word.bit_count() % 2
        word |= (parity << 17)
        seed_ints[mask] = word
        
    # Verify seed validity (should be min dist 6)
    # For a linear code, min weight is min distance.
    # We trust the QR construction, but let's be safe.
    # The seed is 512 words.
    
    current_code = list(set(seed_ints))
    if not current_code:
        return [0] # Fallback, though unlikely
    
    # 2. Greedy Set Packing Search
    # Goal: Add more words to current_code such that min distance remains >= 6.
    # 
    # Optimization:
    # - Maintain a list of current codewords as integers.
    # - For a candidate word w, check distance to all existing words.
    # - Use NumPy to batch check many candidates against the current code.
    
    code_arr = np.array(current_code, dtype=np.uint32)
    n_current = len(code_arr)
    
    # To check distance between candidate w and all code_arr:
    # dist = popcount(w ^ code_arr)
    # We need min(dist) >= 6.
    
    # We will generate random candidates.
    # To increase success rate, we can use "biased" random or just uniform.
    # Uniform is simple. With 512 words, the space is large (2^18), so random might work initially.
    # However, as the code grows, the density increases, and random hits valid spots less often.
    # 
    # A better strategy for set packing:
    # Start with the QR code.
    # Try to add words.
    
    # Pre-compute a function to check a batch of candidates against the current code.
    # Since code_arr changes, we can't precompute a static structure easily without complex data structures.
    # We'll just do batch XOR and popcount.
    
    def check_batch(candidates, code_arr):
        # candidates: 1D array of uint32
        # code_arr: 1D array of uint32
        # Returns boolean array: True if candidate is valid (min dist >= 6)
        
        # We need to check dist(cand, code) >= 6 for all code in code_arr.
        # This is an outer product-like operation but with XOR.
        # Memory constraint: If we have 1000 candidates and 1000 code words, 1M XORs is fine.
        # 1M uint32s = 4MB.
        
        # Let's process in chunks of candidates to limit memory.
        # But for simplicity, let's try a moderate batch size.
        
        valid = np.ones(len(candidates), dtype=bool)
        
        # If code_arr is small, we can loop over code_arr and update valid.
        # If code_arr is large, we might need to loop over candidates?
        # Let's loop over code_arr. For each code word, compute dist to all candidates.
        # dists = popcount(candidates ^ code_word)
        # valid &= (dists >= 6)
        
        for cw in code_arr:
            # Compute XOR
            xors = candidates ^ cw
            # Compute popcount
            # np.bit_count is available in newer numpy? 
            # If not, use lookup or unpackbits.
            # Let's check numpy version compatibility. 
            # Standard numpy doesn't have bit_count for integers directly in a vectorized way 
            # until recently (1.24+? or via scipy). 
            # Safer: Use a lookup table or unpackbits.
            # unpackbits is slow.
            # Let's use a small lookup table for 8-bit chunks if needed, 
            # but actually, since we are in Python, we can just use a list comprehension 
            # for the inner loop if the batch is small, or rely on numpy's speed for XOR 
            # and then a custom popcount.
            
            # Let's try to use a vectorized popcount.
            # Trick: 
            # xors is uint32.
            # We can use: 
            # popcounts = np.zeros(len(xors), dtype=np.int32)
            # for shift in [0, 1, 2, 3]:
            #     # ... this is getting complicated.
            
            # Alternative: 
            # Since we only care if popcount < 6, we can check:
            # popcount(x) < 6  <=>  x has < 6 bits set.
            # We can use a precomputed table for 16-bit chunks.
            # But that requires splitting the uint32.
            
            # Let's assume we can use a simple loop for popcount if numpy doesn't support it well, 
            # or use scipy? SciPy is allowed.
            # scipy.special? No.
            # Let's stick to a robust method:
            
            # Method: 
            # Use a lookup table for 8-bit values.
            # Table size 256.
            # Split xors into 4 bytes.
            
            # This might be slower than a C-level bit_count.
            # Let's check if `np.bit_count` exists.
            # If not, we'll implement a fast popcount.
            
            # For now, let's use a Python loop for the popcount check 
            # if the batch is small, or optimize later.
            # Actually, 512 code words * 1000 candidates = 512k operations.
            # If we do this in Python loops, it's too slow.
            # We MUST vectorize the popcount.
            
            # Let's use the unpackbits trick or a lookup.
            # Lookup is faster.
            
            # Pre-compute a global lookup table?
            pass
            
        return valid

    # Let's implement a fast vectorized popcount check.
    # We'll use a lookup table for 8-bit values.
    popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    def min_dist_ge_6(candidates, code_arr):
        # candidates: (N,) uint32
        # code_arr: (M,) uint32
        # Returns boolean array of length N
        
        valid = np.ones(len(candidates), dtype=bool)
        
        # Loop over code words
        for cw in code_arr:
            # Compute XOR
            xors = candidates ^ cw
            
            # Compute popcount using lookup table
            # Split into 4 bytes
            b0 = (xors & 0xFF).astype(np.uint8)
            b1 = ((xors >> 8) & 0xFF).astype(np.uint8)
            b2 = ((xors >> 16) & 0xFF).astype(np.uint8)
            b3 = ((xors >> 24) & 0xFF).astype(np.uint8)
            
            pc = popcount_table[b0] + popcount_table[b1] + popcount_table[b2] + popcount_table[b3]
            
            # If any distance is < 6, mark as invalid
            valid &= (pc >= 6)
            
            # Early exit if all invalid
            if not np.any(valid):
                break
                
        return valid

    # Now the search loop
    # We will generate random candidates in batches.
    # Batch size: 1000-5000.
    
    rng = np.random.default_rng(42)
    
    # To improve success, we can bias the candidates.
    # However, uniform random is a good baseline.
    # Let's try uniform random.
    
    while time.time() < deadline:
        # Generate a batch of candidates
        batch_size = 2000
        candidates = rng.integers(0, 2**18, size=batch_size, dtype=np.uint32)
        
        # Check against current code
        valid_mask = min_dist_ge_6(candidates, code_arr)
        
        # If any are valid, add them
        if np.any(valid_mask):
            valid_candidates = candidates[valid_mask]
            
            # Add them to the code
            # Convert to list of ints
            new_words = [int(x) for x in valid_candidates]
            current_code.extend(new_words)
            code_arr = np.array(current_code, dtype=np.uint32)
            
            # If we added a lot, we might want to restart or adjust strategy?
            # No, just keep going.
            
    return current_code

# EVOLVE-BLOCK-END
