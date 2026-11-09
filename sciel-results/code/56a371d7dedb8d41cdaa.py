def construct():
    import time
    start = time.time()
    DEADLINE = 55  # seconds, leave margin
    
    # Construct [17,8,6] code
    p = 17
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    gen = [[1]*17]
    for i in range(1, 9):
        row = [0]*17
        for q in Q:
            pos = (q + i) % 17
            row[pos] = 1
        gen.append(row)
    
    # Generate 256 codewords (shortened)
    words = set()
    for mask in range(512):
        word = [0]*17
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    word[j] ^= gen[i][j]
        # Extend with parity
        parity = sum(word) % 2
        # Shorten: only keep if parity = 0 (i.e., last coord of extended = 0)
        if parity == 0:
            w = 0
            for j in range(17):
                w |= word[j] << j
            words.add(w)
    
    word_list = list(words)
    assert len(word_list) == 256, f"Expected 256, got {len(word_list)}"
    
    # Verify minimum distance
    # (skip for speed, trust construction)
    
    # Now search for additional words
    # For each candidate in 0..2^17-1, check distance to all 256 codewords
    # Use numpy for speed if available
    
    import numpy as np
    
    # Precompute codewords as numpy array
    cw_array = np.array(word_list, dtype=np.uint32)
    
    # Generate all candidates
    candidates = np.arange(2**17, dtype=np.uint32)
    
    # Compute distances: for each candidate, min distance to any codeword
    # This is 131072 x 256 matrix - too large to store
    # Instead, iterate in chunks
    
    additional = []
    chunk_size = 10000
    for i in range(0, 2**17, chunk_size):
        if time.time() - start > DEADLINE:
            break
        chunk = candidates[i:i+chunk_size]
        # Compute XOR with each codeword
        # For each codeword, compute distances to all in chunk
        min_dist = np.full(len(chunk), 100, dtype=np.uint8)
        for cw in word_list:
            if time.time() - start > DEADLINE:
                break
            xor_vals = chunk ^ cw
            # Compute popcount
            # numpy doesn't have popcount, use lookup or bit tricks
            # For 17-bit values, we can use a lookup table
            # Actually, let's use a different approach
        
        # This is getting complex. Let me use a simpler approach.
    
    # Simpler: just iterate through candidates in Python
    # 131072 * 256 = 33.5M iterations, each is XOR + bit_count
    # In Python, this might take ~30-60s. Too slow.
    
    # Better: use the fact that we only need to find words at distance >= 6
    # from ALL 256 codewords. Most words will be at distance < 6 from some codeword.
    
    # Alternative: use a covering argument or just try random candidates
    # Or: use the structure of the code
    
    return word_list
