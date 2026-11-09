def construct():
    import time
    start = time.time()
    DEADLINE = 85  # Safety margin

    # 1. Generate [18,9,6] QR code
    N_ext = 18
    Q = [1, 2, 4, 8, 9, 13, 15, 16]
    
    # Base vector
    base = [0] * N_ext
    for q in Q:
        base[q] = 1
        
    # Generate 9 basis vectors (cyclic shifts)
    basis = []
    for i in range(9):
        row = [0] * N_ext
        for j in range(N_ext):
            # Shift base by i
            # (j - i) mod 18
            row[j] = base[(j - i) % N_ext]
        basis.append(row)
        
    # Generate all 512 codewords of [18,9,6]
    ext_words = set()
    for mask in range(512):
        word = [0] * N_ext
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(N_ext):
                    word[j] ^= basis[i][j]
        # Convert to int
        w_int = 0
        for j in range(N_ext):
            w_int |= word[j] << j
        ext_words.add(w_int)
        
    # 2. Shorten to [17,8,6]
    # Keep words where bit 17 is 0
    code = set()
    for w in ext_words:
        if (w >> 17) & 1 == 0:
            # Remove bit 17
            short_w = w & ((1 << 17) - 1)
            code.add(short_w)
            
    # 3. Greedy expansion
    code_list = list(code)
    
    # Optimization: Precompute list for faster iteration?
    # We will iterate candidates 0 to 2^17-1
    # For each candidate, check against code_list
    
    # To speed up, we can sort candidates or just iterate.
    # If we add a word, we append to code_list.
    
    # We need to be careful with time.
    # 131072 candidates.
    # Initial check: 256 words.
    # If we find 10 more, check: 266 words.
    
    # Let's iterate
    for c in range(1 << 17):
        if time.time() - start > DEADLINE:
            break
            
        if c in code:
            continue
            
        # Check distance
        valid = True
        for w in code_list:
            if (c ^ w).bit_count() < 6:
                valid = False
                break
                
        if valid:
            code.add(c)
            code_list.append(c)
            
    return list(code)
