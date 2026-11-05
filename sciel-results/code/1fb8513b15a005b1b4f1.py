def construct():
    import time
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    N = 18
    MIN_DISTANCE = 6
    
    # Quadratic residues mod 17
    p = 17
    Q = set((i * i) % p for i in range(1, p))
    N_res = set(range(p)) - Q - {0}
    
    # Construct Generator Matrix (9 rows, 18 bits)
    G = []
    
    # Row 0: All ones
    G.append((1 << N) - 1)
    
    # Rows 1-8
    for r in range(1, 9):
        word = 1 << (N - 1)  # Extension bit (bit 17) is 1
        for c in range(p):
            x = (r * c) % p
            if x == 0:
                bit = 0
            elif x in Q:
                bit = 0
            else:
                bit = 1
            if bit:
                word |= (1 << c)
        G.append(word)
        
    # Generate all codewords
    words = [0]
    for mask in range(1, 1 << len(G)):
        w = 0
        for i in range(len(G)):
            if mask & (1 << i):
                w ^= G[i]
        words.append(w)
        
    # Verify min distance
    # Optimization: Check only a subset if time is tight, but 512^2 is ~260k checks, which is fast.
    min_dist = N
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < MIN_DISTANCE:
                    break
        if min_dist < MIN_DISTANCE:
            break
            
    if min_dist >= MIN_DISTANCE:
        return words
    else:
        # Fallback to Reed-Muller if construction failed
        words_rm = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words_rm.append(word)
        return words_rm
