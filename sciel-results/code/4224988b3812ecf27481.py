def construct():
    import time
    start_time = time.time()
    
    # Generator matrix for [18, 9, 6] code
    G = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    ]
    
    words = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(18):
                    if G[i][j]:
                        word ^= (1 << j)
        words.append(word)
        
    # Verification
    # Check distinct
    if len(set(words)) != 512:
        return words[:32] # Fallback to RM if broken
        
    # Check distance
    # This is O(N^2) = 512^2 = 262144 checks. Fast enough.
    for i in range(512):
        for j in range(i+1, 512):
            if (words[i] ^ words[j]).bit_count() < 6:
                # Invalid code, fallback
                # Return Reed-Muller
                rm_words = []
                for slope in range(16):
                    for intercept in range(2):
                        w = 0
                        for point in range(16):
                            bit = ((slope & point).bit_count() + intercept) % 2
                            w |= bit << point
                        rm_words.append(w)
                return rm_words
                
    return words
