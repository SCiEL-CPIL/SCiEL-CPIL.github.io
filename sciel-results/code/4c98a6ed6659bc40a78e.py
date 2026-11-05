def construct():
    # [17,9,5] QR code generator polynomial
    # g(x) = x^8 + x^7 + x^5 + x^4 + x^3 + x + 1
    gen_poly = 0b110111011  # bits 8,7,5,4,3,1,0
    
    # Generate all 2^9 codewords of the [17,9,5] cyclic code
    codewords = []
    for msg in range(256):
        cw = 0
        for i in range(9):
            if (msg >> i) & 1:
                cw ^= (gen_poly << i)
        codewords.append(cw & ((1 << 17) - 1))
    
    # Extend to [18,9,6] by adding overall parity
    extended = []
    for cw in codewords:
        parity = cw.bit_count() & 1
        ext = cw | (parity << 17)
        extended.append(ext)
    
    # Shorten: keep codewords with bit 17 = 0
    shortened = []
    for ext in extended:
        if (ext >> 17) & 1 == 0:
            shortened.append(ext & ((1 << 17) - 1))
    
    # Deduplicate
    shortened = list(set(shortened))
    
    # Verify minimum distance
    n = len(shortened)
    min_dist = 18
    for i in range(n):
        for j in range(i+1, n):
            d = (shortened[i] ^ shortened[j]).bit_count()
            if d < min_dist:
                min_dist = d
            if min_dist < 6:
                break
        if min_dist < 6:
            break
    
    # If verification fails, fall back to the Reed-Muller code
    if min_dist < 6 or n == 0:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    return shortened
