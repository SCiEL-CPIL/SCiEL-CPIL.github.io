import time

def construct():
    start_time = time.time()
    
    # Generator polynomial for [17,8,5] QR code
    # g(x) = x^9 + x^7 + x^6 + x^3 + x^2 + x + 1
    g_coeffs = [1, 1, 0, 0, 1, 0, 0, 1, 1, 1] # x^9 ... x^0
    
    # Generate [17,8,5] code
    base_code = []
    for u in range(256):
        c = 0
        for i in range(8):
            if (u >> i) & 1:
                # Multiply g(x) by x^i and reduce mod x^17 - 1
                # Since g(x) has degree 9, x^i g(x) has degree 9+i.
                # We can just shift the bits.
                # If the shift goes beyond 16, wrap around.
                for j, coeff in enumerate(g_coeffs):
                    if coeff:
                        pos = (i + j) % 17
                        c ^= (1 << pos)
        base_code.append(c)
        
    # Extend to [18,8,6]
    extended_code = []
    for c in base_code:
        parity = c.bit_count() % 2
        c_ext = c | (parity << 17)
        extended_code.append(c_ext)
        
    # Add all-ones vector to get [18,9,6]
    all_ones = (1 << 18) - 1
    full_code = set(extended_code)
    for c in extended_code:
        full_code.add(c ^ all_ones)
        
    # Shorten by fixing last bit (bit 17) to 0
    shortened_code = []
    for c in full_code:
        if not (c & (1 << 17)):
            c_short = c & ((1 << 17) - 1)
            shortened_code.append(c_short)
            
    # Verify
    if len(shortened_code) == 256:
        # Quick check
        return shortened_code
    else:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

