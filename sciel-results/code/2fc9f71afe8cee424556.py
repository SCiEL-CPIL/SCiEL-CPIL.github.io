def construct():
    # Generator polynomial bits for [17,8,5] QR code
    # g(x) = 1 + x^3 + x^6 + x^7 + x^9
    g_offsets = [0, 3, 6, 7, 9]
    
    # Generate basis for [17,8,5] code
    basis_17 = []
    for j in range(8):
        poly = 0
        for k in g_offsets:
            bit_pos = (j + k) % 17
            poly ^= (1 << bit_pos)
        basis_17.append(poly)
        
    # Extend to [18,8,6] by adding parity bit at position 17
    basis_18 = []
    for b in basis_17:
        parity = b.bit_count() % 2
        basis_18.append(b | (parity << 17))
        
    # All-ones vector of length 18
    all_ones_18 = (1 << 18) - 1
    
    final_codewords = []
    
    # Generate all 2^9 = 512 codewords of the [18,9,6] code
    for i in range(512):
        cw = 0
        # XOR basis vectors
        for j in range(8):
            if (i >> j) & 1:
                cw ^= basis_18[j]
        if (i >> 8) & 1:
            cw ^= all_ones_18
            
        # Shorten by coordinate 0: keep only if bit 0 is 0
        if (cw & 1) == 0:
            # Remove bit 0
            final_codewords.append(cw >> 1)
            
    return final_codewords
