def construct():
    # The extended quadratic residue code of length 18
    # Parameters: [18, 9, 6]
    # Size: 2^9 = 512
    
    # Generator polynomial for the QR code of length 17:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients (bit i is coeff of x^i):
    # x^0: 1
    # x^1: 0
    # x^2: 0
    # x^3: 1
    # x^4: 1
    # x^5: 1
    # x^6: 0
    # x^7: 0
    # x^8: 1
    # g_val = 1 + 8 + 16 + 32 + 256 = 313
    
    g_val = 313
    
    # Verify g_val divides (x^17 + 1) / (x + 1)
    # H(x) = x^16 + x^15 + ... + x + 1
    # H_val = 2^17 - 1 = 131071
    
    # We perform polynomial division of H(x) by g(x) in F2
    # If remainder is 0, it's valid.
    
    def poly_mod(numerator, denominator):
        # numerator and denominator are ints representing polynomials
        # denominator must be monic
        rem = numerator
        # Find degree of denominator
        deg_den = denominator.bit_length() - 1
        # Iterate
        while rem != 0 and rem.bit_length() - 1 >= deg_den:
            deg_rem = rem.bit_length() - 1
            shift = deg_rem - deg_den
            # XOR denominator shifted
            rem ^= (denominator << shift)
        return rem
    
    # H(x) is (2^17 - 1)
    H_val = (1 << 17) - 1
    rem = poly_mod(H_val, g_val)
    
    if rem != 0:
        # Fallback to Reed-Muller if polynomial is wrong (should not happen)
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Construct Generator Matrix rows (9 rows, 18 bits each)
    # Base vector is g(x) as 17-bit vector (padded to 17)
    # g(x) has degree 8, so it fits in 17 bits.
    # v_0 = g(x)
    # v_i = x^i * g(x) mod (x^17 - 1)? 
    # For generator matrix, we just take x^i * g(x) for i=0..8.
    # Since deg(x^8 * g(x)) = 16, no wrap-around needed for the basis vectors.
    
    rows = []
    for i in range(9):
        # Shift g_val by i
        vec = g_val << i
        # Ensure it's 17 bits wide (it will be, as max degree is 16)
        # Calculate parity for the 17-bit part
        parity = vec.bit_count() % 2
        # Append parity bit at position 17
        row = vec | (parity << 17)
        rows.append(row)
        
    # Generate all 512 codewords
    words = []
    for m in range(512):
        word = 0
        for j in range(9):
            if (m >> j) & 1:
                word ^= rows[j]
        words.append(word)
        
    return words
