import time

N = 18
MIN_DISTANCE = 6

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    # The Extended Quadratic Residue Code of length 18 is a [18, 9, 6] linear code.
    # It is constructed by extending the [17, 9, 5] QR code with an overall parity bit.
    # The generator polynomial for the [17, 9, 5] QR code is:
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # In binary (coeffs for x^8 ... x^0), this is 100111001.
    # Represented as integer (bit 0 is x^0):
    # g(x) = 1 + 0x + 0x^2 + 1x^3 + 1x^4 + 1x^5 + 0x^6 + 0x^7 + 1x^8
    # Value: 2^0 + 2^3 + 2^4 + 2^5 + 2^8 = 1 + 8 + 16 + 32 + 256 = 313
    
    g_poly = 313
    length_17 = 17
    
    # Construct the 9 rows of the generator matrix for the [17,9,5] code.
    gen_rows_17 = []
    for i in range(9):
        row_val = (g_poly << i) % (1 << length_17)
        gen_rows_17.append(row_val)
        
    # Extend to length 18 by adding an overall parity bit.
    gen_matrix = []
    for row_val in gen_rows_17:
        parity = row_val.bit_count() % 2
        extended_val = row_val | (parity << 17)
        gen_matrix.append(extended_val)
        
    # Generate all 512 codewords
    code_words = [0]
    for g in gen_matrix:
        code_words.extend([w ^ g for w in code_words])
        
    # Verification: Check minimum distance
    # For linear codes, min distance is min weight of non-zero codewords.
    # However, to be absolutely safe and align with "Every pair is checked exactly",
    # we can do a full check. 512^2 is small.
    min_d = 100
    n_words = len(code_words)
    for i in range(n_words):
        if time.time() > deadline:
            break
        for j in range(i + 1, n_words):
            dist = (code_words[i] ^ code_words[j]).bit_count()
            if dist < min_d:
                min_d = dist
                if min_d < MIN_DISTANCE:
                    return [] # Invalid, should not happen
    
    if min_d >= MIN_DISTANCE:
        return code_words
        
    # Fallback: Reed-Muller RM(1,4) which has 32 codewords.
    rm_basis = []
    b0 = (1 << 16) - 1
    rm_basis.append(b0)
    for j in range(4):
        bj = 0
        for i in range(16):
            if (i >> j) & 1:
                bj |= 1 << i
        rm_basis.append(bj)
        
    code_words_rm = [0]
    for b in rm_basis:
        code_words_rm.extend([w ^ b for w in code_words_rm])
        
    return code_words_rm
