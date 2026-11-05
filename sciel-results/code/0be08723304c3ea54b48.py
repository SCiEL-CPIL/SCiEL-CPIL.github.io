"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # The maximum size of a binary code of length 18 and minimum distance 6
    # is A_2(18, 6) = 512. This is achieved by the extended quadratic residue
    # code of length 18, which is a linear [18, 9, 6] code.
    
    # We construct the [17, 9, 5] QR code and extend it by a parity bit.
    # The generator polynomial for the [17, 9, 5] QR code is g(x) = x^8 + x^5 + x^4 + x^3 + 1.
    # This corresponds to the binary vector of length 17 with 1s at indices 0, 3, 4, 5, 8.
    
    # 1. Generate the 9 generator rows for the length-17 cyclic code.
    # The generator matrix consists of 9 cyclic shifts of the generator polynomial vector.
    g_poly_bits = [0, 0, 0, 1, 1, 1, 0, 0, 1] # indices 0,3,4,5,8 are 1. Wait, length 9? No, length 17.
    # Let's define the generator vector of length 17.
    gen_vec = [0] * 17
    gen_vec[0] = 1
    gen_vec[3] = 1
    gen_vec[4] = 1
    gen_vec[5] = 1
    gen_vec[8] = 1
    
    gen_rows = []
    for shift in range(9):
        row = 0
        for i in range(17):
            if gen_vec[(i - shift) % 17]:
                row |= (1 << i)
        gen_rows.append(row)
        
    # 2. Generate all 512 codewords.
    # For each message (0 to 511), compute the linear combination of generator rows.
    words = []
    for msg in range(512):
        word = 0
        # Unrolled XOR for speed
        if msg & 1: word ^= gen_rows[0]
        if msg & 2: word ^= gen_rows[1]
        if msg & 4: word ^= gen_rows[2]
        if msg & 8: word ^= gen_rows[3]
        if msg & 16: word ^= gen_rows[4]
        if msg & 32: word ^= gen_rows[5]
        if msg & 64: word ^= gen_rows[6]
        if msg & 128: word ^= gen_rows[7]
        if msg & 256: word ^= gen_rows[8]
        
        # Extend with parity bit at position 17 (0-indexed).
        # The QR code of length 17 has minimum distance 5.
        # The extended code has minimum distance 6.
        # Parity bit = sum of bits mod 2.
        if word.bit_count() % 2 == 1:
            word |= (1 << 17)
            
        words.append(word)
        
    # 3. Verification (Optional but good for safety, though logically guaranteed).
    # Given the strict time limit and the known optimality, we can skip full O(N^2) verification
    # if we trust the construction, but a quick check or spot check is safer.
    # However, 512^2 / 2 = 131k operations is trivial.
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if (wi ^ words[j]).bit_count() < 6:
                # Fallback: return a known valid smaller code if construction failed
                # This should never happen.
                return [0]
                
    return words
# EVOLVE-BLOCK-END
