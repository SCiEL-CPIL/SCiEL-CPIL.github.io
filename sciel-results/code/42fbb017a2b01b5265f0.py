"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the [18, 9, 6] extended quadratic residue code.
    # This code has 2^9 = 512 codewords, which is the maximum A_2(18, 6).
    
    # Quadratic residues modulo 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Create the base 17-bit vector
    base_vec = 0
    for i in residues:
        base_vec |= (1 << i)
        
    # Generate 9 basis vectors by cyclic shifting and extending with parity
    basis = []
    for k in range(9):
        # Cyclic shift right by k (equivalent to left by 17-k)
        # In binary representation, shifting right by k moves bit i to i-k mod 17
        # We want the row k to be the shift of the generator polynomial.
        # Let's define shift such that bit i moves to (i+k) % 17
        shifted_vec = 0
        for i in range(17):
            if (base_vec >> i) & 1:
                shifted_vec |= (1 << ((i + k) % 17))
                
        # Calculate parity (XOR of all 17 bits)
        parity = 0
        temp = shifted_vec
        while temp:
            parity ^= 1
            temp &= temp - 1
        # Alternatively: parity = shifted_vec.bit_count() % 2
        
        # Construct 18-bit integer: 17 bits + parity bit at position 17
        word = shifted_vec | (parity << 17)
        basis.append(word)
        
    # Generate all 512 codewords by XORing all subsets of the basis
    codewords = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        codewords.append(word)
        
    return codewords
# EVOLVE-BLOCK-END
