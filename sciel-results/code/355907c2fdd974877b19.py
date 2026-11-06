"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the extended quadratic-residue code of length 18.
    # This is a [18, 9, 6] linear code, which is optimal for A_2(18, 6).
    
    # Quadratic residues modulo 17
    Q = {1, 2, 4, 8, 16, 13, 9, 15}
    
    # Basis for the [17, 9, 5] quadratic residue code
    basis = []
    
    # Vector 0: All ones
    v0 = (1 << 17) - 1
    basis.append(v0)
    
    # Vector 1: Characteristic vector of Q (length 17, index 0 is 0)
    # Indices 0..16. Q contains 1..16.
    v1 = 0
    for i in Q:
        v1 |= (1 << i)
    
    # Generate cyclic shifts of v1
    # We need 8 more vectors to reach dimension 9.
    # The shifts by 1, 2, ..., 8 should be independent from v0 and each other.
    for shift in range(1, 9):
        # Cyclic shift right by 'shift' positions
        # (v1 >> shift) | ((v1 & ((1 << shift) - 1)) << (17 - shift))
        shifted = (v1 >> shift) | ((v1 & ((1 << shift) - 1)) << (17 - shift))
        basis.append(shifted)
        
    # Now we have 9 basis vectors of length 17.
    # Generate all 2^9 = 512 codewords.
    words = []
    for mask in range(512):
        word = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
        
        # Extend to length 18 by adding a parity bit
        # Parity bit is 1 if the number of 1s is odd
        parity = word.bit_count() % 2
        extended_word = (word << 1) | parity
        
        words.append(extended_word)
        
    return words
# EVOLVE-BLOCK-END
