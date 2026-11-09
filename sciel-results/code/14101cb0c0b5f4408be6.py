import time

def construct():
    N = 18
    MIN_DIST = 6
    
    # Quadratic residues mod 17 are {1, 2, 4, 8, 9, 13, 15, 16}
    # These correspond to powers of 2 mod 17:
    # 2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=15, 2^6=13, 2^7=9
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7
    # Wait, the previous code used indices [0, 1, 3, 7, 8, 12, 14, 15].
    # Let's re-verify the mapping.
    # If index i corresponds to bit position i in the 17-bit word.
    # The cyclic shift property requires the generator to be defined on Z_17.
    # The standard QR code generator matrix rows are shifts of the vector with 1s at quadratic residue positions.
    
    # Let's stick to the logic that produced 256 valid words in the previous run.
    # The previous run verified min_d >= 6.
    
    def rotate_left(x, n, width):
        return ((x << n) | (x >> (width - n))) & ((1 << width) - 1)

    # Define the base vector based on the successful previous implementation
    # Positions: 0, 1, 3, 7, 8, 12, 14, 15
    row0 = 0
    for pos in [0, 1, 3, 7, 8, 12, 14, 15]:
        row0 |= (1 << pos)
        
    basis = []
    for i in range(8):
        basis.append(rotate_left(row0, i, 17))
        
    words_17 = []
    for mask in range(256):
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                word ^= basis[i]
        words_17.append(word)

    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        extended = w | (parity << 17)
        words_18.append(extended)
        
    return words_18
