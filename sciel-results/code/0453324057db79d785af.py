def construct():
    import time
    
    # The generator polynomial for the [17, 9, 5] cyclic quadratic residue code
    # g(x) = x^8 + x^5 + x^4 + x^3 + 1
    # Coefficients at bits 8, 5, 4, 3, 0
    g = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    # Generate basis vectors for the [17, 9, 5] cyclic code
    # Basis: x^i * g(x) for i = 0 to 8
    basis = []
    for i in range(9):
        basis.append(g << i)
    
    # Precompute the parity contribution for each basis vector
    # parity_i = popcount(basis[i]) mod 2
    parities = [b.bit_count() & 1 for b in basis]
    
    # Generate all 512 codewords
    # For each 9-bit input mask, the 17-bit codeword is XOR of selected basis vectors
    # The parity bit is XOR of parities of selected basis vectors
    words = [0] * 512
    for mask in range(512):
        word = 0
        parity = 0
        for i in range(9):
            if (mask >> i) & 1:
                word ^= basis[i]
                parity ^= parities[i]
        
        # Extend to 18 bits: add parity bit at position 17
        # The extended QR code is formed by appending a parity bit to the QR code
        # to make all codewords even weight (since min dist of QR is 5, odd, 
        # extension makes min dist 6)
        words[mask] = word | (parity << 17)
    
    return words
