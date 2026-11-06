import time

def construct():
    """
    Constructs the Extended Quadratic Residue code of length 18, dimension 9, distance 6.
    This code has 512 codewords.
    
    Uses the generator polynomial g(x) = x^8 + x^5 + x^4 + x^3 + 1 for the
    QR(17) code, extended with an overall parity bit.
    
    The 9x18 generator matrix rows are precomputed as integer bitmasks,
    and all 512 codewords are generated via XOR combinations.
    """
    # Generator polynomial: g(x) = x^8 + x^5 + x^4 + x^3 + 1
    G = (1 << 8) | (1 << 5) | (1 << 4) | (1 << 3) | 1
    
    # Build the 9 basis codewords (18-bit integers)
    basis = []
    for i in range(9):
        # 17-bit codeword: G shifted by i positions
        c17 = G << i
        # Overall parity for extension to length 18
        parity = c17.bit_count() % 2
        word = c17 | (parity << 17)
        basis.append(word)
    
    # Generate all 512 codewords as XOR combinations of basis vectors
    words = [0]
    for b in basis:
        words = [w ^ b for w in words]
    
    return words
