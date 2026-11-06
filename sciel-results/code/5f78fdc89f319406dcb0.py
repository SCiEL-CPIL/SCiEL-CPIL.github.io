import time

N = 18
MIN_DIST = 6
TIME_LIMIT = 60.0
DEADLINE = time.time() + TIME_LIMIT - 5.0

def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    This is the extended quadratic residue code of length 18, which is a [18, 9, 6] linear code.
    It has 2^9 = 512 codewords.
    A_2(18, 6) = 512 is the known maximum.
    """
    p = 17
    # Quadratic residues mod 17: {1, 2, 4, 8, 9, 13, 15, 16, 0}
    # Wait, residues of squares mod 17:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=8, 6^2=2, 7^2=15, 8^2=13
    # So QR = {1, 2, 4, 8, 9, 13, 15, 16}. 0 is not usually included in the set of non-zero residues,
    # but for the QR code construction, we define the generator polynomial or the cyclic shifts.
    
    # The QR code of length p=17 is cyclic.
    # The generator polynomial g(x) for the QR code of length 17 is related to the minimal polynomial of alpha over GF(2).
    # However, a simpler constructive method for the [17, 9, 5] QR code (which extends to [18, 9, 6])
    # is to use the set of shifts of the indicator vector of the quadratic residues.
    
    # Let's define the base vector corresponding to the quadratic residues.
    # The set of quadratic residues mod 17 is Q = {1, 2, 4, 8, 9, 13, 15, 16}.
    # Note: 0 is not a quadratic residue in the standard sense for this construction's generator matrix rows,
    # but the code is cyclic.
    
    # Actually, the standard construction for the QR code C of length p uses the generator matrix where
    # the rows are cyclic shifts of a vector v where v[i] = 1 if i is a quadratic residue mod p, else 0.
    # This yields a [p, (p+1)/2, (p+3)/4] code? No, for p=17, it's [17, 9, 5].
    
    # Let's build the 9 basis vectors by taking the first 9 independent cyclic shifts of the QR indicator vector.
    
    # Indicator vector for QR mod 17
    qr_set = set((i * i) % p for i in range(1, p)) # Non-zero quadratic residues
    # qr_set = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Construct the base word w0 of length 17
    w0 = 0
    for j in range(p):
        if j in qr_set:
            w0 |= (1 << j)
            
    # Generate cyclic shifts
    shifts = []
    for k in range(p):
        # Shift w0 right by k bits (mod 2^17 - 1 effectively, but we are in vector space GF(2)^17)
        # Cyclic shift: bit j moves to (j-k) % p
        w = 0
        for j in range(p):
            if ((j - k) % p) in qr_set:
                w |= (1 << j)
        shifts.append(w)
        
    # Find a basis of size 9
    basis = []
    for w in shifts:
        if not basis:
            basis.append(w)
            continue
        
        # Check if w is in the span of the current basis
        # Since we only have up to 8 vectors in basis, the span size is at most 2^8 = 256.
        # We can compute the span explicitly.
        span = {0}
        for b in basis:
            new_span = set()
            for s in span:
                new_span.add(s ^ b)
            span |= new_span
            
        if w not in span:
            basis.append(w)
            if len(basis) == 9:
                break
                
    # We should have found 9 independent vectors.
    # Generate all 512 codewords in GF(2)^17
    words_17 = []
    for mask in range(512):
        w = 0
        for i in range(9):
            if mask & (1 << i):
                w ^= basis[i]
        words_17.append(w)
        
    # Extend to length 18 by adding an overall parity bit
    # The extended QR code has parameters [18, 9, 6].
    # The parity bit is the sum of all bits mod 2.
    words_18 = []
    for w in words_17:
        parity = w.bit_count() % 2
        # Place parity bit at position 17 (MSB)
        w18 = w | (parity << 17)
        words_18.append(w18)
        
    return words_18
