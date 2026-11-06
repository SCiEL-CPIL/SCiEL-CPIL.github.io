def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    This is the extended quadratic residue code of length 18, which is a [18, 9, 6] linear code.
    It has 2^9 = 512 codewords.
    """
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Construct the generator matrix for the QR code of length p.
    # The code is cyclic. We generate the 17 cyclic shifts of the QR indicator vector.
    # We need to find a basis of 9 linearly independent vectors from these shifts.
    
    # Vector v[k] is an integer representing the shift k of the QR indicator.
    # Bit j is 1 if (j - k) % p is in qr.
    # We use integer bit masks for efficiency.
    
    shifts = []
    for k in range(p):
        w = 0
        for j in range(p):
            if (j - k) % p in qr:
                w |= (1 << j)
        shifts.append(w)
        
    # Find a basis of size 9 from the shifts using Gaussian elimination
    # We maintain the basis in row echelon form (sorted by pivot bit)
    basis = [] # list of (pivot_bit, int_val)
    for w in shifts:
        v = w
        # Reduce v using current basis
        for pivot, b in basis:
            if (v >> pivot) & 1:
                v ^= b
        
        if v != 0:
            pivot_v = v.bit_length() - 1
            # Insert into basis keeping it sorted by pivot
            inserted = False
            for i in range(len(basis)):
                if basis[i][0] < pivot_v:
                    basis.insert(i, (pivot_v, v))
                    inserted = True
                    break
            if not inserted:
                basis.append((pivot_v, v))
        
        if len(basis) == 9:
            break
            
    # Now generate all 512 codewords
    words = []
    basis_vecs = [b for _, b in basis]
    
    for mask in range(512):
        w = 0
        for i in range(9):
            if mask & (1 << i):
                w ^= basis_vecs[i]
        
        # Extend to length 18
        parity = w.bit_count() % 2
        w18 = w | (parity << 17)
        words.append(w18)
        
    return words
