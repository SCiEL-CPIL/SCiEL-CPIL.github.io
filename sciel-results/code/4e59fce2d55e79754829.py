import numpy as np
import time

N = 18
MIN_DIST = 6

def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    Uses NumPy for Gaussian elimination to find the basis of the QR code.
    """
    p = 17
    qr = set((i * i) % p for i in range(p))
    
    shifts = []
    for k in range(p):
        w = 0
        for j in range(p):
            if (j - k) % p in qr:
                w |= (1 << j)
        shifts.append(w)
        
    # Convert shifts to a numpy matrix of bits (17x17)
    # rows are the shifts, cols are the bit positions
    M = np.zeros((p, p), dtype=np.uint8)
    for i, w in enumerate(shifts):
        for j in range(p):
            if w & (1 << j):
                M[i, j] = 1
                
    # Gaussian Elimination over GF(2) to find a basis
    # We want to find 9 linearly independent vectors.
    basis_indices = []
    rank = 0
    for col in range(p):
        # Find a row >= rank with a 1 in this column
        pivot_row = None
        for row in range(rank, p):
            if M[row, col] == 1:
                pivot_row = row
                break
        
        if pivot_row is not None:
            # Swap pivot_row with rank
            if pivot_row != rank:
                M[rank], M[pivot_row] = M[pivot_row].copy(), M[rank].copy()
            
            # Eliminate this column from other rows
            for row in range(p):
                if row != rank and M[row, col] == 1:
                    M[row] ^= M[rank]
            
            basis_indices.append(rank) # The row index in the original M? No, M is modified.
            # We need the actual basis vectors.
            # The rows of M that are non-zero in the reduced form are linearly independent.
            # But we lost the original vectors.
            # Better: Keep track of the original vectors corresponding to the basis rows.
            # Or, just extract the basis from the reduced M?
            # The rows of the reduced M are a basis for the code.
            # So we can just take the first 9 non-zero rows of the final M.
            
            rank += 1
            if rank == 9:
                break
                
    # Extract basis vectors from the reduced matrix M
    # The first 'rank' rows of M are the basis in row-echelon form.
    # We convert these back to integers.
    basis_ints = []
    for i in range(rank):
        w = 0
        for j in range(p):
            if M[i, j] == 1:
                w |= (1 << j)
        basis_ints.append(w)
        
    # If rank < 9, something went wrong, but QR code dim is 9.
    # Generate all 512 codewords
    words = []
    for mask in range(512):
        w = 0
        for i in range(len(basis_ints)):
            if mask & (1 << i):
                w ^= basis_ints[i]
        
        parity = w.bit_count() % 2
        w18 = w | (parity << 17)
        words.append(w18)
        
    return words
