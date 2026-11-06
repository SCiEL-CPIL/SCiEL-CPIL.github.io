import numpy as np
import time
from itertools import combinations

N = 18
MIN_DIST = 6

def construct():
    """
    Construct the optimal binary code of length 18 with minimum Hamming distance 6.
    Uses the extended quadratic-residue code of length 18.
    """
    p = 17
    qr = set((i * i) % p for i in range(p))
    
    # Construct the 17x17 QR code matrix
    # Row i is the shift of the QR sequence by i positions
    M = np.zeros((p, p), dtype=np.uint8)
    for i in range(p):
        for j in range(p):
            if (j - i) % p in qr:
                M[i, j] = 1
    
    # Perform Gaussian elimination to find a basis of dimension 9
    rank = 0
    pivot_cols = []
    
    for col in range(p):
        # Find a row with a 1 in this column
        pivot_row = -1
        for row in range(rank, p):
            if M[row, col] == 1:
                pivot_row = row
                break
        
        if pivot_row == -1:
            continue
            
        # Swap rows
        if pivot_row != rank:
            M[rank], M[pivot_row] = M[pivot_row].copy(), M[pivot_row].copy()
            
        # Eliminate this column from all other rows
        for row in range(p):
            if row != rank and M[row, col] == 1:
                M[row] ^= M[rank]
                
        pivot_cols.append(col)
        rank += 1
        
        if rank == 9:
            break
    
    # Extract the 9 basis vectors (first 'rank' rows of reduced M)
    basis = []
    for i in range(rank):
        w = 0
        for j in range(p):
            if M[i, j] == 1:
                w |= (1 << j)
        basis.append(w)
    
    # Generate all 512 codewords of the [17,9,5] QR code
    # Then extend with parity bit to get [18,9,6] code
    words = []
    for mask in range(512):
        w = 0
        for i in range(len(basis)):
            if mask & (1 << i):
                w ^= basis[i]
        
        # Add parity bit at position 17
        parity = w.bit_count() % 2
        w18 = w | (parity << 17)
        words.append(w18)
    
    return words
