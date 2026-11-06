<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987) with NumPy acceleration.'''
import math
import time
import numpy as np


def _jc_distances(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy for speed."""
    n = len(labels)
    seqs = np.array([list(sequences[lab]) for lab in labels], dtype='U1')
    # Compute pairwise Hamming distances
    # For each pair, count mismatches
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(seqs[i] != seqs[j])
            p = mismatches / len(sequences[labels[0]])
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''BIONJ-corrected Neighbor-Joining with vectorized binary-matrix distance computation.'''
import math
import time
import numpy as np


def _jc_distances(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized binary matrix."""
    n = len(labels)
    L = len(sequences[labels[0]])
    # Build binary matrix: each nucleotide encoded as 2 bits, or use 4 one-hot arrays
    # For speed, encode A=0, C=1, G=2, T=3 and use integer comparison
    # Vectorized approach: stack sequences into int array, compute pairwise mismatches
    # Use numpy char arrays for efficient comparison
    seqs = np.array([list(sequences[lab]) for lab in labels], dtype='S1')
    # Pairwise Hamming: for each pair (i,j), count where seqs[i] != seqs[j]
    # Vectorize: broadcast to (n, n, L) would be too large for large L
    # Instead, use the fact that mismatch count = L - match count
    # Compute matches via broadcasting in chunks or use a loop-free approach
    # For n<=32 and L<=2000, n*n*L = 32*32*2000 = 2M, which is fine
    # But let's do it more efficiently with matrix operations
    # Encode each sequence as 4 binary vectors (one per nucleotide)
    A = (seqs == b'A').astype(np.int8)
    C = (seqs == b'C').astype(np.int8)
    G = (seqs == b'G').astype(np.int8)
    T = (seqs == b'T').astype(np.int8)
    # Match count for pair (i,j) = sum over positions where both have same base
    # = A[i].A[j] + C[i].C[j] + G[i].G[j] + T[i].T[j]
    matches = A @ A.T + C @ C.T + G @ G.T + T @ T.T
    mismatches = L - matches
    p = mismatches.astype(np.float64) / L
    # JC correction
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    # Clip values that are invalid (p >= 0.75)
    d[p >= 0.7499] = 3.0
    np.fill_diagonal(d, 0.0)
    return d
>>>>>>> REPLACE

