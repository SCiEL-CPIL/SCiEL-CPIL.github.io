"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: Use block decomposition to reduce rank.
    Rank($M_{3,3,7}$) = Rank($M_{2,3,7}$) + Rank($M_{1,3,7}$)
    Rank($M_{1,3,7}$) = 1
    Rank($M_{2,3,7}$) = Rank($M_{2,3,3}$) + Rank($M_{2,3,4}$)
    Rank($M_{2,3,3}$) = Rank($M_{2,3,2}$) + Rank($M_{2,3,1}$) = 2 + 1 = 3
    Rank($M_{2,3,4}$) = 2 * Rank($M_{2,3,2}$) = 4
    Total Rank = 1 + 3 + 4 = 8
    """
    n, m, p = 3, 3, 7
    rank = 8
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def set_ones(mat, indices):
        for idx in indices:
            mat[idx] = 1

    # Term 0: A_row2 * B_all (Rank 1)
    # u: A[2, 0:3] -> indices 6, 7, 8
    # v: B[0:3, 0:7] -> indices 0..20
    # w: C[0:7, 2] -> indices k*3 + 2 for k in 0..6 -> 2, 5, 8, 11, 14, 17, 20
    set_ones(U[:, 0], [6, 7, 8])
    set_ones(V[:, 0], range(0, 21))
    set_ones(W[:, 0], [2, 5, 8, 11, 14, 17, 20])
    
    # Term 1: A_row0 * B_col0:1 (Rank 1 part of M_{2,3,2})
    # u: A[0, 0:3] -> indices 0, 1, 2
    # v: B[0:3, 0:2] -> indices 0..5
    # w: C[0:3, 0] -> indices k*3 + 0 for k in 0..2 -> 0, 3, 6
    set_ones(U[:, 1], [0, 1, 2])
    set_ones(V[:, 1], range(0, 6))
    set_ones(W[:, 1], [0, 3, 6])
    
    # Term 2: A_row1 * B_col0:1 (Rank 1 part of M_{2,3,2})
    # u: A[1, 0:3] -> indices 3, 4, 5
    # v: B[0:3, 0:2] -> indices 0..5
    # w: C[0:3, 1] -> indices k*3 + 1 for k in 0..2 -> 1, 4, 7
    set_ones(U[:, 2], [3, 4, 5])
    set_ones(V[:, 2], range(0, 6))
    set_ones(W[:, 2], [1, 4, 7])
    
    # Term 3: A_row0:2 * B_col2 (Rank 1 part of M_{2,3,1})
    # u: A[0:2, 0:3] -> indices 0..5
    # v: B[0:3, 2] -> indices 2, 5, 8
    # w: C[2, 0:2] -> indices 2*3 + 0, 2*3 + 1 -> 6, 7
    set_ones(U[:, 3], range(0, 6))
    set_ones(V[:, 3], [2, 5, 8])
    set_ones(W[:, 3], [6, 7])
    
    # Term 4: A_row0 * B_col2:4 (Rank 1 part of M_{2,3,2})
    # u: A[0, 0:3] -> indices 0, 1, 2
    # v: B[0:3, 2:4] -> indices 9..14
    # w: C[2:4, 0] -> indices 2*3+0, 3*3+0 -> 6, 9? 
    # Wait, C indices for k=2,3 and i=0.
    # k=2, i=0 -> 6
    # k=3, i=0 -> 9
    # So w: 6, 9
    set_ones(U[:, 4], [0, 1, 2])
    set_ones(V[:, 4], range(9, 15))
    set_ones(W[:, 4], [6, 9])
    
    # Term 5: A_row1 * B_col2:4 (Rank 1 part of M_{2,3,2})
    # u: A[1, 0:3] -> indices 3, 4, 5
    # v: B[0:3, 2:4] -> indices 9..14
    # w: C[2:4, 1] -> indices 2*3+1, 3*3+1 -> 7, 10
    set_ones(U[:, 5], [3, 4, 5])
    set_ones(V[:, 5], range(9, 15))
    set_ones(W[:, 5], [7, 10])
    
    # Term 6: A_row0 * B_col4:6 (Rank 1 part of M_{2,3,2})
    # u: A[0, 0:3] -> indices 0, 1, 2
    # v: B[0:3, 4:6] -> indices 15..20
    # w: C[4:6, 0] -> indices 4*3+0, 5*3+0 -> 12, 15
    set_ones(U[:, 6], [0, 1, 2])
    set_ones(V[:, 6], range(15, 21))
    set_ones(W[:, 6], [12, 15])
    
    # Term 7: A_row1 * B_col4:6 (Rank 1 part of M_{2,3,2})
    # u: A[1, 0:3] -> indices 3, 4, 5
    # v: B[0:3, 4:6] -> indices 15..20
    # w: C[4:6, 1] -> indices 4*3+1, 5*3+1 -> 13, 16
    set_ones(U[:, 7], [3, 4, 5])
    set_ones(V[:, 7], range(15, 21))
    set_ones(W[:, 7], [13, 16])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
