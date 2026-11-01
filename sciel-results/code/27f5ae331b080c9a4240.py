"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-27 exact decomposition for 3x3 matrix multiplication.
    
    The schoolbook algorithm gives a rank-27 decomposition.
    Each term corresponds to one output element C[i,k] = sum_j A[i,j] * B[j,k].
    
    The decomposition satisfies:
        sum_{r=0}^{26} U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = 1  for all i,j,k in {0,1,2}
    
    This is the trivial rank-27 decomposition where each term isolates one (i,j,k) triple.
    """
    n, m, p = 3, 3, 3
    rank = 27
    
    # Build U, V, W matrices: shape (9, rank)
    U = [[0] * rank for _ in range(9)]
    V = [[0] * rank for _ in range(9)]
    W = [[0] * rank for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * 3 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 3 + i][col] = 1
                col += 1
    
    # Verify the decomposition
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for r in range(rank):
                    s += U[i * 3 + j][r] * V[j * 3 + k][r] * W[k * 3 + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # This should never happen for the schoolbook algorithm
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
