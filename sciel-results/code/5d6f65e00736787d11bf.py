"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-39 exact tensor decomposition for 2x2 by 2x11 matrix multiplication
    using block Strassen algorithm.
    """
    n, m, p = DIMENSIONS
    rank = 39
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    # Process 5 blocks of 2x2
    for block_idx in range(5):
        # Column indices in B and C
        k1 = 2 * block_idx
        k2 = 2 * block_idx + 1
        
        # Strassen terms M1..M7
        # M1 = (a00 + a11)(b0k1 + b1k2)
        U[0, col] += 1; U[3, col] += 1
        V[11*k1, col] += 1; V[11*k2 + 1, col] += 1
        # W contributions:
        # c0k1 += M1, c1k2 += M1
        W[k1*n + 0, col] += 1
        W[k2*n + 1, col] += 1
        col += 1
        
        # M2 = (a10 + a11)b0k1
        U[2, col] += 1; U[3, col] += 1
        V[11*k1, col] += 1
        # W: c1k1 += M2, c1k2 -= M2
        W[k1*n + 1, col] += 1
        W[k2*n + 1, col] -= 1
        col += 1
        
        # M3 = a00(b0k2 - b1k2)
        U[0, col] += 1
        V[11*k2, col] += 1; V[11*k2 + 1, col] -= 1
        # W: c0k2 += M3, c1k2 += M3
        W[k2*n + 0, col] += 1
        W[k2*n + 1, col] += 1
        col += 1
        
        # M4 = a11(b1k1 - b0k1)
        U[3, col] += 1
        V[11*k1 + 1, col] += 1; V[11*k1, col] -= 1
        # W: c0k1 += M4, c1k1 += M4
        W[k1*n + 0, col] += 1
        W[k1*n + 1, col] += 1
        col += 1
        
        # M5 = (a00 + a01)b1k2
        U[0, col] += 1; U[1, col] += 1
        V[11*k2 + 1, col] += 1
        # W: c0k1 -= M5, c0k2 += M5
        W[k1*n + 0, col] -= 1
        W[k2*n + 0, col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b0k1 + b0k2)
        U[2, col] += 1; U[0, col] -= 1
        V[11*k1, col] += 1; V[11*k2, col] += 1
        # W: c1k2 += M6
        W[k2*n + 1, col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b1k1 + b1k2)
        U[1, col] += 1; U[3, col] -= 1
        V[11*k1 + 1, col] += 1; V[11*k2 + 1, col] += 1
        # W: c0k1 += M7
        W[k1*n + 0, col] += 1
        col += 1

    # Process last block of 2x1 (col 10)
    k1 = 10
    
    # N1 = a00 * b0k1
    U[0, col] += 1
    V[11*k1, col] += 1
    W[k1*n + 0, col] += 1
    col += 1
    
    # N2 = a01 * b1k1
    U[1, col] += 1
    V[11*k1 + 1, col] += 1
    W[k1*n + 0, col] += 1
    col += 1
    
    # N3 = a10 * b0k1
    U[2, col] += 1
    V[11*k1, col] += 1
    W[k1*n + 1, col] += 1
    col += 1
    
    # N4 = a11 * b1k1
    U[3, col] += 1
    V[11*k1 + 1, col] += 1
    W[k1*n + 1, col] += 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
