"""Tensor decomposition for 2x2 by 2x16 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix A by a 2x16 matrix B to produce a 2x16 matrix C.
    
    Strategy: Use Strassen's algorithm on 2x2 blocks of columns.
    Total rank = 8 * 7 = 56.
    """
    
    def strassen_block_decomp():
        n, m, p = DIMENSIONS
        num_blocks = p // 2
        rank = num_blocks * 7
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for block in range(num_blocks):
            k0 = block * 2
            k1 = k0 + 1
            base = block * 7
            
            # M1
            U[0][base + 0] = 1
            U[3][base + 0] = 1
            V[k0][base + 0] = 1
            V[16 + k1][base + 0] = 1
            W[k0][base + 0] = 1
            W[k1][base + 0] = 1
            W[2 + k0][base + 0] = 1
            W[2 + k1][base + 0] = 1
            
            # M2
            U[2][base + 1] = 1
            U[3][base + 1] = 1
            V[k0][base + 1] = 1
            W[2 + k0][base + 1] = 1
            
            # M3
            U[0][base + 2] = 1
            V[16 + k0][base + 2] = 1
            V[16 + k1][base + 2] = -1
            W[k1][base + 2] = 1
            
            # M4
            U[3][base + 3] = 1
            V[k1][base + 3] = 1
            V[k0][base + 3] = -1
            W[k0][base + 3] = 1
            W[2 + k0][base + 3] = 1
            
            # M5
            U[0][base + 4] = 1
            U[1][base + 4] = 1
            V[16 + k1][base + 4] = 1
            W[k0][base + 4] = -1
            W[k1][base + 4] = 1
            
            # M6
            U[2][base + 5] = 1
            U[0][base + 5] = -1
            V[k0][base + 5] = 1
            V[k1][base + 5] = 1
            W[2 + k1][base + 5] = 1
            
            # M7
            U[1][base + 6] = 1
            U[3][base + 6] = -1
            V[16 + k0][base + 6] = 1
            V[16 + k1][base + 6] = 1
            W[k0][base + 6] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(decomp):
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        n, m, p = DIMENSIONS
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    decomp = strassen_block_decomp()
    if verify(decomp):
        return decomp
    
    # Fallback (should not happen)
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
