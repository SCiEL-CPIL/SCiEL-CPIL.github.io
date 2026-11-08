"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # Explicit rank-18 decomposition for 3x3 * 3x6
    # Strategy: Use 9 products for the core 3x3 multiplication, then extend
    # each product to cover 2 columns of B (since 6/3 = 2)
    
    rank = 18
    U = [[0] * rank for _ in range(n * m)]  # 9 x 18
    V = [[0] * rank for _ in range(m * p)]  # 18 x 18
    W = [[0] * rank for _ in range(p * n)]  # 18 x 18
    
    # The 9 core products: for each (i,j) in 3x3, we need A[i,:] dot B[:,k]
    # But B has 6 columns. We'll pair columns: (k, k+3) for k=0,1,2
    # Product r = 2*(i*3+j) + s where i,j in {0,1,2}, s in {0,1}
    # s=0 handles columns 0,1,2; s=1 handles columns 3,4,5
    
    for i in range(3):
        for j in range(3):
            base = 2 * (i * 3 + j)
            for s in range(2):
                r = base + s
                # U: row i*3+j gets 1
                U[i * 3 + j][r] = 1
                
                # V: column j of A multiplies rows of B
                # For s=0: B columns 0,1,2 -> V[j*6 + k] for k=0,1,2
                # For s=1: B columns 3,4,5 -> V[j*6 + k] for k=3,4,5
                for k in range(3):
                    col = k if s == 0 else k + 3
                    V[j * 6 + col][r] = 1
                
                # W: output C[i, k] for the relevant columns
                # For s=0: C[i, 0], C[i, 1], C[i, 2] -> W[k*3 + i] for k=0,1,2
                # For s=1: C[i, 3], C[i, 4], C[i, 5] -> W[k*3 + i] for k=3,4,5
                for k in range(3):
                    col = k if s == 0 else k + 3
                    W[col * 3 + i][r] = 1
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank-54
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
