"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Use block decomposition 6 = 2 + 2 + 2
    # 3x3 blocks of 2x2 matrices
    # Number of block products: 3 * 3 * 3 = 27
    # Rank of M_2 (Strassen) = 7
    # Total rank = 27 * 7 = 189
    
    rank = 27 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block indices i, k, j in 0..2
    for i in range(3):
        for k in range(3):
            for j in range(3):
                # This block product A[i,k] * B[k,j] contributes to C[i,j]
                # A[i,k] is a 2x2 matrix at rows 2i..2i+1, cols 2k..2k+1
                # B[k,j] is a 2x2 matrix at rows 2k..2k+1, cols 2j..2j+1
                # C[i,j] is a 2x2 matrix at rows 2i..2i+1, cols 2j..2j+1
                
                # Strassen 2x2:
                # M1 = (a+b)(e+f)
                # M2 = (c+d)(e+f)
                # M3 = a(e-g)
                # M4 = d(h-f)
                # M5 = (a+d)(h-e)
                # M6 = (b+c)(g+h)
                # M7 = (a-b)(f+g)
                
                # A local indices (r, s) -> global (2i+r, 2k+s)
                # B local indices (u, v) -> global (2k+u, 2j+v)
                # C local indices (w, x) -> global (2i+w, 2j+x)
                
                # Helper to get global index
                def idx_A(r, s): return (2*i + r) * 6 + (2*k + s)
                def idx_B(u, v): return (2*k + u) * 6 + (2*j + v)
                def idx_C(w, x): return (2*i + w) * 6 + (2*j + x)
                
                # M1
                U[idx_A(0,0)][col] += 1
                U[idx_A(0,1)][col] += 1
                V[idx_B(0,0)][col] += 1
                V[idx_B(0,1)][col] += 1
                W[idx_C(0,0)][col] += 1
                W[idx_C(1,1)][col] += 1
                col += 1
                
                # M2
                U[idx_A(1,0)][col] += 1
                U[idx_A(1,1)][col] += 1
                V[idx_B(0,0)][col] += 1
                V[idx_B(0,1)][col] += 1
                W[idx_C(1,0)][col] += 1
                W[idx_C(1,1)][col] -= 1
                col += 1
                
                # M3
                U[idx_A(0,0)][col] += 1
                V[idx_B(0,0)][col] += 1
                V[idx_B(1,0)][col] -= 1
                W[idx_C(0,1)][col] += 1
                W[idx_C(1,0)][col] -= 1
                col += 1
                
                # M4
                U[idx_A(1,1)][col] += 1
                V[idx_B(1,1)][col] += 1
                V[idx_B(0,1)][col] -= 1
                W[idx_C(0,0)][col] += 1
                W[idx_C(1,1)][col] += 1
                col += 1
                
                # M5
                U[idx_A(0,0)][col] += 1
                U[idx_A(1,1)][col] += 1
                V[idx_B(1,1)][col] += 1
                V[idx_B(0,0)][col] -= 1
                W[idx_C(0,0)][col] -= 1
                W[idx_C(0,1)][col] += 1
                W[idx_C(1,0)][col] += 1
                W[idx_C(1,1)][col] -= 1
                col += 1
                
                # M6
                U[idx_A(0,1)][col] += 1
                U[idx_A(1,0)][col] += 1
                V[idx_B(1,0)][col] += 1
                V[idx_B(1,1)][col] += 1
                W[idx_C(0,1)][col] += 1
                W[idx_C(1,1)][col] += 1
                col += 1
                
                # M7
                U[idx_A(0,0)][col] += 1
                U[idx_A(0,1)][col] -= 1
                V[idx_B(0,1)][col] += 1
                V[idx_B(1,0)][col] += 1
                W[idx_C(0,0)][col] += 1
                W[idx_C(1,0)][col] -= 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
