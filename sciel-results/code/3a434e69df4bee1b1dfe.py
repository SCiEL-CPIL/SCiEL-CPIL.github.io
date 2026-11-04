import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank is 94
    R = 94
    U = np.zeros((25, R), dtype=float)
    V = np.zeros((25, R), dtype=float)
    W = np.zeros((25, R), dtype=float)
    
    col = 0
    
    # Helper to add trivial terms for a block
    def add_trivial(i_range, j_range, k_range, U, V, W, col):
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    U[i*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*5+i, col] = 1
                    col += 1
        return col
    
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Use Strassen 2x2 (Rank 7)
    # Strassen 2x2 decomposition:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Local indices 0,1 map to Global 0,1
    # U_local[i*2+j, r], V_local[j*2+k, r], W_local[k*2+i, r]
    # We need to map to Global U[i*5+j, col]
    
    # M1
    U[0*5+0, col] = 1; U[1*5+1, col] = 1
    V[0*5+0, col] = 1; V[0*5+1, col] = 1
    W[0*5+0, col] = 1; W[1*5+1, col] = 1
    col += 1
    
    # M2
    U[1*5+0, col] = 1; U[1*5+1, col] = 1
    V[0*5+0, col] = 1
    W[1*5+0, col] = 1; W[1*5+1, col] = 1
    col += 1
    
    # M3
    U[0*5+0, col] = 1
    V[0*5+1, col] = 1; V[1*5+1, col] = -1
    W[0*5+1, col] = 1; W[1*5+1, col] = -1
    col += 1
    
    # M4
    U[1*5+1, col] = 1
    V[1*5+0, col] = 1; V[0*5+0, col] = -1
    W[1*5+0, col] = 1; W[1*5+1, col] = -1
    col += 1
    
    # M5
    U[0*5+0, col] = 1; U[0*5+1, col] = 1
    V[1*5+1, col] = 1
    W[0*5+0, col] = 1; W[0*5+1, col] = 1
    col += 1
    
    # M6
    U[1*5+0, col] = 1; U[0*5+0, col] = -1
    V[0*5+0, col] = 1; V[0*5+1, col] = 1
    W[1*5+0, col] = 1; W[1*5+1, col] = 1
    col += 1
    
    # M7
    U[0*5+1, col] = 1; U[1*5+1, col] = -1
    V[1*5+0, col] = 1; V[1*5+1, col] = 1
    W[0*5+0, col] = -1; W[1*5+1, col] = -1
    col += 1
    
    # 2. A[0:2, 2:5] * B[2:5, 0:2] -> C[0:2, 0:2]
    # i in 0,1; j in 2,3,4; k in 0,1
    col = add_trivial([0, 1], [2, 3, 4], [0, 1], U, V, W, col)
    
    # 3. A[0:2, 2:5] * B[2:5, 2:5] -> C[0:2, 2:5]
    # i in 0,1; j in 2,3,4; k in 2,3,4
    col = add_trivial([0, 1], [2, 3, 4], [2, 3, 4], U, V, W, col)
    
    # 4. A[2:5, 0:2] * B[0:2, 0:2] -> C[2:5, 0:2]
    # i in 2,3,4; j in 0,1; k in 0,1
    col = add_trivial([2, 3, 4], [0, 1], [0, 1], U, V, W, col)
    
    # 5. A[2:5, 0:2] * B[0:2, 2:5] -> C[2:5, 2:5]
    # i in 2,3,4; j in 0,1; k in 2,3,4
    col = add_trivial([2, 3, 4], [0, 1], [2, 3, 4], U, V, W, col)
    
    # 6. A[2:5, 2:5] * B[2:5, 2:5] -> C[2:5, 2:5]
    # i in 2,3,4; j in 2,3,4; k in 2,3,4
    col = add_trivial([2, 3, 4], [2, 3, 4], [2, 3, 4], U, V, W, col)
    
    # Verify col == R
    assert col == R, f"Column mismatch: {col} vs {R}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
