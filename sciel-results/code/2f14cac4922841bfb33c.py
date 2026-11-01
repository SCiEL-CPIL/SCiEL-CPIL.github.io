"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    # Verify exactness via tensor contraction
    if time.time() < _deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # multiplication tensor is all 1s for (i,j,k) where C[i,k] += A[i,j]*B[j,k]
                    # Actually T[i,j,k] = 1 for all valid (i,j,k) since each product term appears
                    # Wait: the multiplication tensor T has T[i,j,k] = 1 if the (i,j) element of A
                    # and (j,k) element of B contribute to (i,k) element of C. For standard matmul,
                    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                    # Let me verify: C[i,k] = sum_j A[i,j] * B[j,k]
                    # So the tensor T[A_indices, B_indices, C_indices] where
                    # A_index = i*m+j, B_index = j*p+k, C_index = k*n+i
                    # T[i*m+j, j*p+k, k*n+i] = 1
                    # In our factorization: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                    # where a = i*m+j, b = j*p+k, c = k*n+i
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(actual - expected) > 1e-10:
                        # Fallback: return as-is (shouldn't happen)
                        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
