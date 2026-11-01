"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # U[i*m+j, c]: coefficient of A[i,j] in product c
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],   # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],    # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],    # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]    # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],   # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],    # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],    # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]    # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,j] from product c
    # W row index = k*n + i (official trace convention)
    # Row 0: k=0,i=0 -> C[0,0]: M1+M4-M5+M7
    # Row 1: k=0,i=1 -> C[1,0]: M2+M4
    # Row 2: k=1,i=0 -> C[0,1]: M3+M5
    # Row 3: k=1,i=1 -> C[1,1]: M1-M2+M3+M6
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = np.array(U_data, dtype=int)
    V = np.array(V_data, dtype=int)
    W = np.array(W_data, dtype=int)

    # Verify correctness by checking the tensor identity
    # The tensor for matrix mult: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # For 2x2x2, T[i,j,k] should be 1 if i==j or j==k or i==k... 
    # Actually, the matrix multiplication tensor T has T[i,j,k]=1 iff we're 
    # computing C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = delta_{i,j} * delta_{j,k}... 
    # No, the tensor is T[i,j,k] where the output C[i,k] = sum_j A[i,j] B[j,k].
    # So the trilinear form: sum_{i,j,k} T[i,j,k] x_i y_j z_k = sum_{i,k} (sum_j A[i,j]B[j,k]) C_out[i,k]
    # The tensor entries are: T[i,j,k] = 1 if this term appears, which is always A[i,j]*B[j,k]
    # contributing to C[i,k]. So T[i,j,k] = 1 for all i,j,k (it's the full 2x2x2 tensor).
    # 
    # Verification: sum_c U[i,j,c] * V[j,k,c] * W[k*n+i, c] should equal 1 for all i,j,k.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(7):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                assert val == 1, f"Verification failed at i={i},j={j},k={k}, got {val}"

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
