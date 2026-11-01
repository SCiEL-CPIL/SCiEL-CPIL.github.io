"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm decomposition (optimal rank for 2x2).
    Verified optimal by Landsberg (1994).
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    rank = 7

    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11) -> C00+=M1, C11+=M1
    # M2 = (A10+A11)B00       -> C10+=M2, C11-=M2
    # M3 = A00(B01-B11)       -> C01+=M3, C11+=M3
    # M4 = A11(B10-B00)       -> C00+=M4, C10+=M4
    # M5 = (A00+A01)B11       -> C00-=M5, C01+=M5
    # M6 = (A10-A00)(B00+B01) -> C11+=M6
    # M7 = (A01-A11)(B10+B11) -> C00+=M7

    # U[i*m+j, c]: coefficient of A[i,j] in product c
    # Rows: A00, A01, A10, A11
    U_data = [
        [1, 0, 1, 0, 1, -1, 0],  # A[0,0]
        [0, 0, 0, 0, 1, 0, 1],   # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],   # A[1,0]
        [1, 1, 0, 1, 0, 0, -1]   # A[1,1]
    ]

    # V[j*p+k, c]: coefficient of B[j,k] in product c
    # Rows: B00, B01, B10, B11
    V_data = [
        [1, 1, 0, -1, 0, 1, 0],  # B[0,0]
        [0, 0, 1, 0, 0, 1, 0],   # B[0,1]
        [0, 0, 0, 1, 0, 0, 1],   # B[1,0]
        [1, 0, -1, 0, 1, 0, 1]   # B[1,1]
    ]

    # W[k*n+i, c]: coefficient for C[i,k] from product c
    # W row index = k*n + i (official trace convention)
    # Row 0: k=0,i=0 -> C[0,0]: M1+M4-M5+M7 = [1,0,0,1,-1,0,1]
    # Row 1: k=0,i=1 -> C[1,0]: M2+M4 = [0,1,0,1,0,0,0]
    # Row 2: k=1,i=0 -> C[0,1]: M3+M5 = [0,0,1,0,1,0,0]
    # Row 3: k=1,i=1 -> C[1,1]: M1-M2+M3+M6 = [1,-1,1,0,0,1,0]
    W_data = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]

    U = np.array(U_data, dtype=float)
    V = np.array(V_data, dtype=float)
    W = np.array(W_data, dtype=float)

    # Verify decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1 for all i,j,k
    # This check is very fast, but we guard it just in case.
    if time.time() < deadline:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] for c in range(rank))
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if not valid:
            # Fallback to schoolbook rank 8 (should not happen)
            rank = 8
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
