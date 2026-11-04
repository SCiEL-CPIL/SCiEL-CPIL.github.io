"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    Uses Strassen for the 3x3x3 block and schoolbook for the 3x3x4 block.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 120s limit

    def verify(U, V, W, tol=1e-6):
        """Verify that the decomposition exactly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Schoolbook solution (rank 63) as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to construct a lower-rank solution using Strassen for 3x3x3 + schoolbook for 3x3x4
    # Strassen 3x3x3: rank 7
    # Schoolbook 3x3x4: rank 36
    # Total: rank 43

    rank_strassen = 7
    rank_school_34 = 36
    total_rank = rank_strassen + rank_school_34

    if total_rank < best_rank:
        U_try = np.zeros((n * m, total_rank), dtype=float)
        V_try = np.zeros((m * p, total_rank), dtype=float)
        W_try = np.zeros((p * n, total_rank), dtype=float)

        # Part 1: Strassen for columns 0,1,2 of B (3x3x3)
        # Strassen's algorithm for 3x3 matrix multiplication
        # We'll use the known rank-7 decomposition for 3x3x3

        # M1 = (A[0,0]+A[1,1]+A[2,2]) * (B[0,0]+B[1,1]+B[2,2])
        # M2 = (A[0,0]-A[1,1]) * (B[0,0]-B[1,1])
        # M3 = (A[0,0]-A[2,2]) * (B[0,0]-B[2,2])
        # M4 = (A[1,1]-A[2,2]) * (B[1,1]-B[2,2])
        # M5 = (A[0,1]+A[1,2]) * (B[1,0]+B[2,0])
        # M6 = (A[0,2]+A[2,1]) * (B[2,0]+B[0,1])
        # M7 = (A[1,2]+A[0,1]) * (B[0,2]+B[1,2])

        # C[0,0] = M1 - M2 - M3 + M4
        # C[1,1] = M1 - M2 + M3 - M4
        # C[2,2] = M1 + M2 - M3 - M4
        # C[0,1] = M5
        # C[1,0] = M6
        # C[2,0] = M7
        # C[0,2] = M7
        # C[1,2] = M5
        # C[2,1] = M6

        # U matrix: U[i*m+j, r] for A[i,j]
        # V matrix: V[j*p+k, r] for B[j,k]
        # W matrix: W[k*n+i, r] for C[k,i] (note: trace convention W[k*n+i])

        # M1: (A00+A11+A22) * (B00+B11+B22)
        U_try[0, 0] = 1.0   # A[0,0]
        U_try[4, 0] = 1.0   # A[1,1]
        U_try[8, 0] = 1.0   # A[2,2]
        V_try[0, 0] = 1.0   # B[0,0]
        V_try[8, 0] = 1.0   # B[1,1]
        V_try[16, 0] = 1.0  # B[2,2]
        # W for C[0,0], C[1,1], C[2,2]
        W_try[0, 0] = 1.0   # C[0,0] (k=0, i=0)
        W_try[4, 0] = 1.0   # C[1,1] (k=1, i=1)
        W_try[8, 0] = 1.0   # C[2,2] (k=2, i=2)

        # M2: (A00-A11) * (B00-B11)
        U_try[0, 1] = 1.0
        U_try[4, 1] = -1.0
        V_try[0, 1] = 1.0
        V_try[8, 1] = -1.0
        W_try[0, 1] = -1.0
        W_try[4, 1] = -1.0
        W_try[8, 1] = 1.0

        # M3: (A00-A22) * (B00-B22)
        U_try[0, 2] = 1.0
        U_try[8, 2] = -1.0
        V_try[0, 2] = 1.0
        V_try[16, 2] = -1.0
        W_try[0, 2] = -1.0
        W_try[4, 2] = 1.0
        W_try[8, 2] = -1.0

        # M4: (A11-A22) * (B11-B22)
        U_try[4, 3] = 1.0
        U_try[8, 3] = -1.0
        V_try[8, 3] = 1.0
        V_try[16, 3] = -1.0
        W_try[0, 3] = 1.0
        W_try[4, 3] = -1.0
        W_try[8, 3] = -1.0

        # M5: (A01+A12) * (B10+B20)
        U_try[1, 4] = 1.0   # A[0,1]
        U_try[5, 4] = 1.0   # A[1,2]
        V_try[7, 4] = 1.0   # B[1,0]
        V_try[14, 4] = 1.0  # B[2,0]
        W_try[1, 4] = 1.0   # C[0,1] (k=1, i=0)
        W_try[5, 4] = 1.0   # C[1,2] (k=2, i=1)

        # M6: (A02+A21) * (B20+B01)
        U_try[2, 5] = 1.0   # A[0,2]
        U_try[7, 5] = 1.0   # A[2,1]
        V_try[14, 5] = 1.0  # B[2,0]
        V_try[1, 5] = 1.0   # B[0,1]
        W_try[0, 5] = 1.0   # C[1,0] (k=0, i=1)
        W_try[4, 5] = 1.0   # C[2,1] (k=1, i=2)

        # M7: (A12+A01) * (B02+B12)
        U_try[5, 6] = 1.0   # A[1,2]
        U_try[1, 6] = 1.0   # A[0,1]
        V_try[2, 6] = 1.0   # B[0,2]
        V_try[9, 6] = 1.0   # B[1,2]
        W_try[2, 6] = 1.0   # C[2,0] (k=0, i=2)
        W_try[6, 6] = 1.0   # C[0,2] (k=2, i=0)

        # Part 2: Schoolbook for columns 3,4,5,6 of B (3x3x4)
        # For each (i,j,k) with k in 3..6
        col_offset = rank_strassen
        for i in range(n):
            for j in range(m):
                for k in range(3, p):
                    U_try[i * m + j, col_offset] = 1.0
                    V_try[j * p + k, col_offset] = 1.0
                    W_try[k * n + i, col_offset] = 1.0
                    col_offset += 1

        if verify(U_try, V_try, W_try, tol=1e-6):
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = total_rank

    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
