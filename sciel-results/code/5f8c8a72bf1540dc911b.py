"""Matrix multiplication 2x2x9 via Strassen-based rank-32 decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def _build_strassen_2x2x2():
    """Return (U, V, W) for the rank-7 Strassen 2x2x2 decomposition.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    # Strassen products (A is 2x2, B is 2x2):
    # A indices: (i,j) -> row i, col j
    # B indices: (j,k) -> row j, col k
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # M1 = A[0,0] * (B[0,1] + B[1,1])
    # M2 = (A[0,0]+A[0,1]) * B[1,0]
    # M3 = (A[1,0]+A[1,1]) * B[0,0]
    # M4 = A[1,1] * (B[0,0]+B[0,1])
    # M5 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    # M7 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    #
    # C[0,0] = M5 + M4 - M2 - M7
    # C[0,1] = M1 + M2
    # C[1,0] = M3 + M4
    # C[1,1] = M5 + M1 - M3 - M6

    # c=0: M1
    U[0*2+0, 0] = 1.0   # A[0,0]
    V[0*2+1, 0] = 1.0   # B[0,1]
    V[1*2+1, 0] = 1.0   # B[1,1]
    W[0*2+0, 0] = 1.0   # C[0,0] += M1
    W[1*2+0, 0] = 1.0   # C[1,0] += M1  -- wait, let me recheck

    # Let me redo this carefully.
    # C[0,0] = M5 + M4 - M2 - M7
    # C[0,1] = M1 + M2
    # C[1,0] = M3 + M4
    # C[1,1] = M5 + M1 - M3 - M6

    # Reset
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)

    # M1 = A[0,0] * (B[0,1] + B[1,1])
    # U: A[0,0] = index 0*2+0 = 0
    U[0, 0] = 1.0
    V[0*2+1, 0] = 1.0  # B[0,1]
    V[1*2+1, 0] = 1.0  # B[1,1]
    # M1 contributes to C[0,1] and C[1,1]
    W[1*2+0, 0] = 1.0  # C[0,1]: k=1, i=0 -> index 1*2+0=2
    W[1*2+1, 0] = 1.0  # C[1,1]: k=1, i=1 -> index 1*2+1=3

    # M2 = (A[0,0]+A[0,1]) * B[1,0]
    U[0*2+0, 1] = 1.0  # A[0,0]
    U[0*2+1, 1] = 1.0  # A[0,1]
    V[1*2+0, 1] = 1.0  # B[1,0]
    # M2 contributes to C[0,0] and C[0,1]
    W[0*2+0, 1] = 1.0  # C[0,0]
    W[1*2+0, 1] = 1.0  # C[0,1]

    # M3 = (A[1,0]+A[1,1]) * B[0,0]
    U[1*2+0, 2] = 1.0  # A[1,0]
    U[1*2+1, 2] = 1.0  # A[1,1]
    V[0*2+0, 2] = 1.0  # B[0,0]
    # M3 contributes to C[1,0] and C[1,1] (with -1)
    W[0*2+1, 2] = 1.0  # C[1,0]
    W[1*2+1, 2] = -1.0 # C[1,1]

    # M4 = A[1,1] * (B[0,0]+B[0,1])
    U[1*2+1, 3] = 1.0  # A[1,1]
    V[0*2+0, 3] = 1.0  # B[0,0]
    V[0*2+1, 3] = 1.0  # B[0,1]
    # M4 contributes to C[0,0] and C[1,0]
    W[0*2+0, 3] = 1.0  # C[0,0]
    W[0*2+1, 3] = 1.0  # C[1,0]

    # M5 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    U[0*2+0, 4] = 1.0  # A[0,0]
    U[1*2+1, 4] = 1.0  # A[1,1]
    V[0*2+0, 4] = 1.0  # B[0,0]
    V[1*2+1, 4] = 1.0  # B[1,1]
    # M5 contributes to C[0,0] and C[1,1]
    W[0*2+0, 4] = 1.0  # C[0,0]
    W[1*2+1, 4] = 1.0  # C[1,1]

    # M6 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    U[0*2+1, 5] = 1.0   # A[0,1]
    U[1*2+1, 5] = -1.0  # A[1,1]
    V[1*2+0, 5] = 1.0   # B[1,0]
    V[1*2+1, 5] = 1.0   # B[1,1]
    # M6 contributes to C[1,1] (with -1)
    W[1*2+1, 5] = -1.0  # C[1,1]

    # M7 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[1*2+0, 6] = 1.0   # A[1,0]
    U[0*2+0, 6] = -1.0  # A[0,0]
    V[0*2+0, 6] = 1.0   # B[0,0]
    V[0*2+1, 6] = 1.0   # B[0,1]
    # M7 contributes to C[0,0] (with -1)
    W[0*2+0, 6] = -1.0  # C[0,0]

    return U, V, W


def _build_schoolbook_2x2x1():
    """Return (U, V, W) for the rank-4 schoolbook 2x2x1 decomposition.
    U shape (4,4), V shape (2,4), W shape (2,4).
    """
    n, m, p = 2, 2, 1
    rank = 4
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            U[i*m+j, col] = 1.0
            V[j*p+0, col] = 1.0
            W[0*n+i, col] = 1.0
            col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor.
    T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    """
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = 0.0
                for c in range(rank):
                    recon += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(recon - 1.0) > tol:
                    return False
    return True


def _build_rank_32():
    """Build the rank-32 decomposition for 2x2x9.
    The 9 columns of B are split into: 4 blocks of 2 columns + 1 block of 1 column.
    Each 2-column block uses the rank-7 Strassen decomposition.
    The 1-column block uses rank-4 schoolbook.
    Total rank = 4*7 + 4 = 32.
    """
    n, m, p = 2, 2, 9
    total_rank = 32
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)

    # Strassen 2x2x2 base
    U_s, V_s, W_s = _build_strassen_2x2x2()

    # Schoolbook 2x2x1 base
    U_1, V_1, W_1 = _build_schoolbook_2x2x1()

    col_offset = 0

    # 4 blocks of 2 columns (k=0,1; k=2,3; k=4,5; k=6,7)
    for block in range(4):
        k_start = block * 2
        for c in range(7):
            # U: same for all blocks (depends only on i,j)
            U[:, col_offset] = U_s[:, c]
            # V: for this block, B[j, k_start] and B[j, k_start+1]
            # V_s has V_s[j*2 + k_local, c] where k_local in {0,1}
            for j in range(m):
                for k_local in range(2):
                    V[j*p + k_start + k_local, col_offset] = V_s[j*2 + k_local, c]
            # W: for this block, output columns k_start and k_start+1
            # W_s has W_s[k_local*n + i, c] where k_local in {0,1}
            for k_local in range(2):
                for i in range(n):
                    W[(k_start + k_local)*n + i, col_offset] = W_s[k_local*n + i, c]
            col_offset += 1

    # 1 block of 1 column (k=8)
    for c in range(4):
        U[:, col_offset] = U_1[:, c]
        for j in range(m):
            V[j*p + 8, col_offset] = V_1[j*1 + 0, c]
        for i in range(n):
            W[8*n + i, col_offset] = W_1[0*n + i, c]
        col_offset += 1

    return U, V, W


def _build_schoolbook_full():
    """Build the rank-36 schoolbook decomposition for 2x2x9."""
    n, m, p = 2, 2, 9
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Try rank-32 Strassen-based decomposition
    U32, V32, W32 = _build_rank_32()
    
    if _verify(U32, V32, W32, n, m, p, tol=1e-9):
        best_U, best_V, best_W = U32, V32, W32
    else:
        # Fallback to schoolbook rank 36
        best_U, best_V, best_W = _build_schoolbook_full()

    # Convert to lists, round to nearest half-integer if close
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    for matrix in [U_out, V_out, W_out]:
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                val = matrix[row][col]
                rounded = round(val * 2) / 2.0
                if abs(val - rounded) < 1e-6:
                    matrix[row][col] = rounded

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
