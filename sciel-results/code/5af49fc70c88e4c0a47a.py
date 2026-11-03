import time
import numpy as np

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm = n * m   # 8
    mp = m * p   # 28
    pn = p * n   # 14

    deadline = time.time() + 110  # leave safety margin before 120s

    # ------------------------------------------------------------------
    # 1) Try the structural Strassen-based decomposition first (fast, exact)
    #    Expected rank: 2 * (3*7 + 1*2) = 46
    # ------------------------------------------------------------------
    U, V, W = _build_strassen_based(n, m, p)
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ------------------------------------------------------------------
    # 2) Tensor-flattening / SVD approach:
    #    Build the multiplication tensor as a 3D array T[i,j,k] = 1.
    #    The rank of the tensor is the minimum r such that
    #    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    #    We formulate this as: for each (j,k) pair, the matrix
    #    T[:,:,j,k] (size n x p) must be in the span of rank-1 products.
    #
    #    More precisely, we fix V to be the identity on m*p entries
    #    (one column per (j,k)), then we need:
    #    T[i,j,k] = sum_r U[i*m+j, r] * W[k*n+i, r] * delta_{r,(j,k)}
    #    => T[i,j,k] = U[i*m+j, j*p+k] * W[k*n+i, j*p+k]
    #
    #    This is a rank-1 factorization of the "flattened" tensor.
    #    The minimal rank equals the tensor rank, which for matmul(2,4,7)
    #    is at most 56 (schoolbook) and at most 46 (Strassen-based).
    #
    #    To find a lower-rank decomposition, we use the following:
    #    Flatten T into a matrix of size (n*p) x (m) by reshaping:
    #    Actually, the standard approach is:
    #    - Reshape T (n x m x p) into a matrix T_mat of size (n*p) x (m)
    #    where T_mat[i*p+k, j] = T[i,j,k]
    #    - The rank of T_mat gives a lower bound on tensor rank
    #    - But this is just the matrix rank, not the tensor rank
    #
    #    For exact matmul tensor, T[i,j,k] = 1 for all i,j,k, so
    #    T_mat is all-ones, rank 1. That's not helpful directly.
    #
    #    The correct formulation: the matmul tensor is the tensor
    #    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta(i,j,k) where
    #    the multiplication tensor for m,n,p has entries
    #    T[i,j,k] = 1 if the (i,k)-th entry of A*B comes from A[i,j]*B[j,k].
    #    Actually for standard matmul: C[i,k] = sum_j A[i,j] B[j,k]
    #    So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
    #
    #    This means every "slice" is all-ones, and the tensor rank is
    #    the smallest r such that the all-ones tensor of size n×m×p
    #    can be written as sum of r rank-1 tensors.
    #
    #    For the all-ones tensor, rank-1 means U_col ⊗ V_col ⊗ W_col.
    #    So we need: 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    #    for all i, j, k.
    #
    #    This is equivalent to saying the matrix M_{(i,j),(k)} of ones
    #    (size (n*m) x p, but really we need the full 3-way) can be
    #    decomposed. The minimal rank is the tensor rank of the
    #    all-ones tensor, which equals min(n*m, m*p, p*n) in general?
    #    No - the tensor rank of the all-ones tensor of size a×b×c
    #    is at most min(a*b, b*c, a*c) and equals that when one
    #    dimension is 1. For 8×28×14, it's at most 112 but could be lower.
    #
    #    Actually, for the all-ones tensor, we can achieve rank = n*m
    #    by setting for each (i,j): U[i*m+j, i*m+j]=1, V[j*p+k, i*m+j]=1,
    #    W[k*n+i, i*m+j]=1. Then sum_r gives 1 for each (i,j,k).
    #    Similarly rank = m*p or p*n works. So rank ≤ min(nm, mp, pn) = min(8,28,14)=8.
    #
    #    Wait, that can't be right for matmul. Let me re-examine.
    #
    #    For standard matrix multiplication C = AB where A is n×m, B is m×p:
    #    C[i,k] = sum_j A[i,j] B[j,k]
    #
    #    The bilinear algorithm requires:
    #    C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
    #
    #    NO. That's wrong. The correct formulation is:
    #
    #    We compute intermediate values M_r = sum_{i,j} U[i*m+j, r] A[i,j]
    #    and N_r = sum_{j,k} V[j*p+k, r] B[j,k]
    #    Then C[i,k] = sum_r W[k*n+i, r] M_r N_r
    #
    #    For this to equal sum_j A[i,j] B[j,k] for ALL A and B, we need:
    #    sum_r W[k*n+i, r] U[i*m+j1, r] V[j1*p+k1, r] = delta_{j1,j2} (when k=k1)
    #    More precisely: for all i,k,j:
    #    sum_r W[k*n+i, r] U[i*m+j, r] V[j*p+k, r] = 1
    #
    #    So the condition is: for every triple (i,j,k):
    #    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    #    This is EXACTLY the condition I had. And the "all-ones" observation
    #    is correct: T[i,j,k]=1 for all valid indices.
    #
    #    Now, can we achieve rank 8? Let's try:
    #    For each i (0..1) and j (0..3), define r = i*4+j.
    #    U[i*4+j, r] = 1, V[j*7+k, r] = 1 for all k, W[k*2+i, r] = 1 for all k.
    #    Then sum_r U[i*4+j', r] V[j'*7+k', r] W[k'*2+i', r]
    #    = sum over r where U and V and W are all 1 simultaneously.
    #    U[i*4+j', r]=1 only when r=i*4+j'.
    #    V[j'*7+k', r]=1 when r is in {j'*7, j'*7+1, ..., j'*7+6}.
    #    So we need i*4+j' = j'*7+k' for some k'.
    #    This is NOT always satisfiable. So rank 8 via this simple scheme fails.
    #
    #    The actual tensor rank of the matmul tensor is what we need.
    #    For 2x4x7, the known upper bound is 46 (from Strassen) or 56 (schoolbook).
    #
    #    Let me just use the Strassen-based construction which gives rank 46.

    # ------------------------------------------------------------------
    # 3) Fallback: schoolbook rank 56
    # ------------------------------------------------------------------
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_strassen_based(n, m, p):
    """Build a Strassen-based decomposition for 2x4x7 matmul.
    
    Strategy:
    - A (2x4) = [A1 | A2] where A1, A2 are 2x2
    - B (4x7) = [B1; B2] where B1, B2 are 2x7
    - C = A1*B1 + A2*B2
    - For each 2x2 * 2x7 product, split 7 cols into 2+2+2+1
    - 2x2*2x2 uses Strassen (rank 7), 2x2*2x1 uses schoolbook (rank 2)
    - Total: 2 * (3*7 + 1*2) = 46
    """
    # Strassen 2x2x2
    U_s = np.zeros((4, 7), dtype=int)
    V_s = np.zeros((4, 7), dtype=int)
    W_s = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)(B01+B10)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[1, 0] = 1; V_s[2, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1
    # M2 = (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1
    # M3 = A00(B00-B10)
    U_s[0, 2] = 1
    V_s[0, 2] = 1; V_s[2, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1
    # M4 = A11(B11-B01)
    U_s[3, 3] = 1
    V_s[3, 3] = 1; V_s[1, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1
    # M5 = (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Schoolbook 2x2x1 (rank 2)
    U_1 = np.zeros((4, 2), dtype=int)
    V_1 = np.zeros((2, 2), dtype=int)
    W_1 = np.zeros((2, 2), dtype=int)
    for j in range(2):
        for i in range(2):
            U_1[i * 2 + j, j] = 1
        V_1[j, j] = 1
        for i in range(2):
            W_1[i, j] = 1

    total_rank = 46
    U = np.zeros((8, total_rank), dtype=int)
    V = np.zeros((28, total_rank), dtype=int)
    W = np.zeros((14, total_rank), dtype=int)

    col = 0
    for block in range(2):
        # A cols: 2*block, 2*block+1
        # B rows: 2*block, 2*block+1
        # B cols: 0..6, split as 2,2,2,1

        # Three 2x2x2 Strassen blocks
        for sub in range(3):
            for r in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_s[i * 2 + j, r]
                # V
                for j in range(2):
                    for k in range(2):
                        V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_s[j * 2 + k, r]
                # W
                for k in range(2):
                    for i in range(2):
                        W[(sub * 2 + k) * 2 + i, col] += W_s[k * 2 + i, r]
                col += 1

        # One 2x2x1 schoolbook block (col 6)
        for r in range(2):
            for i in range(2):
                for j in range(2):
                    U[i * 4 + (2 * block + j), col] += U_1[i * 2 + j, r]
            for j in range(2):
                V[(2 * block + j) * 7 + 6, col] += V_1[j, r]
            for i in range(2):
                W[6 * 2 + i, col] += W_1[i, r]
            col += 1

    return U, V, W


def _build_schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
