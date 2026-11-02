"""Combined program: block Strassen decomposition + fast einsum verification."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]
N, M, P = 3, 4, 4

# EVOLVE-BLOCK-START

def _verify_einsum(U, V, W, n=N, m=M, p=P):
    """Fast verification using einsum: reconstruct tensor and compare to ones."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=1e-9)


def _verify_strict(U, V, W, n=N, m=M, p=P, tol=1e-12):
    """Strict verification with tighter tolerance."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    recon = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(recon, np.ones((n, m, p)), atol=tol)


def schoolbook(na, mb, pb):
    """Schoolbook decomposition for na x mb * mb x pb, rank na*mb*pb."""
    rank = na * mb * pb
    U = [[0] * rank for _ in range(na * mb)]
    V = [[0] * rank for _ in range(mb * pb)]
    W = [[0] * rank for _ in range(pb * na)]
    col = 0
    for i in range(na):
        for j in range(mb):
            for k in range(pb):
                U[i*mb+j][col] = 1
                V[j*pb+k][col] = 1
                W[k*na+i][col] = 1
                col += 1
    return U, V, W


def strassen_2x2():
    """
    Strassen decomposition for 2x2 * 2x2. Rank 7.
    U: (4,7) indexed U[i*2+j], V: (4,7) indexed V[j*2+k], W: (4,7) indexed W[k*2+i]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]

    # M1: (A00+A11)*(B00+B11) -> C00(+1), C11(+1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    # M2: (A10+A11)*B00 -> C10(+1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1

    # M3: A00*(B01-B11) -> C01(+1), C11(+1)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    # M4: A11*(B10-B00) -> C00(+1), C10(+1)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    # M5: (A00+A10)*B11 -> C00(-1), C01(+1)
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    # M6: (A01-A10)*(B00+B10) -> C11(+1)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1

    # M7: (A01-A11)*(B01+B11) -> C00(+1)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W


def _add_sub(U, V, W, U_sub, V_sub, W_sub, i_off, j_off, k_off, na, mb, pb, col_start, n, m, p):
    """Add a sub-decomposition into the full matrices at the given offsets."""
    r = len(U_sub[0])
    for c in range(r):
        col = col_start + c
        for i_prime in range(na):
            for j_prime in range(mb):
                u_val = U_sub[i_prime*mb + j_prime][c]
                if u_val != 0:
                    U[(i_off + i_prime)*m + (j_off + j_prime)][col] += u_val
        for j_prime in range(mb):
            for k_prime in range(pb):
                v_val = V_sub[j_prime*pb + k_prime][c]
                if v_val != 0:
                    V[(j_off + j_prime)*p + (k_off + k_prime)][col] += v_val
        for k_prime in range(pb):
            for i_prime in range(na):
                w_val = W_sub[k_prime*na + i_prime][c]
                if w_val != 0:
                    W[(k_off + k_prime)*n + (i_off + i_prime)][col] += w_val


def _build_2plus1_split():
    """
    Split A (3x4) into rows 0-1 (2x4) and row 2 (1x4).
    Split B (4x4) into four 2x2 blocks.
    C (3x4) similarly split.
    
    2x2*2x2 blocks use Strassen (rank 7) -> 4 blocks * 7 = 28
    1x2*2x2 strips use schoolbook (rank 4) -> 4 strips * 4 = 16
    Total: 44
    """
    n, m, p = N, M, P
    rank_total = 4 * 7 + 4 * 4  # 28 + 16 = 44
    U = [[0] * rank_total for _ in range(n * m)]  # 12 x 44
    V = [[0] * rank_total for _ in range(m * p)]  # 16 x 44
    W = [[0] * rank_total for _ in range(p * n)]  # 12 x 44

    col = 0

    # Top 2 rows of C (i=0,1), left 2 cols (k=0,1):
    #   C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # Block 1: i_off=0, j_off=0, k_off=0, 2x2*2x2 Strassen
    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 0, 0, 2, 2, 2, col, n, m, p)
    col += 7

    # Block 2: i_off=0, j_off=2, k_off=0, 2x2*2x2 Strassen
    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 2, 0, 2, 2, 2, col, n, m, p)
    col += 7

    # Top 2 rows of C, right 2 cols (k=2,3):
    #   C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # Block 3: i_off=0, j_off=0, k_off=2, 2x2*2x2 Strassen
    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 0, 2, 2, 2, 2, col, n, m, p)
    col += 7

    # Block 4: i_off=0, j_off=2, k_off=2, 2x2*2x2 Strassen
    Us, Vs, Ws = strassen_2x2()
    _add_sub(U, V, W, Us, Vs, Ws, 0, 2, 2, 2, 2, 2, col, n, m, p)
    col += 7

    # Bottom row of C (i=2), left 2 cols (k=0,1):
    #   C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # Strip 1: i_off=2, j_off=0, k_off=0, 1x2*2x2 schoolbook
    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 0, 0, 1, 2, 2, col, n, m, p)
    col += 4

    # Strip 2: i_off=2, j_off=2, k_off=0, 1x2*2x2 schoolbook
    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 2, 0, 1, 2, 2, col, n, m, p)
    col += 4

    # Bottom row of C, right 2 cols (k=2,3):
    #   C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    # Strip 3: i_off=2, j_off=0, k_off=2, 1x2*2x2 schoolbook
    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 0, 2, 1, 2, 2, col, n, m, p)
    col += 4

    # Strip 4: i_off=2, j_off=2, k_off=2, 1x2*2x2 schoolbook
    Us, Vs, Ws = schoolbook(1, 2, 2)
    _add_sub(U, V, W, Us, Vs, Ws, 2, 2, 2, 1, 2, 2, col, n, m, p)
    col += 4

    return U, V, W


def _build_col_split():
    """
    Alternative: split B (4x4) into left 2 cols and right 2 cols,
    and A (3x4) into left 2 cols and right 2 cols.
    C (3x4) split into left 2 and right 2 cols.
    
    C[:, 0:2] = A[:, 0:2]*B[0:2, 0:2] + A[:, 2:4]*B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2]*B[0:2, 2:4] + A[:, 2:4]*B[2:4, 2:4]
    
    Each term is 3x2 * 2x2 -> rank 3*2*2 = 12 (schoolbook)
    4 terms * 12 = 48 (no improvement)
    
    But we can split the 3 rows into 2+1:
    For the 2-row part: 2x2 * 2x2 -> Strassen rank 7
    For the 1-row part: 1x2 * 2x2 -> schoolbook rank 4
    Per output-column-pair: 7 + 4 = 11
    4 column-pairs * 11 = 44 (same as before)
    """
    n, m, p = N, M, P
    rank_total = 4 * (7 + 4)  # 4 * 11 = 44
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    col = 0

    # For each (j_off, k_off) pair in {(0,0),(2,0),(0,2),(2,2)}:
    #   Top 2 rows: Strassen 2x2*2x2
    #   Bottom 1 row: schoolbook 1x2*2x2
    for k_off in [0, 2]:
        for j_off in [0, 2]:
            # Top 2 rows
            Us, Vs, Ws = strassen_2x2()
            _add_sub(U, V, W, Us, Vs, Ws, 0, j_off, k_off, 2, 2, 2, col, n, m, p)
            col += 7
            # Bottom 1 row
            Us, Vs, Ws = schoolbook(1, 2, 2)
            _add_sub(U, V, W, Us, Vs, Ws, 2, j_off, k_off, 1, 2, 2, col, n, m, p)
            col += 4

    return U, V, W


def _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=30):
    """
    Try to reduce rank by merging columns that are linearly dependent
    or by finding a lower-rank equivalent decomposition.
    This is a heuristic: check if any columns can be zeroed out.
    """
    start = time.time()
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]

    # Try removing columns one at a time and re-checking
    # This is O(r * verification_cost) which could be expensive
    # Only try for small r
    if r > 48:
        return None

    best_rank = r
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find a column that can be removed (all-zero contribution)
    # or columns that sum to zero
    for col_idx in range(r - 1, -1, -1):
        if time.time() - start > max_time:
            break
        # Skip this column
        U_test = np.delete(U, col_idx, axis=1)
        V_test = np.delete(V, col_idx, axis=1)
        W_test = np.delete(W, col_idx, axis=1)
        if _verify_einsum(U_test, V_test, W_test, n, m, p):
            U, V, W = U_test, V_test, W_test
            best_rank = W.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if best_rank < r:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Returns dict with 'U', 'V', 'W'.
    """
    n, m, p = N, M, P
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # Build the 2+1 row split decomposition (rank 44)
    U, V, W = _build_2plus1_split()

    # Verify with fast einsum
    if _verify_einsum(U, V, W, n, m, p):
        # Try to reduce rank further
        reduced = _try_rank_reduction_via_merging(U, V, W, n, m, p, max_time=20)
        if reduced is not None:
            U, V, W = reduced
            if _verify_einsum(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

        # Strict verification as final check
        if _verify_strict(U, V, W, n, m, p, tol=1e-12):
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 48
    Us, Vs, Ws = schoolbook(n, m, p)
    if _verify_einsum(Us, Vs, Ws, n, m, p):
        return {"U": Us, "V": Vs, "W": Ws}

    # Last resort: return what we have
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
