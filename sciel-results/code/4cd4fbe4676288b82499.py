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


def _randomized_rank_search(n, m, p, target_rank, max_time=60, seed=42):
    """
    Randomized algebraic approach: generate random rank-1 tensors, build a basis
    via Gaussian elimination, then solve for exact rational coefficients using
    Cramer's rule to find a rank-`target_rank` decomposition.
    """
    start = time.time()
    deadline = start + max_time
    rng = np.random.default_rng(seed)

    total_dim = n * m * p  # 48
    U_dim = n * m  # 12
    V_dim = m * p  # 16
    W_dim = p * n  # 12

    def _time_left():
        return time.time() < deadline

    def _try_solve(basis_indices, n, m, p, target_rank):
        """
        Given basis_indices (list of (u_vec, v_vec, w_vec) tuples),
        solve for coefficients c_i such that sum_i c_i * u_i * v_i * w_i = ones_tensor.
        Uses Cramer's rule on a linear system.
        """
        r = len(basis_indices)
        if r != target_rank:
            return None

        # Build the coefficient matrix A (target_rank x target_rank)
        # and the target vector b
        # For each basis tensor t_i, we need: sum_i c_i * (t_i flattened) = ones_flattened
        # But we have 48 equations and r unknowns. We pick r independent equations.

        # Flatten each basis tensor
        basis_flat = []
        for u_vec, v_vec, w_vec in basis_indices:
            u3 = u_vec.reshape(n, m)
            v3 = v_vec.reshape(m, p)
            w3 = w_vec.reshape(p, n)
            t = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
            basis_flat.append(t.flatten())

        basis_flat = np.array(basis_flat)  # (r, 48)
        target = np.ones(48)

        # Pick r linearly independent rows from the 48x r system
        # Actually, we have: sum_i c_i * basis_flat[i] = target
        # This is: B @ c = target where B is (48, r)
        B = basis_flat.T  # (48, r)

        # Find r linearly independent rows
        # Use QR with column pivoting on B.T to find independent rows
        # Or simpler: try all combinations of r rows from 48 (too many for r > 5)
        # Better: use rank-revealing decomposition

        # For small r, we can try a greedy approach
        if r > 8:
            # Use least squares and check residual
            c, residuals, rank, sv = np.linalg.lstsq(B, target, rcond=None)
            if rank < r:
                return None
            recon = B @ c
            if not np.allclose(recon, target, atol=1e-9):
                return None
            # Try to find exact rational solution
            # Round c to nearest rational with small denominator
            from fractions import Fraction
            c_exact = []
            for val in c:
                if abs(val) < 1e-10:
                    c_exact.append(0)
                    continue
                # Try denominators 1, 2, 3, 4, 5, 6, 8, 10, 12
                found = False
                for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]:
                    num = round(val * denom)
                    if abs(val - num / denom) < 1e-9:
                        c_exact.append(Fraction(num, denom))
                        found = True
                        break
                if not found:
                    return None
            return c_exact

        else:
            # For small r, use Cramer's rule with independent row selection
            # Greedily select r independent rows
            selected_rows = []
            current_rank = 0
            for row_idx in range(48):
                if len(selected_rows) == r:
                    break
                test_matrix = B[selected_rows + [row_idx], :]
                test_rank = np.linalg.matrix_rank(test_matrix, tol=1e-10)
                if test_rank == len(selected_rows) + 1:
                    selected_rows.append(row_idx)

            if len(selected_rows) < r:
                return None

            A_sub = B[selected_rows, :]  # (r, r)
            b_sub = target[selected_rows]

            det_A = np.linalg.det(A_sub)
            if abs(det_A) < 1e-10:
                return None

            # Cramer's rule
            c = np.zeros(r)
            for i in range(r):
                A_mod = A_sub.copy()
                A_mod[:, i] = b_sub
                c[i] = np.linalg.det(A_mod) / det_A

            # Check solution
            recon = B @ c
            if not np.allclose(recon, target, atol=1e-9):
                return None

            # Find exact rational solution
            from fractions import Fraction
            c_exact = []
            for val in c:
                if abs(val) < 1e-10:
                    c_exact.append(0)
                    continue
                found = False
                for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]:
                    num = round(val * denom)
                    if abs(val - num / denom) < 1e-9:
                        c_exact.append(Fraction(num, denom))
                        found = True
                        break
                if not found:
                    return None
            return c_exact

    # Main search loop
    best_result = None
    attempts = 0
    max_attempts = 200

    while _time_left() and attempts < max_attempts:
        attempts += 1

        # Generate random rank-1 tensors
        basis = []
        for _ in range(target_rank):
            u_vec = rng.standard_normal(U_dim)
            v_vec = rng.standard_normal(V_dim)
            w_vec = rng.standard_normal(W_dim)
            basis.append((u_vec, v_vec, w_vec))

        # Check if the basis tensors are linearly independent
        basis_flat = []
        for u_vec, v_vec, w_vec in basis:
            u3 = u_vec.reshape(n, m)
            v3 = v_vec.reshape(m, p)
            w3 = w_vec.reshape(p, n)
            t = np.einsum('ij,jk,ki->ijk', u3, v3, w3)
            basis_flat.append(t.flatten())
        basis_flat = np.array(basis_flat)  # (r, 48)

        rank = np.linalg.matrix_rank(basis_flat, tol=1e-10)
        if rank < target_rank:
            continue

        # Try to solve
        c_exact = _try_solve(basis, n, m, p, target_rank)
        if c_exact is not None:
            # Verify the solution
            U_mat = np.zeros((U_dim, target_rank))
            V_mat = np.zeros((V_dim, target_rank))
            W_mat = np.zeros((W_dim, target_rank))
            for i, (u_vec, v_vec, w_vec) in enumerate(basis):
                U_mat[:, i] = u_vec * c_exact[i]
                V_mat[:, i] = v_vec
                W_mat[:, i] = w_vec

            if _verify_einsum(U_mat, V_mat, W_mat, n, m, p):
                best_result = (U_mat.tolist(), V_mat.tolist(), W_mat.tolist())
                return best_result

        # Also try: generate basis, then use least squares to find coefficients
        # and check if they're rational
        if target_rank <= 10:
            # Generate a larger set and try subsets
            pass  # Skip for now, focus on direct approach

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
    if not _verify_einsum(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48
        Us, Vs, Ws = schoolbook(n, m, p)
        return {"U": Us, "V": Vs, "W": Ws}

    best_rank = 44
    best_U, best_V, best_W = U, V, W

    # Try randomized algebraic search for lower rank
    # Start from rank 30 and go up
    for target_rank in range(30, 45):
        if time.time() > deadline - 15:
            break

        result = _randomized_rank_search(n, m, p, target_rank, max_time=20, seed=target_rank)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify_einsum(U_r, V_r, W_r, n, m, p):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break  # Found a valid decomposition at this rank

    # Final verification
    if _verify_strict(best_U, best_V, best_W, n, m, p, tol=1e-12):
        return {"U": best_U, "V": best_V, "W": best_W}

    # Fallback: schoolbook rank 48
    Us, Vs, Ws = schoolbook(n, m, p)
    return {"U": Us, "V": Vs, "W": Ws}

# EVOLVE-BLOCK-END
