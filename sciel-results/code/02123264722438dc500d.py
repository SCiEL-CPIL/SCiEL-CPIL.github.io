"""Combined program: structured Strassen tiling (rank 77) + Null Space based rank reduction."""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 115.0  # safety margin before 120s evaluator limit


def _build_structured_rank77():
    """Build the proven rank-77 nested Strassen tiling decomposition."""
    n, m, p = 4, 4, 6
    # Strassen coefficients for 2x2x2
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]

    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))

    col = 0
    for s in range(7):
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]

            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma

            col += 1

        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]

                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=6, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    # Check zeros
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k1 = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j1 != j2 or i1 != i2:
                    s = 0.0
                    for r in range(R):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > tol:
                        return False
    return True


def _try_nullspace_reduction(U, V, W, tol=1e-9):
    """Try to reduce rank by finding linear dependencies in the tensor space."""
    R = U.shape[1]
    n, m, p = 4, 4, 6
    
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    # Build the full constraint matrix including zero constraints
    # Total constraints: n*m*p (active) + (n*m*m*p*p*n - n*m*p) (zero)
    # But we can be smarter: group by (j, i) type
    # For each (j, i), the active constraint is for all k: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
    # For each (j', i') != (j, i), the zero constraint is: sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i',r] = 0 for all k
    
    # Actually, let's build the matrix differently.
    # The tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We need T[a,b,c] = 1 if (j(a)==j(b) and i(a)==i(c)), else 0
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Let's build A such that A @ coeffs = target
    # But the relationship is nonlinear (product of three vectors).
    # Instead, let's use the fact that if we have a valid decomposition with rank R,
    # and we want to find if there's a linear dependency, we can check if the columns
    # of the "bilinear" matrix are linearly dependent.
    
    # Alternative approach: use the structure to identify independent components.
    # Each term r contributes to T[a,b,c] = U[a,r]*V[b,r]*W[c,r]
    # For a fixed (j,i), the terms that can be non-zero are those where
    # U has support on rows with j-component = j, V on rows with j-component = j,
    # W on rows with i-component = i.
    
    # Let's try a simpler approach: check if removing any single term still gives a valid decomposition.
    # If so, that term is redundant.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R
    
    # Try removing each term one at a time
    for remove_idx in range(R):
        if time.time() - _get_start_time() > _DEADLINE - 5:
            break
        
        U_test = np.delete(U, remove_idx, axis=1)
        V_test = np.delete(V, remove_idx, axis=1)
        W_test = np.delete(W, remove_idx, axis=1)
        
        if _verify_decomposition(U_test, V_test, W_test, n, m, p, tol):
            best_U, best_V, best_W = U_test, V_test, W_test
            best_rank = R - 1
            # After removing one, try again from the reduced set
            U, V, W = U_test, V_test, W_test
            R = best_rank
            remove_idx = 0  # restart from beginning
            continue
    
    if best_rank < R:
        return best_U, best_V, best_W
    return None


def _get_start_time():
    if not hasattr(_get_start_time, '_start'):
        _get_start_time._start = time.time()
    return _get_start_time._start


def construct():
    """Find a low-rank tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    _get_start_time._start = time.time()
    start_time = time.time()
    
    n, m, p = 4, 4, 6
    
    # Build the rank-77 structured decomposition
    U, V, W = _build_structured_rank77()
    
    # Verify it's correct
    if not _verify_decomposition(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Try null space reduction
    reduced = _try_nullspace_reduction(U, V, W)
    if reduced is not None:
        U, V, W = reduced
    
    # Convert to lists with integer entries where possible
    def to_list(mat):
        mat = np.array(mat)
        out = []
        for row in mat:
            out_row = []
            for x in row:
                if abs(x - round(x)) < 1e-9:
                    out_row.append(int(round(x)))
                else:
                    out_row.append(float(x))
            out.append(out_row)
        return out
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
