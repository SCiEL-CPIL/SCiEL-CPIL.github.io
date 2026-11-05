import time
import numpy as np
from itertools import product as iproduct

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 140.0
START_TIME = time.time()


def _time_left():
    return TIME_LIMIT - (time.time() - START_TIME)


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 2x4 * 4x4 matrix multiplication.
    Returns {'U': (8 x rank), 'V': (16 x rank), 'W': (8 x rank)} with
    integer or half-integer entries.
    """
    best = None
    best_rank = 32

    # Try to find a lower-rank decomposition via random search + local refinement
    try:
        result = _search_low_rank()
        if result is not None:
            rank = result['rank']
            if rank < best_rank:
                best = result
                best_rank = rank
    except Exception:
        pass

    # Fallback: rank-28 Strassen
    if best is None or best_rank > 28:
        result = _strassen_28()
        if result is not None:
            best = result
            best_rank = 28

    if best is None:
        best = _schoolbook_32()

    return best


def _schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist(), 'rank': rank}


def _strassen_28():
    """
    Rank-28 decomposition via 4 independent 2x2 Strassen blocks.
    A (2x4) = [A1 | A2], B (4x4) = [[B11,B12],[B21,B22]]
    C = A*B: C[:,0:2] = A1*B11 + A2*B21, C[:,2:4] = A1*B12 + A2*B22
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

    # Strassen 2x2 terms: (U_coeffs, V_coeffs, W_coeffs)
    # Local 2x2 indices: 0=top-left, 1=top-right, 2=bot-left, 3=bot-right
    st = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # 4 sub-products with index mappings
    subprods = [
        # A_left * B_tl -> C_left
        (lambda r, c: r * 4 + c,
         lambda r, c: r * 4 + c,
         lambda r, c: c * 2 + r),
        # A_right * B_bl -> C_left
        (lambda r, c: r * 4 + c + 2,
         lambda r, c: (r + 2) * 4 + c,
         lambda r, c: c * 2 + r),
        # A_left * B_tr -> C_right
        (lambda r, c: r * 4 + c,
         lambda r, c: r * 4 + c + 2,
         lambda r, c: (c + 2) * 2 + r),
        # A_right * B_br -> C_right
        (lambda r, c: r * 4 + c + 2,
         lambda r, c: (r + 2) * 4 + c + 2,
         lambda r, c: (c + 2) * 2 + r),
    ]

    col = 0
    for a_map, b_map, c_map in subprods:
        for u_c, v_c, w_c in st:
            for loc, coeff in u_c.items():
                r, c = divmod(loc, 2)
                U[a_map(r, c), col] += coeff
            for loc, coeff in v_c.items():
                r, c = divmod(loc, 2)
                V[b_map(r, c), col] += coeff
            for loc, coeff in w_c.items():
                r, c = divmod(loc, 2)
                W[c_map(r, c), col] += coeff
            col += 1

    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist(), 'rank': rank}


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U @ diag @ V^T reconstructs the matmul tensor via W."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # The tensor T[i,j,k] = delta_{i,k'} ... 
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # In tensor form: T[i,j,k] = 1 for all (i,j,k) since it's the standard matmul tensor
    # The decomposition: T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    # So for each (i,j,k): sum_t U[i,t]*V[j,t]*W[k,t] should equal T[i,j,k] = 1 if valid

    # Build the target tensor
    T = np.ones((n, m, p), dtype=float)

    # Compute decomposed tensor
    # T_dec[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
    # Reshape: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, t], V[j*p+k, t], W[k*n+i, t]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol


def _search_low_rank():
    """
    Attempt to find a decomposition with rank < 28 using randomized
    initialization and local optimization. Returns None if unsuccessful.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    nU = n * m  # 8
    nV = m * p  # 16
    nW = p * n  # 8

    if _time_left() < 20:
        return None

    best_err = float('inf')
    best_state = None

    np.random.seed(42)

    for attempt in range(30):
        if _time_left() < 15:
            break

        U = np.random.randn(nU, target_rank) * 0.5
        V = np.random.randn(nV, target_rank) * 0.5
        W = np.random.randn(nW, target_rank) * 0.5

        # Local optimization using gradient-free (Nelder-Mead style via random perturbations)
        err = _compute_error(U, V, W, n, m, p)
        best_local_err = err
        best_local = (U.copy(), V.copy(), W.copy())

        for step in range(200):
            if _time_left() < 10:
                break

            # Random perturbation
            perturb = 0.1 / (1 + step * 0.01)
            dU = np.random.randn(nU, target_rank) * perturb
            dV = np.random.randn(nV, target_rank) * perturb
            dW = np.random.randn(nW, target_rank) * perturb

            U_new = U + dU
            V_new = V + dV
            W_new = W + dW

            err_new = _compute_error(U_new, V_new, W_new, n, m, p)

            if err_new < best_local_err:
                best_local_err = err_new
                best_local = (U_new.copy(), V_new.copy(), W_new.copy())
                U, V, W = U_new, V_new, W_new
            else:
                # Decay step size implicitly via the perturb formula
                pass

        if best_local_err < best_err:
            best_err = best_local_err
            best_state = best_local

        if best_err < 1e-6:
            # Found a valid decomposition; try to round to half-integers
            U_b, V_b, W_b = best_state
            U_r, V_r, W_r = _round_to_half_int(U_b, V_b, W_b)
            if _verify_decomposition(U_r, V_r, W_r):
                return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist(),
                        'rank': target_rank}
            # If rounding failed, try exact solve
            U_r, V_r, W_r = _exact_solve_from_init(U_b, V_b, W_b, n, m, p)
            if U_r is not None and _verify_decomposition(U_r, V_r, W_r):
                return {'U': U_r.tolist(), 'V': V_r.tolist(), 'W': W_r.tolist(),
                        'rank': target_rank}

    return None


def _compute_error(U, V, W, n, m, p):
    """Compute max error of tensor reconstruction."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _round_to_half_int(U, V, W):
    """Round entries to nearest half-integer."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r


def _exact_solve_from_init(U, V, W, n, m, p):
    """
    Given approximate U, V, W, try to find exact rational solutions.
    Fix U and V, solve for W exactly. Then check validity.
    """
    rank = U.shape[1]
    nU = n * m
    nV = m * p
    nW = p * n

    # For each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # Fix U and V, solve linear system for W
    # Variable: W[k*n+i, t] for all (k,i,t)
    # Equation: for each (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1

    # Number of unknowns: nW * rank
    # Number of equations: n * m * p
    # If rank >= n*m*p/nW ... this is underdetermined generally.
    # Try to find a solution via least squares first, then check.

    A_mat = np.zeros((n * m * p, nW * rank))
    b_vec = np.ones(n * m * p)

    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for t in range(rank):
                    col = (k * n + i) * rank + t
                    A_mat[eq_idx, col] = U[i * m + j, t] * V[j * p + k, t]
                eq_idx += 1

    # Solve least squares
    W_flat, residuals, rank_lev, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    W_sol = W_flat.reshape(nW, rank)

    # Verify
    if _verify_decomposition(U, V, W_sol, tol=1e-8):
        # Try to round
        U_r, V_r, W_r = _round_to_half_int(U, V, W_sol)
        if _verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        # Return as-is (floats may be accepted)
        return U, V, W_sol

    return None, None, None


if __name__ == '__main__':
    result = construct()
    print(f"Rank: {result['rank']}")
    # Verify
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    n, m, p = DIMENSIONS
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
