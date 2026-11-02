"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Searches for a rank < 36 decomposition using integer/half-integer entries
via a structured random search + local refinement, with a strict deadline
guard. Falls back to the schoolbook rank-36 construction if time runs out
or no better solution is found.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

_DEADLINE_FRACTION = 0.85  # safety margin: stop 85% through the 120s window
_SEARCH_TIMEOUT = 100.0    # hard cap in seconds for the search portion


def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i, j, k] with W[k*n+i, r] convention.

    The decomposition requires: for all i, j, k
        T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where T[i, j, k] = 1 if the (i,j,k) entry participates (it always does
    for standard matmul: C[i,k] = sum_j A[i,j] B[j,k]).
    So T[i,j,k] = 1 for all valid i, j, k.
    """
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _decompose_tensor_to_vector(T, n, m, p):
    """Flatten T to a vector in the order matching U⊗V⊗W structure."""
    return T.flatten()  # order: i, j, k -> index = i*m*p + j*p + k


def _decomposition_residual(U, V, W, T_vec, n, m, p):
    """Compute ||U⊗V⊗W - T_vec||_F for given factor matrices."""
    rank = U.shape[1]
    # Build the full tensor from the factors
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T_full = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                T_full[i, j, k] = s
    return np.linalg.norm(T_full.flatten() - T_vec)


def _round_to_half_int(arr):
    """Round array entries to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact (within tolerance)."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _schoolbook():
    """Rank-36 schoolbook construction."""
    n, m, p = DIMENSIONS
    rank = n * m * p
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
    return U, V, W


def _random_init_half_int(nrows, rank, scale=1.0):
    """Initialize a factor matrix with random half-integer entries."""
    vals = np.random.randint(-2 * scale, 2 * scale + 1, size=(nrows, rank)) * 0.5
    return vals.astype(float)


def _try_rank(n, m, p, rank, T_vec, deadline):
    """Try to find a rank-`rank` decomposition via random init + gradient-free search.

    Uses a coordinate-descent-like approach on the half-integer lattice.
    Returns (U, V, W) if a valid solution is found, else None.
    """
    if time.time() > deadline:
        return None

    best_residual = float('inf')
    best_factors = None
    n_iter = 0
    max_iter = 5000

    # Multiple random restarts
    for restart in range(20):
        if time.time() > deadline:
            break

        U = _random_init_half_int(n * m, rank, scale=1.0)
        V = _random_init_half_int(m * p, rank, scale=1.0)
        W = _random_init_half_int(p * n, rank, scale=1.0)

        # Ensure at least some non-zero entries
        for r in range(rank):
            U[np.random.randint(0, n * m), r] = 1.0
            V[np.random.randint(0, m * p), r] = 1.0
            W[np.random.randint(0, p * n), r] = 1.0

        cur_res = _decomposition_residual(U, V, W, T_vec, n, m, p)
        if cur_res < best_residual:
            best_residual = cur_res
            best_factors = (U.copy(), V.copy(), W.copy())

        # Local search: try single-coordinate half-integer steps
        for it in range(max_iter):
            if time.time() > deadline:
                break
            n_iter += 1

            # Pick a random factor and coordinate
            factor_choice = np.random.randint(0, 3)
            if factor_choice == 0:
                mat, other1, other2 = U, V, W
                nrows = n * m
            elif factor_choice == 1:
                mat, other1, other2 = V, U, W
                nrows = m * p
            else:
                mat, other1, other2 = W, U, V
                nrows = p * n

            row = np.random.randint(0, nrows)
            col = np.random.randint(0, rank)
            step = np.random.choice([-0.5, 0.5, -1.0, 1.0])

            old_val = mat[row, col]
            mat[row, col] = old_val + step

            new_res = _decomposition_residual(U, V, W, T_vec, n, m, p)
            if new_res < cur_res:
                cur_res = new_res
                if cur_res < best_residual:
                    best_residual = cur_res
                    best_factors = (U.copy(), V.copy(), W.copy())
            else:
                mat[row, col] = old_val

            # Early exit if exact
            if cur_res < 1e-9:
                if _verify_exact(U, V, W, n, m, p):
                    return U, V, W

        if best_factors is not None and best_residual < 1e-9:
            U2, V2, W2 = best_factors
            if _verify_exact(U2, V2, W2, n, m, p):
                return U2, V2, W2

    return None


def _structured_search(n, m, p, T_vec, deadline):
    """Structured search leveraging symmetry and known properties.

    For 3x3 * 3x4, the tensor has structure. Try specific patterns:
    1. Binary/half-integer matrices with small entries
    2. Exploit that T[i,j,k]=1 means we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]=1
    """
    if time.time() > deadline:
        return None

    # Try ranks from a reasonable lower bound up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Upper bound: 35 (one less than schoolbook)
    for rank in range(12, 36):
        if time.time() > deadline:
            break

        result = _try_rank(n, m, p, rank, T_vec, deadline)
        if result is not None:
            return result

    return None


def _gradient_descent_refine(U, V, W, n, m, p, T_vec, deadline, max_steps=200):
    """Polish a near-solution using gradient descent on continuous variables,
    then snap to half-integers."""
    if time.time() > deadline:
        return None

    lr = 0.01
    cur_res = _decomposition_residual(U, V, W, T_vec, n, m, p)

    for step in range(max_steps):
        if time.time() > deadline:
            break

        # Compute gradient numerically (expensive but rank is small)
        # Actually, let's use analytic gradients
        # dT/dU[i,j,r] = V[j,k,r] * W[k,i,r] summed over k... 
        # This is complex; use a simple perturbation approach instead

        # Try small perturbations
        improved = False
        for _ in range(30):
            if time.time() > deadline:
                break
            fc = np.random.randint(0, 3)
            if fc == 0:
                mat, nr = U, n * m
            elif fc == 1:
                mat, nr = V, m * p
            else:
                mat, nr = W, p * n

            row = np.random.randint(0, nr)
            col = np.random.randint(0, U.shape[1])
            old = mat[row, col]
            perturbation = np.random.randn() * lr
            mat[row, col] = old + perturbation
            new_res = _decomposition_residual(U, V, W, T_vec, n, m, p)
            if new_res < cur_res:
                cur_res = new_res
                improved = True
            else:
                mat[row, col] = old

        if not improved:
            break

    # Snap to half-integers
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)

    if _verify_exact(U_r, V_r, W_r, n, m, p):
        return U_r, V_r, W_r
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.

    Returns {'U': (9, rank), 'V': (12, rank), 'W': (12, rank)} with
    integer/half-integer entries.
    """
    t0 = time.time()
    hard_deadline = t0 + _SEARCH_TIMEOUT

    n, m, p = DIMENSIONS
    T_vec = _build_target_tensor(n, m, p).flatten()

    # Try to find a better-than-schoolbook solution
    best_result = None

    # Phase 1: Structured search for lower rank
    result = _structured_search(n, m, p, T_vec, hard_deadline)
    if result is not None:
        best_result = result

    # Phase 2: If found, try to verify and possibly improve
    if best_result is not None:
        U, V, W = best_result
        if _verify_exact(U, V, W, n, m, p):
            # Try to reduce rank further by zeroing columns
            rank = U.shape[1]
            for col in range(rank):
                if time.time() > hard_deadline:
                    break
                # Check if column can be zeroed
                if np.all(np.abs(U[:, col]) < 1e-9) and \
                   np.all(np.abs(V[:, col]) < 1e-9) and \
                   np.all(np.abs(W[:, col]) < 1e-9):
                    continue
                # Try setting this column to 0
                U[:, col] = 0
                V[:, col] = 0
                W[:, col] = 0
                if not _verify_exact(U, V, W, n, m, p):
                    # Restore
                    U[:, col] = best_result[0][:, col]
                    V[:, col] = best_result[1][:, col]
                    W[:, col] = best_result[2][:, col]

    # Fallback: schoolbook
    if best_result is None:
        U, V, W = _schoolbook()
    else:
        U, V, W = best_result

    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
