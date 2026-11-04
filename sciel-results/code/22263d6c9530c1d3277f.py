"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Convention: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k.
    """
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False

    # Use numpy for speed
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _verify_numpy(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
    """Fast numpy verification. Returns (is_valid, max_error)."""
    rank = U_mat.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = U_mat[i * m + j]
            for k in range(p):
                v_row = V_mat[j * p + k]
                w_row = W_mat[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
                if err > tol:
                    return False, max_err
    return True, max_err

def _try_random_rank(n, m, p, rank, n_tries=20, time_budget=5.0):
    """
    Try to find a valid decomposition of the given rank via random initialization
    and iterative refinement (gradient-free local search).
    Returns (U, V, W) as lists if found, else None.
    """
    start = time.time()
    deadline = start + time_budget

    U0 = np.array([[0] * rank for _ in range(n * m)], dtype=float)
    V0 = np.array([[0] * rank for _ in range(m * p)], dtype=float)
    W0 = np.array([[0] * rank for _ in range(p * n)], dtype=float)

    for attempt in range(n_tries):
        if time.time() > deadline:
            break

        # Random initialization: sparse-ish to help convergence
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1

        # Iterative local search: perturb one factor at a time
        best_err = float('inf')
        for it in range(500):
            if time.time() > deadline:
                break

            # Compute current error
            max_err = 0.0
            # Sample a subset of (i,j,k) triples for speed
            samples = []
            for _ in range(min(200, n * m * p)):
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                samples.append((i, j, k))

            for i, j, k in samples:
                val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

            if max_err < 1e-7:
                # Full verification
                valid, full_err = _verify_numpy(U, V, W, n, m, p, tol=1e-7)
                if valid:
                    # Convert to lists
                    return (U.tolist(), V.tolist(), W.tolist())

            # If error is not improving, try a perturbation
            # Pick a random factor and a random entry to perturb
            factor = random.choice(['U', 'V', 'W'])
            if factor == 'U':
                row = random.randint(0, n * m - 1)
                col = random.randint(0, rank - 1)
                U[row, col] += random.gauss(0, 0.05)
            elif factor == 'V':
                row = random.randint(0, m * p - 1)
                col = random.randint(0, rank - 1)
                V[row, col] += random.gauss(0, 0.05)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, rank - 1)
                W[row, col] += random.gauss(0, 0.05)

    return None

def _try_structural_reduction(n, m, p, schoolbook_U, schoolbook_V, schoolbook_W):
    """
    Try to reduce the schoolbook decomposition by exploiting structure.
    For 4x5x6, the schoolbook rank is 120. We try to find linear dependencies
    among the rank-1 components.
    
    A rank-1 component is identified by (i, j, k) with:
    U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    
    We look for a linear combination of components that sums to zero on all
    (i', j', k') triples. This is a system of linear equations.
    """
    start = time.time()
    deadline = start + 30.0

    n_comp = n * m * p
    U_arr = np.array(schoolbook_U, dtype=float)
    V_arr = np.array(schoolbook_V, dtype=float)
    W_arr = np.array(schoolbook_W, dtype=float)

    # Each component c corresponds to (i_c, j_c, k_c)
    # The constraint for a linear combination sum_r alpha_r * component_r = 0 is:
    # For each (i, j, k): sum_{r: i_r=i, j_r=j, k_r=k} alpha_r = 0
    # But each (i,j,k) has exactly one component, so this is trivially satisfied
    # only if all alpha_r = 0. So no linear dependencies in the schoolbook.
    #
    # Instead, we need to find a NEW decomposition with fewer components.
    # This is a hard problem. Let's try a different approach:
    # Use the fact that for small matrices, we can try to find a decomposition
    # using known algorithms or heuristics.

    # For now, return None to indicate no reduction found
    return None

def construct():
    """
    Returns the best decomposition found for 4x5x6 matrix multiplication.
    Falls back to schoolbook (rank 120) if no better decomposition is found.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 120

    # Schoolbook decomposition
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = schoolbook_rank

    # Try random search for lower ranks
    # Lower bound: max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    lower_bound = max(n * p, n * m, m * p)

    # Try ranks from lower_bound up to best_rank - 1
    # Focus on ranks that are plausible
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break

        # Only try a few ranks to save time
        if target_rank % 5 != 0 and target_rank != lower_bound:
            continue

        # Try random search
        result = _try_random_rank(n, m, p, target_rank, n_tries=10, time_budget=2.0)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = target_rank
            break  # Found a valid decomposition, stop

    return best_result
# EVOLVE-BLOCK-END
