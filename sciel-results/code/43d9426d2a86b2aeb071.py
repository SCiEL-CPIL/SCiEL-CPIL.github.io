"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses structured random search with ALS refinement to find lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with 10s safety margin

    # Schoolbook fallback
    def get_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition with tolerance for half-integer arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integers(matrix):
        """Snap float matrix to nearest half-integer."""
        return [[int(round(val * 2)) / 2.0 for val in row] for row in matrix]

    def run_als(U, V, W, max_iter=150, tol=1e-10):
        """
        Run ALS iterations to minimize error.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]

        for iteration in range(max_iter):
            if time.time() > deadline:
                return False, U, V, W

            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k] = V[j * p + k] * W[k * n + i]
                    try:
                        u_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j] = u_sol
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i] = U[i * m + j] * W[k * n + i]
                    try:
                        v_sol, residuals, rank_sol, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V[j * p + k] = v_sol
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j] = U[i * m + j] * V[j * p + k]
                    try:
                        w_sol, residuals, rank_sol, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W[k * n + i] = w_sol
                    except:
                        pass

            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                        err += (val - 1.0) ** 2

            if err < tol:
                return True, U, V, W

        return False, U, V, W

    def generate_structured_init(candidate_rank, seed=None):
        """
        Generate structured initial matrices that respect the tensor structure.
        Use sparse patterns with some density to help ALS converge.
        """
        if seed is not None:
            np.random.seed(seed)

        U = np.zeros((n * m, candidate_rank))
        V = np.zeros((m * p, candidate_rank))
        W = np.zeros((p * n, candidate_rank))

        # For each rank index, assign non-zero entries based on a structured pattern
        for r in range(candidate_rank):
            # Choose which (i,j) pairs get non-zero U entries
            # Use a random subset but ensure coverage
            num_u_entries = max(1, candidate_rank // 3)
            u_indices = np.random.choice(n * m, size=min(num_u_entries, n * m), replace=False)
            for idx in u_indices:
                U[idx, r] = np.random.choice([-1.0, 1.0])

            # Choose which (j,k) pairs get non-zero V entries
            num_v_entries = max(1, candidate_rank // 4)
            v_indices = np.random.choice(m * p, size=min(num_v_entries, m * p), replace=False)
            for idx in v_indices:
                V[idx, r] = np.random.choice([-1.0, 1.0])

            # Choose which (k,i) pairs get non-zero W entries
            num_w_entries = max(1, candidate_rank // 4)
            w_indices = np.random.choice(p * n, size=min(num_w_entries, p * n), replace=False)
            for idx in w_indices:
                W[idx, r] = np.random.choice([-1.0, 1.0])

        return U, V, W

    def generate_random_init(candidate_rank, scale=0.5, seed=None):
        """Generate random initial matrices."""
        if seed is not None:
            np.random.seed(seed)

        U = np.random.randn(n * m, candidate_rank) * scale
        V = np.random.randn(m * p, candidate_rank) * scale
        W = np.random.randn(p * n, candidate_rank) * scale
        return U, V, W

    # Try to find lower rank decompositions
    best_result = None
    best_rank = schoolbook_rank

    # Try ranks from 30 down to 24
    for candidate_rank in range(30, 23, -1):
        if time.time() > deadline:
            break

        # Try multiple random restarts for this rank
        max_restarts = 8 if candidate_rank > 26 else 12

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Alternate between structured and random initializations
            if restart % 2 == 0:
                U, V, W = generate_structured_init(candidate_rank, seed=restart * 1000 + candidate_rank)
            else:
                U, V, W = generate_random_init(candidate_rank, scale=0.3, seed=restart * 1000 + candidate_rank + 1)

            # Run ALS
            converged, U_final, V_final, W_final = run_als(U, V, W, max_iter=100, tol=1e-10)

            if converged:
                # Snap to half-integers
                U_snap = snap_to_half_integers(U_final.tolist())
                V_snap = snap_to_half_integers(V_final.tolist())
                W_snap = snap_to_half_integers(W_final.tolist())

                # Verify exactness of snapped version
                if verify_decomposition(U_snap, V_snap, W_snap, candidate_rank):
                    best_result = {
                        "U": U_snap,
                        "V": V_snap,
                        "W": W_snap
                    }
                    best_rank = candidate_rank
                    break

        if best_result is not None:
            break

    if best_result is not None:
        return best_result
    else:
        return get_schoolbook()
# EVOLVE-BLOCK-END
