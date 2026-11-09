import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T
    T = build_tensor()

    def check_decomposition(U, V, W_mat, tol=1e-9):
        """Check if the decomposition is valid using einsum (fast)."""
        result = np.einsum('ac,bc,dc->abd', U, V, W_mat)
        return float(np.max(np.abs(result - T))) < tol

    def solve_for_W(U, V, R):
        """Given U and V, solve for W using least squares."""
        A_sys = np.zeros((n * m * p, R * n * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for r in range(R):
                        col = r * n * p + k * n + i
                        A_sys[row, col] = U[i * m + j, r] * V[j * p + k, r]
        b_vec = np.ones(n * m * p)
        W_flat, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
        W_mat = np.zeros((p * n, R))
        for r in range(R):
            for i in range(n):
                for k in range(p):
                    W_mat[k * n + i, r] = W_flat[r * n * p + k * n + i]
        return W_mat

    def is_rational_half_integer(M, tol=1e-6):
        """Check if matrix entries are close to half-integers."""
        for val in M.flatten():
            if abs(val * 2 - round(val * 2)) > tol:
                return False
        return True

    def snap_to_half_integers(M):
        """Snap entries to nearest half-integer."""
        return np.round(M * 2) / 2.0

    best_rank = n * m * p
    best_result = get_schoolbook()

    # Try ranks from 23 downwards (since 24 is schoolbook)
    # We focus on 23 and 22 as they are the most likely improvements if any exist.
    for R in range(23, 8, -1):
        if time.time() > deadline - 20:
            break

        found = False
        np.random.seed(42 + R)

        # Discrete search: Randomize U and V with small coefficients, solve for W
        # This is a structural change from gradient descent to discrete sampling + exact solve
        num_trials = 5000
        for t in range(num_trials):
            if time.time() > deadline - 5:
                break

            # Initialize U and V with random sparse entries from {-1, 0, 1}
            # Sparsity helps in finding structured solutions
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            
            # Each column of U and V should have some non-zero entries
            for c in range(R):
                # Pick random rows to be non-zero
                nz_u = np.random.choice(n * m, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(m * p, size=np.random.randint(1, 4), replace=False)
                
                U[nz_u, c] = np.random.choice([-1, 1], size=len(nz_u))
                V[nz_v, c] = np.random.choice([-1, 1], size=len(nz_v))

            # Solve for W
            W_mat = solve_for_W(U, V, R)
            
            # Check if W is close to half-integers
            if is_rational_half_integer(W_mat, tol=1e-5):
                # Snap and verify
                W_snapped = snap_to_half_integers(W_mat)
                U_snapped = snap_to_half_integers(U)
                V_snapped = snap_to_half_integers(V)
                
                if check_decomposition(U_snapped, V_snapped, W_snapped, tol=1e-9):
                    if R < best_rank:
                        best_rank = R
                        best_result = {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
                    found = True
                    break

        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
