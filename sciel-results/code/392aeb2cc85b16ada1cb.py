"""Matrix multiplication 2,3,8: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix mult."""
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline with margin (evaluator kills at 150s)

    n, m, p = DIMENSIONS

    # Build the target tensor non-zeros
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                non_zeros.append((u, v, w))

    K = len(non_zeros)
    U_dims = n * m
    V_dims = m * p
    W_dims = p * n

    # Precompute indices grouped by each mode
    indices_by_w = {i: [] for i in range(W_dims)}
    indices_by_u = {i: [] for i in range(U_dims)}
    indices_by_v = {i: [] for i in range(V_dims)}

    for idx, (u, v, w) in enumerate(non_zeros):
        indices_by_w[w].append(idx)
        indices_by_u[u].append(idx)
        indices_by_v[v].append(idx)

    def verify_decomp(U, V, W):
        """
        Checks if U, V, W exactly decompose the tensor.
        Returns True if valid.
        """
        r = U.shape[1]
        # Check all non-zero entries
        for u, v, w in non_zeros:
            # Sum over c: U[u, c] * V[v, c] * W[w, c]
            prod = np.sum(U[u, :] * V[v, :] * W[w, :])
            if abs(prod - 1.0) > 1e-9:
                return False

        # Check all zero entries
        nz_set = set(non_zeros)
        for u in range(U_dims):
            for v in range(V_dims):
                for w in range(W_dims):
                    if (u, v, w) not in nz_set:
                        prod = np.sum(U[u, :] * V[v, :] * W[w, :])
                        if abs(prod) > 1e-9:
                            return False
        return True

    def round_to_half(X):
        # Round to nearest 0.5
        return np.round(X * 2) / 2

    def al_solve(U, V, W, rank, iters=30, tol=1e-12):
        """
        Alternating Least Squares for CP decomposition.
        Minimizes sum (T - sum U_c V_c W_c)^2
        """
        for it in range(iters):
            # Update W
            for w in range(W_dims):
                if not indices_by_w[w]:
                    continue
                idxs = indices_by_w[w]
                # Construct A matrix: rows are idxs, cols are c
                # A[r, c] = U[u_r, c] * V[v_r, c]
                A = np.empty((len(idxs), rank))
                for r, idx in enumerate(idxs):
                    u, v, _ = non_zeros[idx]
                    A[r, :] = U[u, :] * V[v, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[w, :] = sol

            # Update U
            for u in range(U_dims):
                if not indices_by_u[u]:
                    continue
                idxs = indices_by_u[u]
                A = np.empty((len(idxs), rank))
                for r, idx in enumerate(idxs):
                    _, v, w = non_zeros[idx]
                    A[r, :] = V[v, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[u, :] = sol

            # Update V
            for v in range(V_dims):
                if not indices_by_v[v]:
                    continue
                idxs = indices_by_v[v]
                A = np.empty((len(idxs), rank))
                for r, idx in enumerate(idxs):
                    u, _, w = non_zeros[idx]
                    A[r, :] = U[u, :] * W[w, :]
                b = np.ones(len(idxs))
                sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[v, :] = sol

            # Check convergence
            if it > 0:
                # Calculate error
                err = 0.0
                for idx in range(K):
                    u, v, w = non_zeros[idx]
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    err += (val - 1.0)**2
                if err < tol:
                    break

        return U, V, W

    def try_rank(rank, max_restarts=5):
        """Try to find a valid decomposition of given rank."""
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None

            # Initialize with random values
            U = np.random.randn(U_dims, rank)
            V = np.random.randn(V_dims, rank)
            W = np.random.randn(W_dims, rank)

            # Run ALS
            U, V, W = al_solve(U, V, W, rank, iters=30)

            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)

            # Verify
            if verify_decomp(U_r, V_r, W_r):
                return U_r, V_r, W_r

            # Also try without rounding in case the solution is already integer/half-integer
            if verify_decomp(U, V, W):
                return U, V, W

        return None

    # Try ranks from low to high
    # Schoolbook rank is n*m*p = 48
    # We try ranks below schoolbook to see if we can find a better one
    # The rank of matrix multiplication is at least max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But it's likely much higher. Let's try from 24 up to 47.
    # However, this is computationally expensive. Let's be strategic.
    # We know rank 48 works (schoolbook). Let's try to find if rank < 48 works.
    # Given the small dimensions, we can afford to try a few ranks.
    
    # Start from a reasonable lower bound and go up
    # Actually, let's just try a few promising ranks.
    # If we don't find anything better, we return the schoolbook decomposition.
    
    best_result = None

    # Try ranks from 30 to 47 (descending is not useful, ascending is)
    # But we want the MINIMAL rank. So we should try 30, 31, 32...
    # If rank 35 works, we return it.
    # If none work, we return rank 48.
    
    # Let's try a subset of ranks to save time.
    # We'll try ranks 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
    # If any of these work, we return the lowest one.
    # If none work, we return rank 48.
    
    # Actually, let's try from 30 to 47, but with fewer restarts for lower ranks
    # and more for higher ranks? No, let's just try a few.
    
    # Given the time constraint, let's try ranks 40, 42, 44, 46, 47
    # If none work, return 48.
    
    ranks_to_try = [47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    # We want the minimal rank, so we should try from low to high.
    # But if we try from low to high, we might waste time on impossible ranks.
    # Let's try from high to low, and keep the best (lowest) rank found.
    # No, that's wrong. We want the MINIMAL rank.
    # So we should try from low to high, and stop as soon as we find a valid one.
    
    # But if rank 30 is impossible, we waste time.
    # Let's try a few ranks in the middle first.
    
    # Actually, let's just try from 30 to 47, and return the first valid one.
    # If none are valid, return 48.
    
    for rank in range(30, 48):
        if time.time() > deadline - 10:  # Leave 10s for schoolbook
            break
        
        result = try_rank(rank, max_restarts=3)
        if result is not None:
            U_r, V_r, W_r = result
            # Convert to lists
            U_list = [[float(U_r[i, j]) for j in range(rank)] for i in range(U_dims)]
            V_list = [[float(V_r[i, j]) for j in range(rank)] for i in range(V_dims)]
            W_list = [[float(W_r[i, j]) for j in range(rank)] for i in range(W_dims)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook decomposition
    rank = n * m * p
    U = np.zeros((U_dims, rank))
    V = np.zeros((V_dims, rank))
    W = np.zeros((W_dims, rank))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    U_list = [[int(U[i, j]) for j in range(rank)] for i in range(U_dims)]
    V_list = [[int(V[i, j]) for j in range(rank)] for i in range(V_dims)]
    W_list = [[int(W[i, j]) for j in range(rank)] for i in range(W_dims)]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
