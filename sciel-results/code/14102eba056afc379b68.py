import numpy as np
import time
import random

def construct():
    n, m, p = 3, 4, 7
    rank_schoolbook = n * m * p  # 84
    start_time = time.time()
    deadline = start_time + 110  # safety margin: return by 110s, limit is 120s

    # Build the matrix multiplication tensor T[i,j,k] = delta (it's 1 for all valid i,j,k)
    # The tensor has shape (n*m, m*p, p*n) in the factored form
    # U shape: (n*m, r), V shape: (m*p, r), W shape: (p*n, r)
    # The reconstruction is: for each column c, U[:,c] * V[:,c] * W[:,c] outer product
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i,j,k

    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition exactly reconstructs the tensor."""
        # For each (i,j,k), compute sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # This should equal 1 for all valid i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_fast(U, V, W, tol=1e-10):
        """Faster verification using numpy."""
        r = U.shape[1]
        # Build the full tensor via einsum
        # T[i*m+j, j*p+k, k*n+i] should be 1
        # Compute: for each c, outer product
        # More efficient: reshape and use matrix operations
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # For a fixed (i,j,k), the value is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Let's compute this using broadcasting for all i,j,k at once
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # For each (i,j,k): sum_c U3[i,j,c]*V3[j,k,c]*W3[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return np.allclose(T, 1.0, atol=tol)

    # Try to find a lower rank decomposition using a simple approach
    # Strategy: Start with schoolbook and try to eliminate terms
    # or use a randomized approach with random hyperplanes

    # Approach: Try various rank values from rank_schoolbook-1 downward
    # For each rank, try to find a solution using iterative refinement

    best_rank = rank_schoolbook
    best_result = (U_school.copy(), V_school.copy(), W_school.copy())

    # Try to reduce rank by combining terms
    # For 2x2x2, Strassen shows rank 7 < 8
    # For our case, let's try a structured approach

    # Method 1: Try rank = rank_schoolbook - d for small d
    # Use a greedy/local search approach

    # Actually, let's try a more principled approach:
    # The tensor decomposition problem can be attacked via:
    # 1. Random initialization + iterative refinement
    # 2. Exploiting symmetry/structure

    # Let's try: initialize U, V, W randomly and use ALS (alternating least squares)
    # to minimize the residual ||T_reconstructed - T_target||_F

    def als_decomposition(target_tensor, max_rank, max_iter=200, tol=1e-12):
        """
        Try to find a decomposition of rank at most max_rank using ALS.
        target_tensor: shape (n*m, m*p, p*n) with 1s
        Returns (U, V, W) or None if no solution found.
        """
        r = max_rank
        np.random.seed(42)
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)

        for it in range(max_iter):
            # Update U: for each (i,j), solve least squares
            # T_target[i*m+j, :, :] = sum_c U[i*m+j, c] * (V[:, c] * W[:, c])
            # Let A_c = V[:, c] * W[:, c], shape (m*p, p*n)
            # Then T_target[i*m+j, :, :].flatten() = sum_c U[i*m+j, c] * A_c.flatten()
            # So we solve: [A_1.flatten() | A_2.flatten() | ... | A_r.flatten()] @ u = t
            # where t is the target vector

            # Build the system for U
            # A is (m*p*p*n, r) where A[:,c] = (V[:,c] * W[:,c]).flatten()
            A_U = np.outer(V.flatten(), W.flatten())  # shape (m*p*p*n, r)
            # Actually: A_U[:, c] = (V[:, c] * W[:, c]).flatten()
            # V[:, c] has shape (m*p,), W[:, c] has shape (p*n,)
            # Their element-wise product has shape... wait, they're different lengths unless m*p == p*n
            # m*p = 4*7 = 28, p*n = 7*3 = 21. They're different!
            # So we can't simply do outer product.
            # We need: for each (j,k) index in V and (k,i) index in W...
            # Actually the target is indexed by (i,j,k), and:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For fixed (i,j), varying k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1 for all k
            # Let b_c(k) = V[j*p+k, c] * W[k*n+i, c]
            # Then sum_c U[i*m+j, c] * b_c(k) = 1 for k=0..p-1
            # This is a system of p equations in r unknowns (the U[i*m+j, :] vector)
            # For fixed (i,j), build matrix B_{(i,j)} of shape (p, r) where B_{(i,j)}[k,c] = V[j*p+k,c]*W[k*n+i,c]
            # Solve B @ u = 1 (vector of ones of length p)

            for i in range(n):
                for j in range(m):
                    # Build B of shape (p, r)
                    B = np.zeros((p, r))
                    for k in range(p):
                        B[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve B @ u = ones(p)
                    # Use least squares
                    try:
                        u, _, _, _ = np.linalg.lstsq(B, np.ones(p), rcond=None)
                    except:
                        continue
                    U[i * m + j, :] = u

            # Similarly update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    except:
                        continue
                    V[j * p + k, :] = v

            # Similarly update W
            for k in range(p):
                for i in range(n):
                    B = np.zeros((m, r))
                    for j in range(m):
                        B[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        w, _, _, _ = np.linalg.lstsq(B, np.ones(m), rcond=None)
                    except:
                        continue
                    W[k * n + i, :] = w

            # Check residual
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            residual = np.max(np.abs(T - 1.0))
            if residual < tol:
                return U, V, W

            if time.time() > deadline - 10:
                break

        return None

    # Try ranks below schoolbook
    for r in range(rank_schoolbook - 1, 40, -1):
        if time.time() > deadline - 20:
            break

        result = als_decomposition(None, r, max_iter=100, tol=1e-10)
        if result is not None:
            U_sol, V_sol, W_sol = result
            if verify_fast(U_sol, V_sol, W_sol):
                best_rank = r
                best_result = (U_sol, V_sol, W_sol)
                break  # Found a solution, no need to try higher ranks... wait, we want minimum
                # Actually we're going downward, so first success is minimum found
                break

    if time.time() > deadline - 15:
        # Time to return
        U_out, V_out, W_out = best_result
        return {
            "U": U_out.tolist(),
            "V": V_out.tolist(),
            "W": W_out.tolist(),
        }

    # If we haven't found anything better, return schoolbook
    U_out, V_out, W_out = best_result
    return {
        "U": U_out.tolist(),
        "V": V_out.tolist(),
        "W": W_out.tolist(),
    }
