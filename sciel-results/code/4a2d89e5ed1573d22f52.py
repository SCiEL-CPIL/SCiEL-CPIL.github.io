"""Rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.

Strategy:
  1. Start with the schoolbook decomposition (rank 48).
  2. Perform a sequence of local "rank-reduction" attempts: for each candidate rank r
     below 48, try to find a decomposition by iteratively optimizing the factor matrices.
  3. Use a deterministic construction based on the structure of the problem.

For 2x3 by 3x8, the tensor has dimensions 6 x 24 x 16.
The known minimal rank is not trivially below 48, but we attempt to find it.

We use a simple iterative approach:
  - Initialize with a random (but deterministic seed) decomposition of rank r.
  - Use alternating least squares / gradient descent to minimize the error.
  - If the error reaches 0 (within exact integer/half-integer constraints), we have a solution.

Since exact integer/half-integer entries are required, we round and verify at the end.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]
DEADLINE_SECONDS = 110  # Safety margin before 120s limit

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16

    # The matrix multiplication tensor T has entries T[i*m+j, j'*p+k, k'*n+i] = 1 if i=i', j=j', k=k'
    # i.e., T[a, b, c] = 1 where a = i*m+j, b = j'*p+k, c = k'*n+i, and i=i', j=j', k=k'

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def make_tensor(r):
        """Create the matrix multiplication tensor of shape (nm, mp, pn)."""
        T = np.zeros((nm, mp, pn), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T

    T = make_tensor(0)

    def verify(U, V, W, tol=1e-9):
        """Check if U @ V @ W^T approximates T exactly (within tol)."""
        # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # Compute: for each r, outer product of U[:,r] and V[:,r] gives (nm, mp), then multiply by W[:,r]
        # We can do: sum over r of U[:,r] * V[:,r] * W[:,r] = np.einsum('ar,br,cr->abc', U, V, W)
        try:
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(reconstructed, T, atol=tol):
                return True
            return False
        except Exception:
            return False

    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(2 * x) / 2.0

    def try_rank(r, max_iter=500, seed=42):
        """Try to find a rank-r decomposition using ALS-style optimization."""
        if time.time() - start_time > DEADLINE_SECONDS:
            return None

        rng = np.random.RandomState(seed)

        # Initialize with small random values
        U = rng.randn(nm, r) * 0.5
        V = rng.randn(mp, r) * 0.5
        W = rng.randn(pn, r) * 0.5

        best_error = np.inf
        best_U, best_V, best_W = None, None, None

        for iteration in range(max_iter):
            if time.time() - start_time > DEADLINE_SECONDS:
                break

            # Update U: for each column a, U[a, :] = sum_{b,c} T[a,b,c] * V[b,:] * W[c,:] / sum_{b,c} (V[b,:]^2 * W[c,:]^2 + eps)
            # Actually, the least squares solution for U given V, W:
            # min_U sum_{a,b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
            # d/dU[a,r] = -2 sum_{b,c} T[a,b,c] V[b,r] W[c,r] + 2 U[a,r] sum_{b,c} V[b,r]^2 W[c,r]^2
            # So U[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2

            # Vectorized:
            # V[b,r] * W[c,r] for all b,c,r -> shape (mp, pn, r)
            VW = np.einsum('br,cr->bcr', V, W)  # (mp, pn, r)
            # T[a,b,c] * VW[b,c,r] -> sum over b,c -> shape (nm, r)
            numerator_U = np.einsum('abc,bcr->ar', T, VW)
            # sum_{b,c} V[b,r]^2 W[c,r]^2 -> shape (r,)
            denom_U = np.einsum('br,cr->r', V**2, W**2)
            denom_U = np.maximum(denom_U, 1e-12)
            U = numerator_U / denom_U

            # Update V:
            UW = np.einsum('ar,cr->acr', U, W)  # (nm, pn, r)
            numerator_V = np.einsum('abc,acr->br', T, UW)
            denom_V = np.einsum('ar,cr->r', U**2, W**2)
            denom_V = np.maximum(denom_V, 1e-12)
            V = numerator_V / denom_V

            # Update W:
            UV = np.einsum('ar,br->abr', U, V)  # (nm, mp, r)
            numerator_W = np.einsum('abc,abr->cr', T, UV)
            denom_W = np.einsum('ar,br->r', U**2, V**2)
            denom_W = np.maximum(denom_W, 1e-12)
            W = numerator_W / denom_W

            # Compute error
            reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.max(np.abs(reconstructed - T))
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if error < 1e-10:
                break

        # Check if we found an exact solution
        if best_error < 1e-10:
            # Round to half-integers
            U_r = round_to_half(best_U)
            V_r = round_to_half(best_V)
            W_r = round_to_half(best_W)
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
            # Try without rounding first
            if verify(best_U, best_V, best_W, tol=1e-9):
                # Entries are already close to integers
                U_r = np.round(best_U)
                V_r = np.round(best_V)
                W_r = np.round(best_W)
                if verify(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r

        return None

    # Try ranks from 24 up to 47 (since 48 is schoolbook)
    # Use a few different seeds for diversity
    best_result = None
    for r in range(24, 48):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        for seed in [42, 123, 777, 2024]:
            if time.time() - start_time > DEADLINE_SECONDS - 3:
                break
            result = try_rank(r, max_iter=300, seed=seed)
            if result is not None:
                U, V, W = result
                best_result = (U, V, W, r)
                break
        if best_result is not None:
            break

    if best_result is not None:
        U, V, W, r = best_result
        # Convert to lists of floats (which are integers or half-integers)
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
