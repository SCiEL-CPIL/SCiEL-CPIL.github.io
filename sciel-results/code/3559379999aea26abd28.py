"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 138.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    # Build the multiplication tensor T[i,j,k] where:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 (it's just the structure tensor)
    # We need to find rank-r decomposition:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    # Start with schoolbook and try to find improvements
    U = np.zeros((n * m, schoolbook_rank), dtype=float)
    V = np.zeros((m * p, schoolbook_rank), dtype=float)
    W = np.zeros((p * n, schoolbook_rank), dtype=float)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    # Try to find a lower rank by checking if any columns can be removed
    # or if there's a linear dependency we can exploit
    # Strategy: Try rank r from schoolbook_rank-1 down to a lower bound
    # Lower bound for 3x3x4: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # Also Strassen-type lower bounds suggest ~27-30 for 3x3x4

    # Try a structured approach: check if we can merge some basis elements
    # For each pair of columns, check if one can be expressed as a combination
    # of others (linear dependency in the factor space)

    # First, verify schoolbook works
    def verify(U_mat, V_mat, W_mat, rank):
        # Check: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
        # For all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r] should give something that
        # when contracted with W gives the right result
        # Actually the condition is:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # = sum_j sum_r A[i,j]*U[i*m+j,r] * sum_r V[j*p+k,r]*B[j,k] * W[k*n+i,r]
        # Wait, the decomposition is:
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # where T[i,j,k] = 1 for the multiplication tensor
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i * m + j, r] * V_mat[j * p + k, r] * W_mat[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, schoolbook_rank):
        # Should not happen, but fallback
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to reduce rank by finding linear dependencies
    # A column r can be removed if for all i,j,k:
    # sum_{r' != r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] == 1
    # This is a nonlinear constraint, so let's try a different approach

    # Approach: Try to find a rank r < 36 decomposition by solving a system
    # For small r, we can try to find U,V,W such that the tensor equation holds

    # Let's try ranks from 30 down to 25, using a heuristic search
    # For each rank r, we have r variables in each of U,V,W
    # Total unknowns: r*(n*m + m*p + p*n) = r*(9+12+12) = 33r
    # Constraints: n*m*p = 36
    # So for r >= 36/33 ≈ 1.1, we have more unknowns than constraints

    # But the constraints are nonlinear (trilinear), so this is hard

    # Instead, let's try a structured approach based on known results:
    # For 3x3x4, the best known rank is 30 (from various papers)
    # But constructing such a decomposition is non-trivial

    # Let's try a greedy column elimination approach:
    # Try to see if we can express one column as a linear combination of others
    # in the sense that the tensor contribution matches

    # For a simpler approach, let's try to find if there's a rank 35 decomposition
    # by checking if any single column can be removed

    # Actually, let me try a completely different approach:
    # Use the fact that for matrix multiplication, we can use block methods

    # For 3x3 * 3x4, think of A as 3x3, B as 3x4
    # We can partition B into two 3x2 blocks: B = [B1, B2]
    # Then C = A*B = [A*B1, A*B2]
    # Each A*Bj is 3x2, which can be done with rank 3*3*2 = 18
    # Total: 36 (no gain)

    # Alternative: use Strassen-like ideas
    # For 2x2: rank 7
    # For 3x3: rank 23 (known best)
    # For 3x4: unknown, but likely around 30

    # Given the time constraints and difficulty of finding a provably lower rank,
    # let me try a randomized search for a lower rank decomposition

    # Try rank 30 with random initialization and iterative refinement
    np.random.seed(42)

    def try_rank(r, max_iters=500):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() > deadline - 5:
            return None

        # Random initialization
        U_r = np.random.randn(n * m, r) * 0.1
        V_r = np.random.randn(m * p, r) * 0.1
        W_r = np.random.randn(p * n, r) * 0.1

        best_err = float('inf')

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            # Compute the current tensor approximation
            # T_approx[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r_idx in range(r):
                            s += U_r[i*m+j, r_idx] * V_r[j*p+k, r_idx] * W_r[k*n+i, r_idx]
                        err += (s - 1.0) ** 2

            if err < best_err:
                best_err = err

            if err < 1e-10:
                break

            # Gradient step (simplified)
            # dT/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r] for each k
            # This is complex, so let's use a simpler update

            # Update each factor using the residual
            for r_idx in range(r):
                # For U: update U[i*m+j, r_idx] based on residual
                for i in range(n):
                    for j in range(m):
                        grad = 0.0
                        for k in range(p):
                            res = 0.0
                            for r2 in range(r):
                                res += U_r[i*m+j, r2] * V_r[j*p+k, r2] * W_r[k*n+i, r2]
                            res -= 1.0
                            grad += res * V_r[j*p+k, r_idx] * W_r[k*n+i, r_idx]
                        U_r[i*m+j, r_idx] -= 0.001 * grad

                # For V
                for j in range(m):
                    for k in range(p):
                        grad = 0.0
                        for i in range(n):
                            res = 0.0
                            for r2 in range(r):
                                res += U_r[i*m+j, r2] * V_r[j*p+k, r2] * W_r[k*n+i, r2]
                            res -= 1.0
                            grad += res * U_r[i*m+j, r_idx] * W_r[k*n+i, r_idx]
                        V_r[j*p+k, r_idx] -= 0.001 * grad

                # For W
                for k in range(p):
                    for i in range(n):
                        grad = 0.0
                        for j in range(m):
                            res = 0.0
                            for r2 in range(r):
                                res += U_r[i*m+j, r2] * V_r[j*p+k, r2] * W_r[k*n+i, r2]
                            res -= 1.0
                            grad += res * U_r[i*m+j, r_idx] * V_r[j*p+k, r_idx]
                        W_r[k*n+i, r_idx] -= 0.001 * grad

        # Check final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U_r[i*m+j, r_idx] * V_r[j*p+k, r_idx] * W_r[k*n+i, r_idx]
                    err += (s - 1.0) ** 2

        if err < 1e-8:
            return U_r, V_r, W_r
        return None

    # Try various ranks
    for r in [35, 34, 33, 32, 31, 30, 29, 28]:
        if time.time() > deadline - 10:
            break

        result = try_rank(r, max_iters=200)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r, r):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break

    # If no lower rank found, return schoolbook
    if best_rank == schoolbook_rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Convert to exact fractions if possible (entries should be near integers/half-integers)
    def to_exact(mat):
        """Convert near-integer/half-integer values to exact fractions."""
        result = np.zeros_like(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                # Round to nearest 0.5
                rounded = np.round(v * 2) / 2
                if abs(v - rounded) < 1e-6:
                    result[i, j] = rounded
                else:
                    result[i, j] = v
        return result

    best_U = to_exact(best_U)
    best_V = to_exact(best_V)
    best_W = to_exact(best_W)

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
