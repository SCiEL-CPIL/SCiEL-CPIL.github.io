"""Strassen-like recursive partitioning with coefficient local search."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def time_left():
        return DEADLINE - (time.time() - t0)

    # ---- schoolbook (rank 63) ----
    r0 = n * m * p  # 63
    U0 = np.zeros((n * m, r0))
    V0 = np.zeros((m * p, r0))
    W0 = np.zeros((p * n, r0))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, c] = 1.0
                V0[j * p + k, c] = 1.0
                W0[k * n + i, c] = 1.0
                c += 1

    def verify_matmul_fast(U, V, W, tol=1e-8):
        """Fast vectorized verification."""
        r = U.shape[1]
        U_r = U.reshape(n, m, r)
        V_r = V.reshape(m, p, r)
        W_r = W.reshape(p, n, r)
        # C[i, k, j1, j2] = sum_r W[k, i, r] * U[i, j1, r] * V[j2, k, r]
        try:
            C = np.einsum('kir,ijr,jkr->ikij', W_r, U_r, V_r, optimize=True)
        except Exception:
            return False
        I_m = np.eye(m)
        for i in range(n):
            for k in range(p):
                if not np.allclose(C[i, k, :, :], I_m, atol=tol):
                    return False
        return True

    if not verify_matmul_fast(U0, V0, W0):
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_rank = r0
    best_solution = (U0.copy(), V0.copy(), W0.copy())

    # Precompute constraint indices
    constraints = []
    for i in range(n):
        for k in range(p):
            for j1 in range(m):
                for j2 in range(m):
                    constraints.append((i, k, j1, j2, 1.0 if j1 == j2 else 0.0))

    def compute_residuals(U, V, W):
        """Compute all constraint residuals as a vector."""
        r = U.shape[1]
        residuals = np.zeros(len(constraints))
        idx = 0
        for (i, k, j1, j2, expected) in constraints:
            s = 0.0
            for c in range(r):
                s += W[k*n+i, c] * U[i*m+j1, c] * V[j2*p+k, c]
            residuals[idx] = s - expected
            idx += 1
        return residuals

    def sum_of_squares(U, V, W):
        """Smooth surrogate: sum of squared residuals."""
        res = compute_residuals(U, V, W)
        return np.dot(res, res)

    random.seed(42)
    np.random.seed(42)

    # ---- Strategy: Try to reduce rank by finding linear dependencies ----
    # For matrix multiplication, the key structure is that each (i,k) pair
    # has m^2 = 9 constraints. The schoolbook uses one term per (i,j,k).
    # We can try to combine terms that share the same (i,k) pair.
    
    # For a fixed (i,k), we need: sum_c W[i,k,c] * U[i,*,c] * V[*,k,c] = I_m
    # This is a rank-m decomposition of the identity using outer products.
    # The minimum rank for this sub-problem is m (using U[i,j,c]=delta_{j,c}, V[j,k,c]=delta_{j,c})
    # But we can potentially share structure across different (i,k) pairs.

    # Try structured search: for each (i,k) pair, the contribution must form I_m
    # Total rank = sum over (i,k) of rank needed for that pair
    # If all pairs are independent, rank = n*p*m = 63 (schoolbook)
    # To reduce rank, we need to share terms across different (i,k) pairs.

    # ---- Direct search for rank < 63 using structured initialization ----
    # Key insight: if two terms share the same i and k but different j,
    # they contribute to the same (i,k) block. We can try to merge them.

    # Try all ranks from 62 down to 50
    for target_rank in range(r0 - 1, 49, -1):
        if time_left() < 5:
            break
        if target_rank >= best_rank:
            continue

        max_attempts = 500
        found = False
        for attempt in range(max_attempts):
            if time_left() < 2:
                break

            # Build solution by starting from schoolbook and trying to merge terms
            # Select which terms to keep and which to merge
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))

            # Keep first (target_rank) schoolbook terms as base
            for c in range(target_rank):
                i = c // (m*p)
                rem = c % (m*p)
                j = rem // p
                k = rem % p
                U_r[i*m+j, c] = 1.0
                V_r[j*p+k, c] = 1.0
                W_r[k*n+i, c] = 1.0

            # Now try to adjust the last few terms to absorb the removed terms
            # The removed terms are indices target_rank to r0-1
            # We need to find coefficients such that the full sum still works

            # Use local search: perturb the last few columns
            n_perturb = min(5, target_rank)
            for c in range(target_rank - n_perturb, target_rank):
                for idx in range(n*m):
                    U_r[idx, c] += random.gauss(0, 0.3)
                for idx in range(m*p):
                    V_r[idx, c] += random.gauss(0, 0.3)
                for idx in range(p*n):
                    W_r[idx, c] += random.gauss(0, 0.3)

            # Check validity
            if verify_matmul_fast(U_r, V_r, W_r):
                best_rank = target_rank
                best_solution = (U_r.copy(), V_r.copy(), W_r.copy())
                found = True
                break

        if found:
            break

    # ---- Return the best solution found ----
    U_best, V_best, W_best = best_solution

    def to_list(arr):
        result = []
        for row in arr:
            new_row = []
            for x in row:
                hx = round(x * 2) / 2.0
                if abs(x - hx) < 1e-9:
                    new_row.append(hx)
                else:
                    new_row.append(float(x))
            result.append(new_row)
        return result

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}

# EVOLVE-BLOCK-END
