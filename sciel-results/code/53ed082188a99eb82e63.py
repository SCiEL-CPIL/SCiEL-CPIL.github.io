"""Find a rank-minimal exact tensor decomposition for multiplying a 3x4 by 4x5 matrix."""
import time
import itertools
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_residuals = n * m * p

    # Schoolbook rank as fallback
    def schoolbook():
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

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p

    # Try ranks from a lower bound upward
    min_rank = 20
    max_rank = 59  # We stop before 60 because 60 is schoolbook

    np.random.seed(42)

    def residual_func(params, rank):
        # params is a flat vector of length rank * (nm + mp + pn)
        U = params[:rank * nm].reshape(nm, rank)
        V = params[rank * nm:rank * (nm + mp)].reshape(mp, rank)
        W = params[rank * (nm + mp):].reshape(pn, rank)

        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)

        T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)

        # Target is 1 everywhere
        R = T_approx - 1.0
        return R.ravel()

    # Strategy: exploit linear structure by treating middle index as vector space
    # Use change of basis to diagonalize interaction
    # The key insight: for 3x4 by 4x5, we can potentially reduce rank below schoolbook
    # by finding linear combinations that exploit the structure

    for rank in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break

        found = False

        # Budget per rank
        time_per_rank = 4.0
        start_rank_time = time.time()

        # Number of restarts depends on available time
        max_restarts = 8

        for restart in range(max_restarts):
            if time.time() > start_rank_time + time_per_rank:
                break
            if time.time() > deadline:
                break

            # Random initialization with structure-aware scaling
            # If U, V, W are O(1), T is O(rank). We want T=1.
            # Scale by 1/sqrt(rank) for balanced contribution
            scale = 1.0 / np.sqrt(rank)

            # Initialize with some structure: group by middle index
            U0 = np.zeros((n * m, rank))
            V0 = np.zeros((m * p, rank))
            W0 = np.zeros((p * n, rank))

            # For each rank component, assign to specific (i,j,k) triples
            # with some random mixing to allow optimization to find structure
            for r in range(rank):
                # Randomly select which (i,j,k) this rank component will primarily contribute to
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)

                # Assign primary contribution
                U0[i * m + j, r] = 1.0 / np.sqrt(3)
                V0[j * p + k, r] = 1.0 / np.sqrt(3)
                W0[k * n + i, r] = 1.0 / np.sqrt(3)

                # Add small random perturbation to all entries to break symmetry
                U0[:, r] += np.random.randn(n * m) * scale * 0.1
                V0[:, r] += np.random.randn(m * p) * scale * 0.1
                W0[:, r] += np.random.randn(p * n) * scale * 0.1

            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])

            try:
                n_params = rank * (nm + mp + pn)
                method = 'lm' if n_params >= total_residuals else 'trf'

                # Use tight tolerances for exactness
                res = least_squares(
                    residual_func, x0, args=(rank,),
                    method=method, max_nfev=500,
                    ftol=1e-14, xtol=1e-14, gtol=1e-14
                )
            except Exception:
                continue

            if res.cost < 1e-10:
                # Verify
                U_sol = res.x[:rank * nm].reshape(nm, rank)
                V_sol = res.x[rank * nm:rank * (nm + mp)].reshape(mp, rank)
                W_sol = res.x[rank * (nm + mp):].reshape(pn, rank)

                U_r = U_sol.reshape(n, m, rank)
                V_r = V_sol.reshape(m, p, rank)
                W_r = W_sol.reshape(p, n, rank)
                T_approx = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)

                if np.max(np.abs(T_approx - 1.0)) < 1e-8:
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = rank
                    found = True
                    break

        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
