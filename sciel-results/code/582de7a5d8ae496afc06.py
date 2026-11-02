import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    positions = []  # list of (a, b, c) where target[a,b,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0
                positions.append((a, b, c))

    num_pos = len(positions)  # 24

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose target."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def check_half_integer(X, atol=1e-7):
        """Check if all entries are close to half-integers."""
        return np.allclose(X * 2, np.round(X * 2), atol=atol)

    # Pre-compute all feasible rank-1 terms with small half-integer entries
    # Each term is (u, v, w) where u in R^nm, v in R^mp, w in R^pn
    # u[a]*v[b]*w[c] contributes to target[a,b,c]
    # We want terms where u, v, w have entries in {-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}
    # and the product u[a]*v[b]*w[c] is non-zero only at target positions

    # Strategy: enumerate sparse rank-1 terms with small entries
    # A rank-1 term (u,v,w) contributes to position (a,b,c) with value u[a]*v[b]*w[c]
    # For the term to be "valid", it should only contribute at target positions
    # OR we allow contributions anywhere but the sum must equal target

    # Let's use a different approach: greedy set cover
    # Each candidate term is a vector in R^(nm*mp*pn) representing its contribution
    # We want a subset of terms whose sum equals the target tensor

    # Generate candidate terms with bounded half-integer entries
    # To keep it tractable, use entries from a small set
    # Focus on terms that are "aligned" with the structure

    # Actually, let's use the ALS approach but with a smarter initialization
    # and a final exact rounding step

    def try_rank_als(r, num_restarts=80, max_iters=40):
        """
        Try to find a rank r decomposition using Alternating Least Squares (ALS).
        """
        if time.time() > deadline:
            return None

        best_U, best_V, best_W = None, None, None
        best_err = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            # Initialization strategies
            if restart % 4 == 0:
                # Random initialization
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif restart % 4 == 1:
                # Schoolbook-like initialization with perturbation
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
                    col = idx % r
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                U += np.random.randn(nm, r) * 0.05
                V += np.random.randn(mp, r) * 0.05
                W += np.random.randn(pn, r) * 0.05
            elif restart % 4 == 2:
                # Sparse random
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
                # Zero out some entries to encourage sparsity
                for _ in range(r // 2):
                    U[np.random.randint(0, nm), :] = 0
                    V[np.random.randint(0, mp), :] = 0
                    W[np.random.randint(0, pn), :] = 0
            else:
                # Structured: group by i, j, k
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                # Assign columns to (i,j,k) groups
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < r:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                                col += 1
                # Add random perturbations
                U += np.random.randn(nm, r) * 0.1
                V += np.random.randn(mp, r) * 0.1
                W += np.random.randn(pn, r) * 0.1

            # ALS Optimization
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Update U
                A_U = np.zeros((mp * pn, r))
                for k in range(r):
                    A_U[:, k] = np.kron(W[:, k], V[:, k])

                T_flat = target.reshape(nm, mp * pn)
                for a in range(nm):
                    try:
                        u_a, _, _, _ = np.linalg.lstsq(A_U, T_flat[a], rcond=None)
                        U[a, :] = u_a
                    except Exception:
                        pass

                # Update V
                A_V = np.zeros((nm * pn, r))
                for k in range(r):
                    A_V[:, k] = np.kron(W[:, k], U[:, k])

                T_reshaped_V = np.zeros((mp, nm * pn))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_V[b, a * pn + c] = target[a, b, c]

                for b in range(mp):
                    try:
                        v_b, _, _, _ = np.linalg.lstsq(A_V, T_reshaped_V[b], rcond=None)
                        V[b, :] = v_b
                    except Exception:
                        pass

                # Update W
                A_W = np.zeros((nm * mp, r))
                for k in range(r):
                    A_W[:, k] = np.kron(V[:, k], U[:, k])

                T_reshaped_W = np.zeros((pn, nm * mp))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_W[c, a * mp + b] = target[a, b, c]

                for c in range(pn):
                    try:
                        w_c, _, _, _ = np.linalg.lstsq(A_W, T_reshaped_W[c], rcond=None)
                        W[c, :] = w_c
                    except Exception:
                        pass

            # Check convergence and rounding
            if time.time() > deadline - 5:
                break

            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.sum((T_approx - target)**2)

            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < 1e-8:
                    # Try rounding to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2

                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    else:
                        # Check if unrounded is valid (floating point)
                        if check_decomposition(U, V, W):
                            return U, V, W

        # If best_err is very low, try one final polish
        if best_err < 1e-5 and best_U is not None:
            from scipy.optimize import least_squares
            params = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])

            def residuals(params):
                U_p = params[:r * nm].reshape(nm, r)
                V_p = params[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = params[r * (nm + mp):].reshape(pn, r)
                T_approx = np.einsum('ak,bk,ck->abc', U_p, V_p, W_p)
                return (T_approx - target).flatten()

            try:
                res = least_squares(residuals, params, method='lm', max_nfev=1000)
                U_p = res.x[:r * nm].reshape(nm, r)
                V_p = res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = res.x[r * (nm + mp):].reshape(pn, r)

                U_r = np.round(U_p * 2) / 2
                V_r = np.round(V_p * 2) / 2
                W_r = np.round(W_p * 2) / 2

                if check_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                elif check_decomposition(U_p, V_p, W_p):
                    return U_p, V_p, W_p
            except Exception:
                pass

        return None

    # Try ranks from 16 to 23 (below schoolbook rank 24)
    for r in range(16, 24):
        if time.time() > deadline - 15:
            break

        res = try_rank_als(r, num_restarts=60, max_iters=35)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
