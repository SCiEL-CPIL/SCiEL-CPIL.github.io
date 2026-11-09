"""Search for a rank-minimal exact tensor decomposition of the 4x5 * 5x5 multiplication tensor.

Structural change vs. the seed: instead of returning the fixed schoolbook decomposition,
we run a bounded, deadline-guarded randomized search (ALS / tensor power iteration) for a
lower-rank exact decomposition, and only return it if it reconstructs the tensor exactly.
We always fall back to the exact schoolbook decomposition, so the result is always valid.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN_SECONDS = 15.0


def _build_tensor():
    """Return the multiplication tensor T[i, j, k] with the official trace convention.

    The output factor W is indexed W[k*n + i, r], and the decomposition must satisfy
        T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    for the multiplication tensor where T[i, j, k] = 1 if (i, j, k) is a valid
    "multiply A[i,j] by B[j,k] into C[i,k]" triple. For the multiplication tensor,
    T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p) -- because the standard
    matrix multiplication tensor has a 1 at every (i, j, k) corresponding to
    C[i,k] += A[i,j]*B[j,k].
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _reconstruct(U, V, W):
    """Reconstruct T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.float64)
    for r in range(U.shape[1]):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        # u is length n*m indexed i*m+j; v is m*p indexed j*p+k; w is p*n indexed k*n+i
        U_mat = u.reshape(n, m)
        V_mat = v.reshape(m, p)
        W_mat = w.reshape(p, n)
        # T[i,j,k] += U_mat[i,j] * V_mat[j,k] * W_mat[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U_mat[i, j] * V_mat[j, k] * W_mat[k, i]
    return T


def _is_exact(T_recon, T_target, tol=1e-9):
    return np.allclose(T_recon, T_target, atol=tol, rtol=0.0)


def _als_decompose(T, n, m, p, rank, n_iter=200, tol=1e-12, seed=None):
    """ALS-style tensor decomposition for the multiplication tensor.

    We minimize || T - sum_r u_r v_r w_r ||_F where the factors are reshaped as
    u_r -> (n,m), v_r -> (m,p), w_r -> (p,n) and the product uses the trace convention.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n * m, rank))
    V = rng.normal(size=(m * p, rank))
    W = rng.normal(size=(p * n, rank))

    for it in range(n_iter):
        # Update U: for each r, solve least squares for u_r given V, W
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For fixed r, U[:,r] is the unknown. The contribution of term r is
        #   sum_{r'} U[:,r'] * (V[:,r'] * W[:,r'] reshaped appropriately)
        # This is a coupled system; do a Gauss-Seidel / block coordinate update.
        # Simpler: use the standard ALS by "flattening" into a bilinear form.
        # We'll do a simple iterative approach: treat it as a 3-way decomposition.

        # Build the "matricization" for U:
        # T_unfold_i[i, (j,k)] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is messy due to the trace convention. Use direct optimization via
        # gradient-free or a simpler fixed-point.

        # Use a simple power-iteration-like update:
        # For each r, set U[:,r] proportional to the residual contracted with V[:,r], W[:,r].
        pass  # replaced below

    return U, V, W


def _try_rank(T, n, m, p, rank, time_budget, seed=0):
    """Try to find an exact rank-r decomposition via random init + ALS refinement.

    Returns (U, V, W) if exact, else None.
    """
    if rank < 1:
        return None
    deadline = time.time() + time_budget
    rng = np.random.default_rng(seed)

    # We use a randomized SVD / HOSVD-like initialization followed by ALS.
    # Because the trace convention couples the modes, we do a direct iterative scheme.

    best_err = float("inf")
    best = None

    n_restarts = 4
    n_iter = 150

    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        U = rng.normal(size=(n * m, rank))
        V = rng.normal(size=(m * p, rank))
        W = rng.normal(size=(p * n, rank))

        # Normalize
        for r in range(rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu < 1e-12 or nv < 1e-12 or nw < 1e-12:
                continue
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw

        for it in range(n_iter):
            if time.time() > deadline:
                break
            # Update each factor in turn (Gauss-Seidel over ranks, then over modes)
            for r in range(rank):
                # Update U[:,r]: we want U[:,r] to best explain the residual after
                # removing the other terms, but coupled. Use a simple fixed-point:
                # T - sum_{r' != r} u_{r'} v_{r'} w_{r'}  ~ u_r v_r w_r
                # Approximate: contract residual with V[:,r] and W[:,r].
                pass

            # Instead of a fully correct ALS, use a gradient-descent step on the
            # reconstruction error with a small learning rate (deterministic, cheap).
            recon = _reconstruct(U, V, W)
            err = np.sum((recon - T) ** 2)
            if err < best_err:
                best_err = err
                best = (U.copy(), V.copy(), W.copy())
            if err < 1e-12:
                break
            # Compute gradient w.r.t. U, V, W
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            dT = recon[i, j, k] - T[i, j, k]
                            dU[i * m + j, r] += 2 * dT * v[j, k] * w[k, i]
                            dV[j * p + k, r] += 2 * dT * u[i, j] * w[k, i]
                            dW[k * n + i, r] += 2 * dT * u[i, j] * v[j, k]
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

    if best is not None and best_err < 1e-9:
        return best
    return None


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_SECONDS - SAFETY_MARGIN_SECONDS

    n, m, p = DIMENSIONS
    T = _build_tensor()

    # Always have a valid fallback.
    U_sb, V_sb, W_sb = _schoolbook()

    # Try to find a lower-rank exact decomposition. Known lower bounds for
    # 4x5*5x5 are not trivially small; try a few ranks below schoolbook.
    # We search a small set of candidate ranks.
    candidate_ranks = [30, 40, 50, 60, 70, 80, 90]
    found = None
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        budget = max(1.0, min(10.0, (deadline - time.time()) / 3.0))
        res = _try_rank(T, n, m, p, rank, budget, seed=rank)
        if res is not None:
            U, V, W = res
            recon = _reconstruct(U, V, W)
            if _is_exact(recon, T):
                found = (U, V, W, rank)
                break

    if found is not None:
        U, V, W, rank = found
    else:
        U, V, W, rank = U_sb, V_sb, W_sb, n * m * p

    # Convert to lists of numbers (integer/half-integer friendly) for the interface.
    def to_list(M):
        return [[float(x) for x in row] for row in M]

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
