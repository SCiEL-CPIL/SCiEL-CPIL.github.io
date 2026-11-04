"""Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _round_half(A):
    """Round each entry to the nearest half-integer (multiple of 0.5)."""
    return np.round(A * 2.0) / 2.0


def _verify(U, V, W, T):
    """Check exact decomposition: U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] sums to T[i,j,k]."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    # Reshape for einsum: U -> (n,m,r), V -> (m,p,r), W -> (p,n,r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    # Sum over r: Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(result, T, atol=1e-9, rtol=0.0)


def _objective(params, U, V, W, T, n, m, p):
    """Smooth epigraph objective: t + penalty for constraint violation.
    params = [U_flat, V_flat, W_flat, t] concatenated."""
    r = U.shape[1]
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r

    Uf = params[:size_U].reshape(n, m, r)
    Vf = params[size_U:size_U + size_V].reshape(m, p, r)
    Wf = params[size_U + size_V:size_U + size_V + size_W].reshape(p, n, r)
    t = params[-1]

    # Compute decomposition result
    result = np.einsum('ijc,jkc,kic->ijk', Uf, Vf, Wf)
    diff = result - T
    norm = np.sqrt(np.sum(diff * diff))

    # Epigraph: minimize t subject to norm <= t
    # Use smooth penalty: t + max(0, norm - t)^2
    excess = norm - t
    penalty = excess * excess if excess > 0 else 0.0
    return t + penalty


def _grad_objective(params, U, V, W, T, n, m, p):
    """Gradient of the smooth epigraph objective."""
    r = U.shape[1]
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r

    Uf = params[:size_U].reshape(n, m, r)
    Vf = params[size_U:size_U + size_V].reshape(m, p, r)
    Wf = params[size_U + size_V:size_U + size_V + size_W].reshape(p, n, r)
    t = params[-1]

    result = np.einsum('ijc,jkc,kic->ijk', Uf, Vf, Wf)
    diff = result - T
    norm = np.sqrt(np.sum(diff * diff))

    excess = norm - t
    # d(t + max(0,norm-t)^2)/dt = 1 - 2*max(0,norm-t)
    dt = 1.0 - 2.0 * max(0.0, excess)

    # d/dU[i,j,c] of norm: (1/norm) * 2 * diff[i,j] * Vf[j,k,c] * Wf[k,i,c] summed over k
    # Actually: result[i,j,k] = sum_c Uf[i,j,c]*Vf[j,k,c]*Wf[k,i,c]
    # d(norm)/d(Uf[i,j,c]) = (1/norm) * sum_k diff[i,j,k] * Vf[j,k,c] * Wf[k,i,c]
    # d(norm)/d(Vf[j,k,c]) = (1/norm) * sum_i diff[i,j,k] * Uf[i,j,c] * Wf[k,i,c]
    # d(norm)/d(Wf[k,i,c]) = (1/norm) * sum_j diff[i,j,k] * Uf[i,j,c] * Vf[j,k,c]

    if norm > 1e-12:
        inv_norm = 1.0 / norm
        # Gradient wrt Uf
        # dU[i,j,c] = inv_norm * sum_k diff[i,j,k] * Vf[j,k,c] * Wf[k,i,c]
        dU = np.einsum('ijk,jkc,kic->ijc', diff, Vf, Wf) * inv_norm
        dV = np.einsum('ijk,ijc,kic->jkc', diff, Uf, Wf) * inv_norm
        dW = np.einsum('ijk,ijc,jkc->kic', diff, Uf, Vf) * inv_norm

        if excess > 0:
            # d(penalty)/dU = 2*excess * d(norm)/dU
            dU *= 2.0 * excess
            dV *= 2.0 * excess
            dW *= 2.0 * excess
    else:
        dU = np.zeros_like(Uf)
        dV = np.zeros_like(Vf)
        dW = np.zeros_like(Wf)

    grad = np.concatenate([
        dU.ravel(), dV.ravel(), dW.ravel(), [dt]
    ])
    return grad


def _try_rank(r, T, n, m, p, deadline, max_iters=2000):
    """Try to find a rank-r decomposition. Returns (U,V,W,success)."""
    if time.time() > deadline:
        return None, False

    # Initialize with schoolbook-like random init
    np.random.seed(42)
    U = np.random.randn(n, m, r) * 0.1
    V = np.random.randn(m, p, r) * 0.1
    W = np.random.randn(p, n, r) * 0.1

    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r

    # Initial t: upper bound on norm
    result0 = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    t0 = np.sqrt(np.sum((result0 - T) ** 2)) + 1.0

    params = np.concatenate([U.ravel(), V.ravel(), W.ravel(), [t0]])

    from scipy.optimize import minimize

    best_params = params.copy()
    best_val = _objective(params, U, V, W, T, n, m, p)

    # L-BFGS-B optimization
    try:
        res = minimize(
            _objective,
            params,
            jac=_grad_objective,
            args=(U, V, W, T, n, m, p),
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass

    if time.time() > deadline:
        return None, False

    # Extract best
    Uf = best_params[:size_U].reshape(n, m, r)
    Vf = best_params[size_U:size_U + size_V].reshape(m, p, r)
    Wf = best_params[size_U + size_V:size_U + size_V + size_W].reshape(p, n, r)

    # Round to half-integers
    U_r = _round_half(Uf)
    V_r = _round_half(Vf)
    W_r = _round_half(Wf)

    # Verify
    if _verify(U_r, V_r, W_r, T):
        # Try to remove zero columns
        U_mat = U_r.reshape(n * m, r)
        V_mat = V_r.reshape(m * p, r)
        W_mat = W_r.reshape(p * n, r)

        # Find non-zero columns
        col_norms = np.sqrt(np.sum(U_mat ** 2, axis=0) + np.sum(V_mat ** 2, axis=0) + np.sum(W_mat ** 2, axis=0))
        keep = col_norms > 1e-9
        if not np.all(keep):
            U_mat = U_mat[:, keep]
            V_mat = V_mat[:, keep]
            W_mat = W_mat[:, keep]
            if _verify(U_mat, V_mat, W_mat, T):
                return U_mat, V_mat, W_mat, True

        return U_mat, V_mat, W_mat, True

    return None, False


def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    T = _build_tensor()

    # Start with schoolbook (rank 96)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # Try decreasing ranks
    for r in range(rank - 1, max(30, rank - 30), -1):
        if time.time() > deadline:
            break
        U_try, V_try, W_try, success = _try_rank(r, T, n, m, p, deadline, max_iters=1000)
        if success:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = r
            break

    # Convert to lists for return
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
