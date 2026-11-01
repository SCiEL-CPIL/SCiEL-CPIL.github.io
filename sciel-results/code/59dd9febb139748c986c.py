"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]
_DEADLINE = 140.0
_START_TIME = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _objective(params, T_flat, n, m, p, r):
    """Sum of squared errors between the tensor reconstruction and target."""
    sz_u = n * m * r
    sz_v = m * p * r
    U = params[:sz_u].reshape(n * m, r)
    V = params[sz_u:sz_u + sz_v].reshape(m * p, r)
    W = params[sz_u + sz_v:].reshape(p * n, r)
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T_approx.reshape(-1) - T_flat
    return np.sum(diff * diff)


def _objective_grad(params, T, n, m, p, r):
    """Gradient of the objective with respect to all parameters."""
    sz_u = n * m * r
    sz_v = m * p * r
    U = params[:sz_u].reshape(n * m, r)
    V = params[sz_u:sz_u + sz_v].reshape(m * p, r)
    W = params[sz_u + sz_v:].reshape(p * n, r)
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    diff = T_approx - T
    gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])


def _try_rank(rank, T, T_flat, n, m, p, rng, max_restarts=20, time_budget=12.0):
    """
    Try to find a rank-`rank` decomposition using multiple random restarts
    with L-BFGS-B optimization. Returns (U, V, W) if found, else None.
    """
    t0 = time.time()
    for restart in range(max_restarts):
        if time.time() - t0 > time_budget:
            break
        if _remaining() < 5.0:
            break

        # Random initialization with varying scales
        scale = rng.uniform(0.3, 1.5)
        U = rng.standard_normal((n * m, rank)) * scale
        V = rng.standard_normal((m * p, rank)) * scale
        W = rng.standard_normal((p * n, rank)) * scale
        params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])

        try:
            res = minimize(
                _objective, params,
                args=(T_flat, n, m, p, rank),
                jac=lambda p: _objective_grad(p, T, n, m, p, rank),
                method='L-BFGS-B',
                options={'maxiter': 400, 'gtol': 1e-10, 'ftol': 1e-12, 'maxfun': 6000}
            )
            if res.fun < 1e-10:
                sz_u = n * m * rank
                sz_v = m * p * rank
                U_best = res.x[:sz_u].reshape(n * m, rank)
                V_best = res.x[sz_u:sz_u + sz_v].reshape(m * p, rank)
                W_best = res.x[sz_u + sz_v:].reshape(p * n, rank)
                return U_best, V_best, W_best
        except Exception:
            pass
    return None


def _polish(U, V, W, T, T_flat, n, m, p, rank):
    """
    High-precision polish pass: refine the decomposition with tighter tolerances.
    Also tries to snap to the nearest half-integer grid and verify.
    """
    # First, do a polish optimization
    params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    try:
        res = minimize(
            _objective, params,
            args=(T_flat, n, m, p, rank),
            jac=lambda p: _objective_grad(p, T, n, m, p, rank),
            method='L-BFGS-B',
            options={'maxiter': 800, 'gtol': 1e-14, 'ftol': 1e-16, 'maxfun': 12000}
        )
        if res.fun < 1e-12:
            sz_u = n * m * rank
            sz_v = m * p * rank
            U = res.x[:sz_u].reshape(n * m, rank)
            V = res.x[sz_u:sz_u + sz_v].reshape(m * p, rank)
            W = res.x[sz_u + sz_v:].reshape(p * n, rank)
    except Exception:
        pass

    # Try snapping to half-integers
    U_round = np.round(U * 2.0) / 2.0
    V_round = np.round(V * 2.0) / 2.0
    W_round = np.round(W * 2.0) / 2.0

    T_approx = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
    if np.max(np.abs(T_approx - T)) < 1e-9:
        return U_round, V_round, W_round

    # Try snapping to integers
    U_int = np.round(U).astype(float)
    V_int = np.round(V).astype(float)
    W_int = np.round(W).astype(float)
    T_approx2 = np.einsum('ar,br,cr->abc', U_int, V_int, W_int)
    if np.max(np.abs(T_approx2 - T)) < 1e-9:
        return U_int, V_int, W_int

    # Return the polished (non-snapped) values if they're close enough
    T_approx3 = np.einsum('ar,br,cr->abc', U, V, W)
    if np.max(np.abs(T_approx3 - T)) < 1e-9:
        return U, V, W

    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    T_flat = T.reshape(-1)
    rank_sb = n * m * p

    # Start with schoolbook as best
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_sb

    rng = np.random.default_rng(42)

    # Try to reduce rank from rank_sb-1 down to 12
    # For each rank, use time-budgeted search
    for rank in range(rank_sb - 1, 11, -1):
        if _remaining() < 18:
            break

        # Time budget per rank: distribute remaining time
        remaining_ranks = rank_sb - 1 - rank + 1
        time_per_rank = min(15.0, max(3.0, (_remaining() - 10.0) / max(1, remaining_ranks)))

        result = _try_rank(rank, T, T_flat, n, m, p, rng,
                           max_restarts=25, time_budget=time_per_rank)

        if result is not None:
            U_found, V_found, W_found = result
            # Polish
            polished = _polish(U_found, V_found, W_found, T, T_flat, n, m, p, rank)
            if polished is not None:
                best_U, best_V, best_W = polished
                best_rank = rank
                continue  # Found this rank, try lower
            else:
                # Even without polish, if the objective is tiny, accept it
                T_approx = np.einsum('ar,br,cr->abc', U_found, V_found, W_found)
                if np.max(np.abs(T_approx - T)) < 1e-8:
                    best_U, best_V, best_W = U_found, V_found, W_found
                    best_rank = rank
                    continue

    # Convert to list format
    if best_rank < rank_sb:
        U_list = [[float(round(x * 2) / 2) for x in row] for row in best_U]
        V_list = [[float(round(x * 2) / 2) for x in row] for row in best_V]
        W_list = [[float(round(x * 2) / 2) for x in row] for row in best_W]

        # Final verification
        U_v = np.array(U_list, dtype=float)
        V_v = np.array(V_list, dtype=float)
        W_v = np.array(W_list, dtype=float)
        T_approx = np.einsum('ar,br,cr->abc', U_v, V_v, W_v)
        if np.max(np.abs(T_approx - T)) < 1e-9:
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook decomposition
    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
