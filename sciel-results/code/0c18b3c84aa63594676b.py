import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS


def _verify(U, V, W, rank, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _residuals(U, V, W, rank):
    """Compute the 12 tensor-entry residuals (should all be 0 for a valid decomposition)."""
    res = np.empty(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                res[idx] = s - 1.0
                idx += 1
    return res


def _als_step_U(V, W, rank, n, m, p):
    """Solve for U given V, W: for each (i,j), U[i,j,:] = (V[j,:] o W[:,i]).T @ rhs."""
    U = np.zeros((n * m, rank))
    for i in range(n):
        for j in range(m):
            # rhs[k] = 1 for all k
            # U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r,k = 1
            # => sum_r U[i,j,r] * (sum_k V[j,k,r] * W[k,i,r]) = 1
            # This is a linear system in U[i,j,:]: M @ u = 1, where M[r] = sum_k V[j,k,r]*W[k,i,r]
            M = np.array([np.sum(V[j, :] * W[:, i]) for _ in range(1)])  # placeholder
            # Actually M is a scalar per r:
            M = np.sum(V[j, :] * W[:, i], axis=1)  # shape (rank,)
            # We want sum_r M[r] * U[i,j,r] = 1. Underdetermined; use least-norm:
            # u = M / (M.M)
            mm = np.dot(M, M)
            if mm > 1e-12:
                U[i * m + j, :] = M / mm
            else:
                U[i * m + j, :] = 0
    return U


def _als_step_V(U, W, rank, n, m, p):
    """Solve for V given U, W: for each (j,k), V[j,k,:] = (U[:,j] o W[k,:]).T @ rhs."""
    V = np.zeros((m * p, rank))
    for j in range(m):
        for k in range(p):
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r,i = 1
            # => sum_r V[j,k,r] * (sum_i U[i,j,r]*W[k,i,r]) = 1
            M = np.sum(U[:, j] * W[k, :], axis=1)  # shape (rank,)
            mm = np.dot(M, M)
            if mm > 1e-12:
                V[j * p + k, :] = M / mm
            else:
                V[j * p + k, :] = 0
    return V


def _als_step_W(U, V, rank, n, m, p):
    """Solve for W given U, V: for each (k,i), W[k,i,:] = (U[i,:] o V[:,k]).T @ rhs."""
    W = np.zeros((p * n, rank))
    for k in range(p):
        for i in range(n):
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] summed over r,j = 1
            # => sum_r W[k,i,r] * (sum_j U[i,j,r]*V[j,k,r]) = 1
            M = np.sum(U[i, :] * V[:, k], axis=1)  # shape (rank,)
            mm = np.dot(M, M)
            if mm > 1e-12:
                W[k * n + i, :] = M / mm
            else:
                W[k * n + i, :] = 0
    return W


def _rank11_decomposition():
    """
    Rank-11 decomposition: Strassen (rank 7) for 2x2 block + schoolbook (rank 4) for col 2.
    """
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val):
        U[i * m + j][r] = val

    def set_v(j, k, r, val):
        V[j * p + k][r] = val

    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)
    set_w(1, 1, 0, 1)

    # Term 1: (A10+A11)B00 -> C01+=1, C11+=-1
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    set_w(1, 1, 1, -1)

    # Term 2: A00(B01-B11) -> C01+=1, C11+=1
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    set_w(1, 1, 2, 1)

    # Term 3: A11(B10-B00) -> C00+=1, C01+=1
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1)
    set_w(0, 1, 3, 1)

    # Term 4: (A00+A01)B11 -> C00+=-1, C01+=1
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1)
    set_w(1, 0, 4, 1)

    # Term 5: (A10-A00)(B00+B01) -> C11+=1
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 6: (A01-A11)(B10+B11) -> C00+=1
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 7: A00*B02 -> C02+=1
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 8: A01*B12 -> C02+=1
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 9: A10*B02 -> C12+=1
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 10: A11*B12 -> C12+=1
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _try_rank10_search(deadline):
    """
    Attempt to find a rank-10 decomposition via numerical search.
    Returns a dict with integer/half-integer entries if found and verified, else None.
    """
    if time.time() > deadline - 20:
        return None

    rank = 10
    base = _rank11_decomposition()
    U11 = np.array(base["U"], dtype=float)  # 4 x 11
    V11 = np.array(base["V"], dtype=float)  # 6 x 11
    W11 = np.array(base["W"], dtype=float)  # 6 x 11

    rng = np.random.default_rng(12345)

    best = None
    best_cost = float('inf')

    def polish(U0, V0, W0):
        """ALS + least_squares polishing; return (cost, U, V, W)."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        for _ in range(30):
            if time.time() > deadline - 15:
                break
            U = _als_step_U(V, W, rank, n, m, p)
            V = _als_step_V(U, W, rank, n, m, p)
            W = _als_step_W(U, V, rank, n, m, p)

        # Final least_squares polish
        try:
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def residuals(params):
                Uf = params[:4 * rank].reshape(n * m, rank)
                Vf = params[4 * rank:4 * rank + m * p * rank].reshape(m * p, rank)
                Wf = params[4 * rank + m * p * rank:].reshape(p * n, rank)
                return _residuals(Uf, Vf, Wf, rank)

            sol = least_squares(residuals, x0, method='lm', max_nfev=200)
            U = sol.x[:4 * rank].reshape(n * m, rank)
            V = sol.x[4 * rank:4 * rank + m * p * rank].reshape(m * p, rank)
            W = sol.x[4 * rank + m * p * rank:].reshape(p * n, rank)
        except Exception:
            pass

        cost = float(np.sum(_residuals(U, V, W, rank) ** 2))
        return cost, U, V, W

    # Strategy 1: drop one term from rank-11 and polish
    for drop_term in range(11):
        if time.time() > deadline - 15:
            break
        keep = [r for r in range(11) if r != drop_term]
        U0 = U11[:, keep]
        V0 = V11[:, keep]
        W0 = W11[:, keep]
        cost, U, V, W = polish(U0, V0, W0)
        if cost < best_cost:
            best_cost = cost
            best = (U, V, W)
        if best_cost < 1e-12:
            break

    # Strategy 2: random structured starts
    for trial in range(8):
        if time.time() > deadline - 15:
            break
        U0 = rng.normal(scale=0.5, size=(n * m, rank))
        V0 = rng.normal(scale=0.5, size=(m * p, rank))
        W0 = rng.normal(scale=0.5, size=(p * n, rank))
        cost, U, V, W = polish(U0, V0, W0)
        if cost < best_cost:
            best_cost = cost
            best = (U, V, W)
        if best_cost < 1e-12:
            break

    if best is None:
        return None
    if best_cost > 1e-8:
        return None

    U, V, W = best
    # Round to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    # Re-verify after rounding
    if not _verify(U.tolist(), V.tolist(), W.tolist(), rank, tol=1e-9):
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start = time.time()
    deadline = start + 130  # safety margin before 150s wall-clock

    # Try to find a rank-10 decomposition
    result10 = None
    try:
        result10 = _try_rank10_search(deadline)
    except Exception:
        result10 = None

    if result10 is not None:
        return result10

    # Fall back to the guaranteed rank-11 decomposition
    return _rank11_decomposition()
