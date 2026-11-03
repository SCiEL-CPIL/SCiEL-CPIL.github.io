"""Matrix multiplication 3x4x5 rank-minimal exact tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]


def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _validate(U, V, W, T):
    n, m, p = T.shape
    rank = U.shape[1]
    recon = np.zeros((n, m, p))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                uj = u[i, j]
                if uj == 0:
                    continue
                for k in range(p):
                    recon[i, j, k] += uj * v[j, k] * w[k, i]
    return np.allclose(recon, T, atol=1e-9)


def _is_half_integer(arr, max_val=1000):
    if not np.all(np.isfinite(arr)):
        return False
    doubled = arr * 2
    if not np.all(np.isclose(doubled, np.round(doubled), atol=1e-6)):
        return False
    if not np.all(np.abs(arr) <= max_val):
        return False
    return True


def _schoolbook():
    n, m, p = DIMENSIONS
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
    return U, V, W, rank


def _try_remove_term(U, V, W, T, r_remove):
    """Try to remove column r_remove by absorbing its contribution into remaining columns via U adjustment."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    mask = [i for i in range(R) if i != r_remove]
    U_rem = U[:, mask]
    V_rem = V[:, mask]
    W_rem = W[:, mask]
    
    # Residual from removed term
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            uj = U[i * m + j, r_remove]
            if uj != 0:
                for k in range(p):
                    residual[i, j, k] = uj * V[j * p + k, r_remove] * W[k * n + i, r_remove]
    
    # Solve for delta_U
    delta_U = np.zeros((n * m, len(mask)))
    solvable = True
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, len(mask)))
            for k in range(p):
                M[k, :] = V_rem[j * p + k, :] * W_rem[k * n + i, :]
            if np.linalg.matrix_rank(M) < len(mask):
                # Underdetermined - try least squares
                pass
            try:
                sol, res, rank_s, sv = np.linalg.lstsq(M.T, residual[i, j, :], rcond=1e-10)
                delta_U[i * m + j, :] = sol
            except:
                solvable = False
                break
        if not solvable:
            break
    
    if not solvable:
        return None
    
    U_new = U_rem + delta_U
    if not _is_half_integer(U_new, max_val=1000):
        return None
    
    # Verify
    if _validate(U_new, V_rem, W_rem, T):
        return U_new, V_rem, W_rem
    return None


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor()
    
    U, V, W, rank = _schoolbook()
    best_U, best_V, best_W, best_rank = U.copy(), V.copy(), W.copy(), rank
    
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try removing terms one at a time from schoolbook
    changed = True
    while changed and time.time() < deadline:
        changed = False
        R = best_U.shape[1]
        for r_remove in range(R - 1, -1, -1):
            if time.time() > deadline:
                break
            result = _try_remove_term(best_U, best_V, best_W, T, r_remove)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = best_U.shape[1]
                changed = True
                break
    
    U_list = [[float(best_U[i, r]) for r in range(best_rank)] for i in range(n * m)]
    V_list = [[float(best_V[i, r]) for r in range(best_rank)] for i in range(m * p)]
    W_list = [[float(best_W[i, r]) for r in range(best_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
