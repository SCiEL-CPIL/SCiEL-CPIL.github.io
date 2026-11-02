"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _build_strassen_2x2x2():
    """Strassen rank-7 decomposition for 2x2 x 2x2.
    Returns U (4x7), V (4x7), W (4x7) with W indexed by [k*2+i, r]."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Term 1: M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # Term 2: M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # Term 3: M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # Term 4: M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # Term 5: M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # Term 6: M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # Term 7: M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _build_rank53():
    """Build the guaranteed rank-53 decomposition using 7 Strassen blocks + 1 schoolbook."""
    n, m, p = DIMENSIONS
    U_str, V_str, W_str = _build_strassen_2x2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)

    total_rank = 7 * 7 + 4  # 53

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col_offset = 0
    for b in range(7):
        start_k = 2 * b
        U[:, col_offset:col_offset+7] = U_str
        for j in range(m):
            for k_local in range(2):
                k_global = start_k + k_local
                V[j * p + k_global, col_offset:col_offset+7] = V_str[j * 2 + k_local, :]
        for k_local in range(2):
            k_global = start_k + k_local
            for i in range(n):
                W[k_global * n + i, col_offset:col_offset+7] = W_str[k_local * 2 + i, :]
        col_offset += 7

    # Last column: schoolbook for k=14
    U[:, col_offset:col_offset+4] = U_sb
    for j in range(m):
        V[j * p + 14, col_offset:col_offset+4] = V_sb[j, :]
    for i in range(n):
        W[14 * n + i, col_offset:col_offset+4] = W_sb[i, :]

    return U, V, W

def _residuals_full(U, V, W, n=2, m=2, p=15):
    """Compute residuals for the full (2,2,15) tensor."""
    res = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]) - 1.0
                idx += 1
    return res

def _try_rank52(U53, V53, W53, deadline):
    """
    Try to find a rank-52 decomposition by optimizing from rank-53 warm start.
    Uses a smooth objective and multiple restarts with different strategies.
    """
    n, m, p = DIMENSIONS
    rank = 52
    n_params = n*m*rank + m*p*rank + p*n*rank  # 4*52 + 30*52 + 30*52 = 3808

    def objective(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        res = _residuals_full(U, V, W)
        return np.sum(res**2)

    def jac(params):
        """Jacobian via finite differences (expensive but reliable)."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        res = _residuals_full(U, V, W)
        # Use gradient-free method instead
        return None

    # Strategy 1: Zero out one column of rank-53 and optimize
    best_obj = np.inf
    best_params = None

    # Try zeroing each of the last 4 columns (the schoolbook part)
    for zero_col in [49, 50, 51, 52]:
        if time.time() > deadline - 20:
            break
        x0 = np.concatenate([
            U53[:, :rank].flatten(),
            V53[:, :rank].flatten(),
            W53[:, :rank].flatten()
        ])
        # Zero out the column we want to eliminate
        # The column index in the flattened params
        U_flat = x0[:n*m*rank]
        V_flat = x0[n*m*rank : n*m*rank + m*p*rank]
        W_flat = x0[n*m*rank + m*p*rank :]

        # We're keeping first 52 columns, so column 52 of the original is dropped
        # But we want to see if the remaining 52 can be adjusted
        # Actually, let's just try the warm start as-is
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-12:
                break
        except Exception:
            continue

    # Strategy 2: Random perturbations
    for seed in range(5):
        if time.time() > deadline - 15:
            break
        np.random.seed(seed)
        x0 = np.concatenate([
            U53[:, :rank].flatten(),
            V53[:, :rank].flatten(),
            W53[:, :rank].flatten()
        ])
        x0 += np.random.randn(n_params) * 0.05
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-12:
                break
        except Exception:
            continue

    if best_params is None or best_obj > 1e-8:
        return None, None, None

    U = best_params[:n*m*rank].reshape(n*m, rank)
    V = best_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = best_params[n*m*rank + m*p*rank :].reshape(p*n, rank)

    # Snap to half-integers and re-verify
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    res_snap = _residuals_full(U_snap, V_snap, W_snap)
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap

    # Check if raw solution is already close to half-integer
    if np.allclose(U*2, np.round(U*2), atol=1e-5) and \
       np.allclose(V*2, np.round(V*2), atol=1e-5) and \
       np.allclose(W*2, np.round(W*2), atol=1e-5):
        return U_snap, V_snap, W_snap

    # Try integer snap
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    res_int = _residuals_full(U_int, V_int, W_int)
    if np.max(np.abs(res_int)) < 1e-6:
        return U_int, V_int, W_int

    return None, None, None

def _try_rank51(U52, V52, W52, deadline):
    """
    Try to find a rank-51 decomposition by optimizing from rank-52 warm start.
    """
    n, m, p = DIMENSIONS
    rank = 51
    n_params = n*m*rank + m*p*rank + p*n*rank  # 4*51 + 30*51 + 30*51 = 3708

    def objective(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        res = _residuals_full(U, V, W)
        return np.sum(res**2)

    best_obj = np.inf
    best_params = None

    # Warm start: use first 51 columns of rank-52 solution
    x0 = np.concatenate([
        U52[:, :rank].flatten(),
        V52[:, :rank].flatten(),
        W52[:, :rank].flatten()
    ])

    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 15})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Random perturbations
    for seed in range(3):
        if time.time() > deadline - 10:
            break
        np.random.seed(seed)
        x0 = np.concatenate([
            U52[:, :rank].flatten(),
            V52[:, :rank].flatten(),
            W52[:, :rank].flatten()
        ])
        x0 += np.random.randn(n_params) * 0.03
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 80, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 15})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
            if best_obj < 1e-12:
                break
        except Exception:
            continue

    if best_params is None or best_obj > 1e-8:
        return None, None, None

    U = best_params[:n*m*rank].reshape(n*m, rank)
    V = best_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W = best_params[n*m*rank + m*p*rank :].reshape(p*n, rank)

    # Snap to half-integers and re-verify
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    res_snap = _residuals_full(U_snap, V_snap, W_snap)
    if np.max(np.abs(res_snap)) < 1e-6:
        return U_snap, V_snap, W_snap

    if np.allclose(U*2, np.round(U*2), atol=1e-5) and \
       np.allclose(V*2, np.round(V*2), atol=1e-5) and \
       np.allclose(W*2, np.round(W*2), atol=1e-5):
        return U_snap, V_snap, W_snap

    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    res_int = _residuals_full(U_int, V_int, W_int)
    if np.max(np.abs(res_int)) < 1e-6:
        return U_int, V_int, W_int

    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Baseline: rank-53 solution
    U53, V53, W53 = _build_rank53()
    if not _verify_decomposition(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
        U53, V53, W53 = _build_schoolbook(n, m, p)
        return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}

    # Try to find a rank-52 solution
    if time.time() < deadline - 30:
        U52, V52, W52 = _try_rank52(U53, V53, W53, deadline - 10)
        if U52 is not None:
            if _verify_decomposition(U52, V52, W52, n, m, p, tol=1e-6):
                # Try to go further to rank-51
                if time.time() < deadline - 10:
                    U51, V51, W51 = _try_rank51(U52, V52, W52, deadline)
                    if U51 is not None:
                        if _verify_decomposition(U51, V51, W51, n, m, p, tol=1e-6):
                            return {"U": U51.tolist(), "V": V51.tolist(), "W": W51.tolist()}
                return {"U": U52.tolist(), "V": V52.tolist(), "W": W52.tolist()}

    # Fallback: rank-53
    return {"U": U53.tolist(), "V": V53.tolist(), "W": W53.tolist()}
# EVOLVE-BLOCK-END
