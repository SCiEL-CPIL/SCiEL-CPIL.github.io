import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 7]


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify U,V,W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            rhs = 0.0
                            for r in range(rank):
                                rhs += U[i*m+j, r] * V[jp*p+kp, r] * W[kp*n+ip, r]
                            if abs(lhs - rhs) > tol:
                                return False
    return True


def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W


def _build_strassen_rank25(n, m, p):
    """Exact rank-25 decomposition: Strassen (7) per 2-col block + schoolbook (4) for last col."""
    n, m, p = 2, 2, 7
    rank = 25
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    def add_strassen(col_start, k_start):
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1
        # M2 = (A10 + A11) * B0,ks
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1
        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1
        # M4 = A11 * (B1,ks - B0,ks)
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1
        # M5 = (A00 + A01) * B1,ks+1
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1
        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1
        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1

    add_strassen(0, 0)
    add_strassen(7, 2)
    add_strassen(14, 4)

    # Schoolbook for column 6 (4 terms)
    col_start = 21
    U[0, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    U[1, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 0, col_start+1] += 1
    U[2, col_start+2] += 1
    V[7*0 + 6, col_start+2] += 1
    W[2*6 + 1, col_start+2] += 1
    U[3, col_start+3] += 1
    V[7*1 + 6, col_start+3] += 1
    W[2*6 + 1, col_start+3] += 1

    return U, V, W


def _build_tensor_flat(n, m, p):
    n_m = n*m
    m_p = m*p
    p_n = p*n
    T = np.zeros((n_m, m_p, p_n))
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            if i == ip and j == jp and k == kp:
                                T[i*m+j, jp*p+k, kp*n+ip] = 1.0
    return T.flatten()


def _try_reduce_rank(U0, V0, W0, n, m, p, target_rank, deadline, max_trials=6):
    """Try to find a rank-target_rank decomposition seeded from the given (larger) solution."""
    n_m = n*m
    m_p = m*p
    p_n = p*n
    T_flat = _build_tensor_flat(n, m, p)

    def residuals(params):
        U_r = params[:n_m*target_rank].reshape(n_m, target_rank)
        V_r = params[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
        W_r = params[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
        T_calc = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r)
        return T_calc.flatten() - T_flat

    r0 = U0.shape[1]
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        # Seed: keep the first target_rank columns of the known solution (Strassen),
        # optionally perturb to escape local minima.
        np.random.seed(7 + trial)
        scale = 0.0 if trial == 0 else (0.1 + 0.15*trial)
        U = U0[:, :min(target_rank, r0)].copy()
        V = V0[:, :min(target_rank, r0)].copy()
        W = W0[:, :min(target_rank, r0)].copy()
        if target_rank > r0:
            U = np.hstack([U, np.random.randn(n_m, target_rank-r0)*0.3])
            V = np.hstack([V, np.random.randn(m_p, target_rank-r0)*0.3])
            W = np.hstack([W, np.random.randn(p_n, target_rank-r0)*0.3])
        if scale > 0:
            U = U + np.random.randn(n_m, target_rank)*scale
            V = V + np.random.randn(m_p, target_rank)*scale
            W = W + np.random.randn(p_n, target_rank)*scale

        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        try:
            result = least_squares(residuals, x0, method='lm',
                                   max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            if result.success or np.linalg.norm(result.fun) < 1e-9:
                U_new = result.x[:n_m*target_rank].reshape(n_m, target_rank)
                V_new = result.x[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
                W_new = result.x[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
                if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                    return U_new, V_new, W_new
        except Exception:
            pass
    return None, None, None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n, m, p = DIMENSIONS

    # Base: exact rank-25 Strassen decomposition (guaranteed valid).
    U, V, W = _build_strassen_rank25(n, m, p)
    best = (U, V, W)

    if not _verify_decomposition(U, V, W, n, m, p):
        # Should never happen; fall back to schoolbook.
        U, V, W = _make_schoolbook(n, m, p)
        best = (U, V, W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to shave rank off the Strassen solution using the refinement machinery.
    for target_rank in (24, 23, 22):
        if time.time() > deadline - 20:
            break
        U_new, V_new, W_new = _try_reduce_rank(U, V, W, n, m, p, target_rank, deadline - 15)
        if U_new is not None:
            best = (U_new, V_new, W_new)
            break  # got a smaller rank; stop (lower is better)

    return {"U": best[0].tolist(), "V": best[1].tolist(), "W": best[2].tolist()}
