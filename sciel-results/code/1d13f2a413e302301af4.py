import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]


def _target_tensor(n, m, p):
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _tensor_residual(U, V, W):
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    return T - 1.0


def _objective_complex(params, n, m, p, rank):
    u_sz = n * m * rank
    v_sz = m * p * rank
    re_U = params[:u_sz]
    im_U = params[u_sz:2*u_sz]
    re_V = params[2*u_sz:2*u_sz + v_sz]
    im_V = params[2*u_sz + v_sz:2*u_sz + 2*v_sz]
    re_W = params[2*u_sz + 2*v_sz:2*u_sz + 2*v_sz + v_sz]
    im_W = params[2*u_sz + 2*v_sz + v_sz:2*u_sz + 3*v_sz]

    U = (re_U + 1j*im_U).reshape(n, m, rank)
    V = (re_V + 1j*im_V).reshape(m, p, rank)
    W = (re_W + 1j*im_W).reshape(p, n, rank)

    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    resid = (T - 1.0).ravel()
    return float(resid.real @ resid.real + resid.imag @ resid.imag)


def _grad_complex(params, n, m, p, rank):
    u_sz = n * m * rank
    v_sz = m * p * rank
    re_U = params[:u_sz].reshape(n, m, rank)
    im_U = params[u_sz:2*u_sz].reshape(n, m, rank)
    re_V = params[2*u_sz:2*u_sz + v_sz].reshape(m, p, rank)
    im_V = params[2*u_sz + v_sz:2*u_sz + 2*v_sz].reshape(m, p, rank)
    re_W = params[2*u_sz + 2*v_sz:2*u_sz + 2*v_sz + v_sz].reshape(p, n, rank)
    im_W = params[2*u_sz + 2*v_sz + v_sz:2*u_sz + 3*v_sz].reshape(p, n, rank)

    U = re_U + 1j*im_U
    V = re_V + 1j*im_V
    W = re_W + 1j*im_W

    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    resid = T - 1.0  # complex

    # dF/dU = 2 Re( conj(resid) * (V * W) ) where V*W is elementwise over (j,k)
    VW = V * W  # (j,k) complex broadcast
    U_grad = 2.0 * np.real(np.conj(resid) * VW)

    UW = U * W  # (i,k)
    V_grad = 2.0 * np.real(np.conj(resid) * UW)

    UV = U * V  # (i,j)
    W_grad = 2.0 * np.real(np.conj(resid) * UV)

    return np.concatenate([
        U_grad.ravel(), (2.0*np.imag(np.conj(resid)*VW)).ravel(),
        V_grad.ravel(), (2.0*np.imag(np.conj(resid)*UW)).ravel(),
        W_grad.ravel(), (2.0*np.imag(np.conj(resid)*UV)).ravel(),
    ])


def _rationalize_scalar(x, max_den=64):
    if abs(x) < 1e-7:
        return 0.0
    # Try small rational a/b with small b
    best = None
    best_err = abs(x)
    for b in range(1, max_den + 1):
        a = round(x * b)
        if a == 0:
            continue
        cand = a / b
        e = abs(cand - x)
        if e < best_err * 0.999 and e < 1e-4:
            best_err = e
            best = cand
    if best is not None:
        return best
    # Snap to nearest half-integer
    return round(2*x) / 2.0


def _rationalize_matrix(M, max_den=64):
    out = np.zeros_like(M, dtype=float)
    for idx in np.ndindex(M.shape):
        out[idx] = _rationalize_scalar(M[idx], max_den=max_den)
    return out


def _verify_exact(U, V, W, n, m, p, trials=30):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        M = np.einsum('ijr,ij->r', U3, A)
        N = np.einsum('jkr,jk->r', V3, B)
        C_comp = np.einsum('kir,r->ik', W3, M * N)
        if not np.allclose(C_comp, C_exp, atol=1e-9):
            return False
    return True


def _make_rank11(n, m, p):
    rank = 11
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    def set_u(i, j, r, val): U[i*m + j][r] = val
    def set_v(j, k, r, val): V[j*p + k][r] = val
    def set_w(k, i, r, val): W[k*n + i][r] = val

    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    set_w(0,0,0,1); set_w(1,1,0,1)

    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    set_w(0,1,1,1); set_w(1,1,1,-1)

    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    set_w(1,0,2,1); set_w(1,1,2,1)

    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    set_w(0,0,3,1); set_w(0,1,3,1)

    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    set_w(0,0,4,-1); set_w(1,0,4,1)

    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    set_w(1,1,5,1)

    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    set_w(0,0,6,1)

    set_u(0,0,7,1); set_v(0,2,7,1); set_w(2,0,7,1)
    set_u(0,1,8,1); set_v(1,2,8,1); set_w(2,0,8,1)
    set_u(1,0,9,1); set_v(0,2,9,1); set_w(2,1,9,1)
    set_u(1,1,10,1); set_v(1,2,10,1); set_w(2,1,10,1)

    return {"U": U, "V": V, "W": W}


def _pack_complex(U, V, W, n, m, p, rank):
    u_sz = n*m*rank
    v_sz = m*p*rank
    parts = []
    for M in (U, V, W):
        parts.append(np.real(M).ravel())
    for M in (U, V, W):
        parts.append(np.imag(M).ravel())
    return np.concatenate(parts)


def _unpack_complex(params, n, m, p, rank):
    u_sz = n*m*rank
    v_sz = m*p*rank
    re_U = params[:u_sz].reshape(n, m, rank)
    im_U = params[u_sz:2*u_sz].reshape(n, m, rank)
    re_V = params[2*u_sz:2*u_sz+v_sz].reshape(m, p, rank)
    im_V = params[2*u_sz+v_sz:2*u_sz+2*v_sz].reshape(m, p, rank)
    re_W = params[2*u_sz+2*v_sz:2*u_sz+2*v_sz+v_sz].reshape(p, n, rank)
    im_W = params[2*u_sz+2*v_sz+v_sz:2*u_sz+3*v_sz].reshape(p, n, rank)
    return (re_U + 1j*im_U, re_V + 1j*im_V, re_W + 1j*im_W)


def _homotopy_from_rank11_to_rank10(n, m, p):
    """Treat rank-11 as a specialization of rank-10 by combining two factors,
    then optimize rank-10 complex parameters from that seed."""
    sol11 = _make_rank11(n, m, p)
    U11 = np.array(sol11['U'], dtype=float).reshape(n, m, 11)
    V11 = np.array(sol11['V'], dtype=float).reshape(m, p, 11)
    W11 = np.array(sol11['W'], dtype=float).reshape(p, n, 11)

    rank = 10
    # Seed by dropping one factor (least-norm) and averaging into neighbors
    U10 = U11[:, :, :9].astype(complex)
    V10 = V11[:, :, :9].astype(complex)
    W10 = W11[:, :, :9].astype(complex)
    # Blend the dropped factor (index 9) into factor 8 as a complex perturbation
    U10[:, :, 8] += 0.5 * U11[:, :, 9]
    V10[:, :, 8] += 0.5 * V11[:, :, 9]
    W10[:, :, 8] += 0.5j * W11[:, :, 9]
    # Add a small complex component to factor 9 (new) to break realness
    U10[:, :, 9] = (U11[:, :, 9] + 1j * U11[:, :, 8]) * 0.5
    V10[:, :, 9] = (V11[:, :, 9] + 1j * V11[:, :, 8]) * 0.5
    W10[:, :, 9] = (W11[:, :, 9] + 1j * W11[:, :, 8]) * 0.5

    return _pack_complex(U10, V10, W10, n, m, p, rank)


def _search_rank10_complex(n, m, p, deadline_fn, max_iters=400):
    rank = 10
    best_err = float('inf')
    best_params = None

    seeds = []
    # Seed 1: homotopy from rank-11
    seeds.append(_homotopy_from_rank11_to_rank10(n, m, p))

    # Seeds 2-4: random complex initializations of various scales
    np.random.seed(1234)
    total = 3 * (n*m + m*p + p*n) * rank
    for scale in (0.3, 0.7, 1.5):
        seeds.append(np.random.randn(total) * scale)

    for s, p0 in enumerate(seeds):
        if deadline_fn() < 15:
            break
        try:
            res = minimize(_objective_complex, p0,
                           args=(n, m, p, rank),
                           jac=_grad_complex,
                           method='L-BFGS-B',
                           options={'maxiter': max_iters, 'ftol': 1e-14,
                                    'gtol': 1e-9, 'disp': False,
                                    'maxfun': 8000})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x.copy()
        except Exception:
            continue

    # Polish
    if best_params is not None and best_err < 1e-4 and deadline_fn() > 10:
        try:
            res2 = minimize(_objective_complex, best_params,
                            args=(n, m, p, rank),
                            jac=_grad_complex,
                            method='L-BFGS-B',
                            options={'maxiter': 800, 'ftol': 1e-16,
                                     'gtol': 1e-11, 'disp': False,
                                     'maxfun': 20000})
            if res2.fun < best_err:
                best_err = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    if best_params is None or best_err > 1e-6:
        return None

    U, V, W = _unpack_complex(best_params, n, m, p, rank)

    # Project to rational/integer lattice
    Ur = _rationalize_matrix(np.real(U))
    Vr = _rationalize_matrix(np.real(V))
    Wr = _rationalize_matrix(np.real(W))

    # Check exactness of the rational (real) projection
    Ua = Ur.reshape(n, m, rank)
    Va = Vr.reshape(m, p, rank)
    Wa = Wr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', Ua, Va, Wa)
    if np.allclose(T, 1.0, atol=1e-9):
        U_list = [[0.0]*rank for _ in range(n*m)]
        V_list = [[0.0]*rank for _ in range(m*p)]
        W_list = [[0.0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    U_list[i*m+j][r] = float(Ur[i,j,r])
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    V_list[j*p+k][r] = float(Vr[j,k,r])
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_list[k*n+i][r] = float(Wr[k,i,r])
        sol = {"U": U_list, "V": V_list, "W": W_list}
        if _verify_exact(Ur, Vr, Wr, n, m, p, trials=25):
            return sol

    # If real projection fails, try also projecting the imaginary part and
    # forming a combined real decomposition is not valid; instead fall back
    # to using the complex solution's real part snapped to half-integers.
    Ur2 = np.round(np.real(U) * 2) / 2.0
    Vr2 = np.round(np.real(V) * 2) / 2.0
    Wr2 = np.round(np.real(W) * 2) / 2.0
    Ua2 = Ur2.reshape(n, m, rank)
    Va2 = Vr2.reshape(m, p, rank)
    Wa2 = Wr2.reshape(p, n, rank)
    T2 = np.einsum('ijr,jkr,kir->ijk', Ua2, Va2, Wa2)
    if np.allclose(T2, 1.0, atol=1e-9):
        U_list = [[0.0]*rank for _ in range(n*m)]
        V_list = [[0.0]*rank for _ in range(m*p)]
        W_list = [[0.0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    U_list[i*m+j][r] = float(Ur2[i,j,r])
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    V_list[j*p+k][r] = float(Vr2[j,k,r])
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W_list[k*n+i][r] = float(Wr2[k,i,r])
        sol = {"U": U_list, "V": V_list, "W": W_list}
        if _verify_exact(Ur2, Vr2, Wr2, n, m, p, trials=25):
            return sol

    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = 145

    def time_left():
        return DEADLINE - (time.time() - t0)

    sol11 = _make_rank11(n, m, p)
    U11 = np.array(sol11['U'], dtype=float)
    V11 = np.array(sol11['V'], dtype=float)
    W11 = np.array(sol11['W'], dtype=float)
    if _verify_exact(U11, V11, W11, n, m, p, trials=5):
        pass

    sol10 = _search_rank10_complex(n, m, p, time_left, max_iters=400)
    if sol10 is not None:
        return sol10
    return sol11
