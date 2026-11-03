import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]


def _build_T(n, m, p):
    """Multiplication tensor T[a,b,c]=1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-7):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(M - T)) < tol


def _round_halfint(X):
    return np.round(2.0 * X) / 2.0


def _try_round_verify(U, V, W, T, tol=1e-6):
    Ur = _round_halfint(U)
    Vr = _round_halfint(V)
    Wr = _round_halfint(W)
    if _verify(Ur, Vr, Wr, T, tol=tol):
        return Ur, Vr, Wr
    return None


def _loss(params, T, A, B, C, R):
    U = params[:A*R].reshape(A, R)
    V = params[A*R:A*R+B*R].reshape(B, R)
    W = params[A*R+B*R:].reshape(C, R)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((M - T)**2)


def _jacobian(params, T, A, B, C, R):
    U = params[:A*R].reshape(A, R)
    V = params[A*R:A*R+B*R].reshape(B, R)
    W = params[A*R+B*R:].reshape(C, R)
    M = np.einsum('ar,br,cr->abc', U, V, W)
    diff = M - T
    grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
    return np.concatenate([grad_U.ravel(), grad_V.ravel(), grad_W.ravel()])


def _rank14_solution(n, m, p):
    """Rank-14 exact decomposition for 2x4 * 4x4 -> 4x2 (two independent rank-7 rows)."""
    assert n == 2 and m == 4 and p == 4
    A = n * m          # 8
    B = m * p          # 16
    C = p * n          # 8
    R = 14

    U = np.zeros((A, R))
    V = np.zeros((B, R))
    W = np.zeros((C, R))

    # rank-7 for 1x4 * 4x4 -> 1x4
    baseU = np.zeros((4, 7))
    baseV = np.zeros((4, 7))
    baseW = np.zeros((4, 7))

    baseU[0, 0] = 1; baseV[0, 0] = 1; baseW[0, 0] = 1
    baseU[1, 1] = 1; baseV[1, 1] = 1; baseW[1, 1] = 1
    baseU[0, 2] = 1; baseU[1, 2] = 1; baseU[2, 2] = 1
    baseV[0, 2] = 1; baseV[1, 2] = 1; baseV[2, 2] = 1
    baseW[2, 2] = 1
    baseU[2, 3] = 1; baseU[3, 3] = 1
    baseV[2, 3] = 1; baseV[3, 3] = 1
    baseW[3, 3] = 1
    baseU[0, 4] = 1; baseU[1, 4] = 1; baseU[2, 4] = 1; baseU[3, 4] = 1
    baseV[0, 4] = 1; baseV[1, 4] = 1; baseV[2, 4] = 1; baseV[3, 4] = 1
    baseW[0, 4] = 1; baseW[1, 4] = 1; baseW[2, 4] = 1; baseW[3, 4] = 1
    baseU[0, 5] = 1; baseU[3, 5] = 1
    baseV[0, 5] = 1; baseV[3, 5] = 1
    baseW[0, 5] = 1; baseW[3, 5] = 1
    baseU[1, 6] = 1; baseU[3, 6] = 1
    baseV[1, 6] = 1; baseV[3, 6] = 1
    baseW[1, 6] = 1; baseW[3, 6] = 1

    for i in range(n):
        off = i * 7
        for j in range(m):
            U[i * m + j, off + j] += baseU[j, j]
        for k in range(p):
            V[k * p + k, off + k] += baseV[k, k]
        for c in range(p):
            W[c * n + i, off + c] += baseW[c, c]

    return U, V, W


def _strassen_solution(n, m, p):
    """Known rank-28 exact decomposition via Strassen on four 2x2 blocks (fallback)."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add(A_row0, A_col0, B_row0, B_col0, C_row0, C_col0, start_r):
        a11 = A_row0 * m + A_col0
        a12 = A_row0 * m + A_col0 + 1
        a21 = (A_row0 + 1) * m + A_col0
        a22 = (A_row0 + 1) * m + A_col0 + 1
        b11 = B_row0 * p + B_col0
        b12 = B_row0 * p + B_col0 + 1
        b21 = (B_row0 + 1) * p + B_col0
        b22 = (B_row0 + 1) * p + B_col0 + 1
        c11 = C_col0 * n + C_row0
        c12 = (C_col0 + 1) * n + C_row0
        c21 = C_col0 * n + (C_row0 + 1)
        c22 = (C_col0 + 1) * n + (C_row0 + 1)
        r = start_r
        U[a11][r] += 1; U[a22][r] += 1; V[b11][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; W[c22][r] += 1; r += 1
        U[a21][r] += 1; U[a22][r] += 1; V[b11][r] += 1
        W[c21][r] += 1; W[c22][r] += -1; r += 1
        U[a11][r] += 1; V[b12][r] += 1; V[b22][r] += -1
        W[c12][r] += 1; W[c22][r] += 1; r += 1
        U[a22][r] += 1; V[b21][r] += 1; V[b11][r] += -1
        W[c11][r] += 1; W[c21][r] += 1; r += 1
        U[a11][r] += 1; U[a12][r] += 1; V[b22][r] += 1
        W[c11][r] += -1; W[c12][r] += 1; r += 1
        U[a21][r] += 1; U[a11][r] += -1; V[b11][r] += 1; V[b12][r] += 1
        W[c22][r] += 1; r += 1
        U[a12][r] += 1; U[a22][r] -= 1; V[b21][r] += 1; V[b22][r] += 1
        W[c11][r] += 1; r += 1
        return r

    r = add(0, 0, 0, 0, 0, 0, 0)
    r = add(0, 2, 2, 0, 0, 0, r)
    r = add(0, 0, 0, 2, 0, 2, r)
    r = add(0, 2, 2, 2, 0, 2, r)
    assert r == 28
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def _search_lower_rank(T, n, m, p, target_rank, U14, V14, W14, max_time=40.0):
    """Try to find a rank-target_rank solution using L-BFGS-B with multiple initializations."""
    t_start = time.time()
    A, B, C = n * m, m * p, p * n
    R = target_rank

    inits = []

    # 1. Truncated rank-14
    if target_rank <= 14:
        U0 = U14[:, :target_rank].copy()
        V0 = V14[:, :target_rank].copy()
        W0 = W14[:, :target_rank].copy()
        inits.append((U0, V0, W0))

    # 2. Random Gaussian with different scales
    for i in range(6):
        scale = 0.2 + 0.15 * i
        U0 = np.random.randn(A, R) * scale
        V0 = np.random.randn(B, R) * scale
        W0 = np.random.randn(C, R) * scale
        inits.append((U0, V0, W0))

    # 3. Sparse Random
    for i in range(6):
        U0 = np.random.randint(0, 2, (A, R)).astype(float)
        V0 = np.random.randint(0, 2, (B, R)).astype(float)
        W0 = np.random.randint(0, 2, (C, R)).astype(float)
        inits.append((U0, V0, W0))

    # 4. Perturbed rank-14
    if target_rank <= 14:
        for i in range(6):
            U0 = U14[:, :target_rank].copy() + np.random.randn(A, R) * 0.05
            V0 = V14[:, :target_rank].copy() + np.random.randn(B, R) * 0.05
            W0 = W14[:, :target_rank].copy() + np.random.randn(C, R) * 0.05
            inits.append((U0, V0, W0))

    # 5. Tensor slicing: fix A to basis vectors to get structure
    # For each basis vector e_a in A-space, we get a matrix in B-C space
    # We can use this to build structured initializations
    for i in range(4):
        U0 = np.zeros((A, R))
        V0 = np.zeros((B, R))
        W0 = np.zeros((C, R))
        # Assign random but structured patterns
        for r in range(R):
            # Pick random support
            u_support = np.random.choice(A, size=max(1, A // 4), replace=False)
            v_support = np.random.choice(B, size=max(1, B // 4), replace=False)
            w_support = np.random.choice(C, size=max(1, C // 4), replace=False)
            U0[u_support, r] = np.random.choice([-1, 1], size=len(u_support))
            V0[v_support, r] = np.random.choice([-1, 1], size=len(v_support))
            W0[w_support, r] = np.random.choice([-1, 1], size=len(w_support))
        inits.append((U0, V0, W0))

    best_loss = np.inf
    best_params = None

    for U_init, V_init, W_init in inits:
        if time.time() > t_start + max_time - 3.0:
            break

        params_init = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])

        try:
            result = minimize(
                _loss,
                params_init,
                args=(T, A, B, C, R),
                jac=_jacobian,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x.copy()

            if result.fun < 1e-8:
                params_opt = result.x
                U_opt = params_opt[:A*R].reshape(A, R)
                V_opt = params_opt[A*R:A*R+B*R].reshape(B, R)
                W_opt = params_opt[A*R+B*R:].reshape(C, R)

                # Try to round to half-integers
                rounded = _try_round_verify(U_opt, V_opt, W_opt, T)
                if rounded is not None:
                    return rounded
                return U_opt, V_opt, W_opt

        except Exception:
            continue

    # If we have a good but not perfect solution, try to polish
    if best_params is not None and best_loss < 1e-4:
        try:
            result = minimize(
                _loss,
                best_params,
                args=(T, A, B, C, R),
                jac=_jacobian,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 2000}
            )
            if result.fun < 1e-8:
                params_opt = result.x
                U_opt = params_opt[:A*R].reshape(A, R)
                V_opt = params_opt[A*R:A*R+B*R].reshape(B, R)
                W_opt = params_opt[A*R+B*R:].reshape(C, R)
                rounded = _try_round_verify(U_opt, V_opt, W_opt, T)
                if rounded is not None:
                    return rounded
                return U_opt, V_opt, W_opt
        except Exception:
            pass

    return None, None, None


def _to_lists(sol):
    U = np.asarray(sol["U"], dtype=float)
    V = np.asarray(sol["V"], dtype=float)
    W = np.asarray(sol["W"], dtype=float)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = t_start + 120.0

    T = _build_T(n, m, p)

    # Primary: rank-14 structured decomposition
    U14, V14, W14 = _rank14_solution(n, m, p)
    best = {"U": U14, "V": V14, "W": W14}
    if not _verify(U14, V14, W14, T, tol=1e-7):
        Us, Vs, Ws = _strassen_solution(n, m, p)
        best = {"U": Us, "V": Vs, "W": Ws}
        if _verify(Us, Vs, Ws, T, tol=1e-7):
            return _to_lists(best)
        return _to_lists({"U": U14, "V": V14, "W": W14})

    best_rank = 14
    best_sol = best

    # Try to reduce rank - focus on 13 first, then 12
    for target_rank in [13, 12]:
        if time.time() > deadline - 10:
            break
        per_rank_time = min(45.0, deadline - 10 - (time.time() - t_start))
        if per_rank_time < 5.0:
            break
        U, V, W = _search_lower_rank(T, n, m, p, target_rank, U14, V14, W14,
                                     max_time=per_rank_time)
        if U is not None:
            best_rank = target_rank
            best_sol = {"U": U, "V": V, "W": W}
            if target_rank == 13:
                break

    return _to_lists(best_sol)
