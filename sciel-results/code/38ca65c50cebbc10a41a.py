import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def _build_rank50():
    """Construct a rank-50 exact decomposition via block structure:
       - 2x2x9 block (j in {0,1}): Strassen on 4 blocks of k-pairs + schoolbook for k=8
       - 2x1x9 block (j=2): schoolbook
    Returns U (6x50), V (27x50), W (18x50) as numpy arrays.
    """
    n, m, p = 2, 3, 9
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0

    def add_term(U_entries, V_entries, W_entries, c_idx):
        for (u_idx, u_val) in U_entries:
            U[u_idx, c_idx] += u_val
        for (v_idx, v_val) in V_entries:
            V[v_idx, c_idx] += v_val
        for (w_idx, w_val) in W_entries:
            W[w_idx, c_idx] += w_val

    # --- 2x2x9 block: Strassen for k in {0..7} (4 groups of 2) ---
    for g in range(4):
        k0 = 2 * g
        k1 = 2 * g + 1
        # M1 = (a00+a11)*(b00+b01+b10+b11)
        add_term(
            [(0 * m + 0, 1), (1 * m + 1, 1)],
            [(0 * p + k0, 1), (0 * p + k1, 1), (1 * p + k0, 1), (1 * p + k1, 1)],
            [(k0 * n + 0, 1), (k0 * n + 1, 1), (k1 * n + 0, 1), (k1 * n + 1, 1)],
            c
        )
        c += 1
        # M2 = (a10+a11)*b00
        add_term(
            [(1 * m + 0, 1), (1 * m + 1, 1)],
            [(0 * p + k0, 1)],
            [(k0 * n + 1, 1), (k1 * n + 1, -1)],
            c
        )
        c += 1
        # M3 = a00*(b11-b01)
        add_term(
            [(0 * m + 0, 1)],
            [(1 * p + k1, 1), (0 * p + k1, -1)],
            [(k1 * n + 0, 1), (k1 * n + 1, 1)],
            c
        )
        c += 1
        # M4 = a11*(b00-b10)
        add_term(
            [(1 * m + 1, 1)],
            [(0 * p + k0, 1), (1 * p + k0, -1)],
            [(k0 * n + 0, -1), (k0 * n + 1, 1)],
            c
        )
        c += 1
        # M5 = (a00+a01)*b11
        add_term(
            [(0 * m + 0, 1), (0 * m + 1, 1)],
            [(1 * p + k1, 1)],
            [(k0 * n + 0, 1), (k1 * n + 0, 1)],
            c
        )
        c += 1
        # M6 = (a10-a00)*(b00+b01)
        add_term(
            [(1 * m + 0, 1), (0 * m + 0, -1)],
            [(0 * p + k0, 1), (0 * p + k1, 1)],
            [(k1 * n + 1, 1)],
            c
        )
        c += 1
        # M7 = (a01-a11)*(b10+b11)
        add_term(
            [(0 * m + 1, 1), (1 * m + 1, -1)],
            [(1 * p + k0, 1), (1 * p + k1, 1)],
            [(k0 * n + 0, -1)],
            c
        )
        c += 1

    # --- k=8 schoolbook for 2x2x1 (rank 4) ---
    k8 = 8
    add_term([(0 * m + 0, 1)], [(0 * p + k8, 1)], [(k8 * n + 0, 1)], c); c += 1
    add_term([(0 * m + 1, 1)], [(1 * p + k8, 1)], [(k8 * n + 0, 1)], c); c += 1
    add_term([(1 * m + 0, 1)], [(0 * p + k8, 1)], [(k8 * n + 1, 1)], c); c += 1
    add_term([(1 * m + 1, 1)], [(1 * p + k8, 1)], [(k8 * n + 1, 1)], c); c += 1

    # --- 2x1x9 block (j=2): schoolbook, rank 18 ---
    for i in range(n):
        for k in range(p):
            add_term([(i * m + 2, 1)], [(2 * p + k, 1)], [(k * n + i, 1)], c)
            c += 1

    return U[:, :c], V[:, :c], W[:, :c]


def _build_tensor():
    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T


def _round_half(A):
    return np.round(A * 2) / 2


def _verify(U, V, W, T, tol=1e-8):
    T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
    err = np.sum((T - T_hat) ** 2)
    return err < tol, err


def _polish(U, V, W, T, start_time, time_limit, max_steps=3000):
    """High-precision polish: Adam-style gradient descent to drive error to 0,
    then attempt rounding to half-integers."""
    lr = 0.05
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    t = 0
    best_err = np.inf
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        t += 1
        T_hat = np.einsum('ur,vr,wr->uvw', U, V, W)
        E = T - T_hat
        err = np.sum(E ** 2)
        if err < best_err:
            best_err = err
        if err < 1e-14:
            break
        dU = -np.einsum('uvw,vr,wr->ur', E, V, W)
        dV = -np.einsum('uvw,ur,wr->vr', E, U, W)
        dW = -np.einsum('uvw,ur,vr->wr', E, U, V)
        mU = beta1 * mU + (1 - beta1) * dU; vU = beta2 * vU + (1 - beta2) * dU ** 2
        mV = beta1 * mV + (1 - beta1) * dV; vV = beta2 * vV + (1 - beta2) * dV ** 2
        mW = beta1 * mW + (1 - beta1) * dW; vW = beta2 * vW + (1 - beta2) * dW ** 2
        mU_hat = mU / (1 - beta1 ** t); vU_hat = vU / (1 - beta2 ** t)
        mV_hat = mV / (1 - beta1 ** t); vV_hat = vV / (1 - beta2 ** t)
        mW_hat = mW / (1 - beta1 ** t); vW_hat = vW / (1 - beta2 ** t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        if step % 200 == 0:
            lr *= 0.9
    return U, V, W, best_err


def construct():
    n, m, p = DIMENSIONS
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    T = _build_tensor()
    start_time = time.time()
    time_limit = 115  # safety margin before 150s kill

    # Start from known rank-50 structural solution
    U50, V50, W50 = _build_rank50()
    ok, err = _verify(U50, V50, W50, T)
    best_rank = 54
    best_U, best_V, best_W = None, None, None

    if ok:
        best_rank = 50
        best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()
        # Polish to ensure exactness and attempt rounding
        U50, V50, W50, _ = _polish(U50, V50, W50, T, start_time, time_limit, max_steps=2000)
        U50r, V50r, W50r = _round_half(U50), _round_half(V50), _round_half(W50)
        ok_r, err_r = _verify(U50r, V50r, W50r, T)
        if ok_r:
            best_U, best_V, best_W = U50r, V50r, W50r
        else:
            best_U, best_V, best_W = U50, V50, W50

    # Fallback schoolbook if structural solution failed
    if best_rank == 54:
        U_s = np.zeros((U_dim, 54)); V_s = np.zeros((V_dim, 54)); W_s = np.zeros((W_dim, 54))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j; v = j * p + k; w = k * n + i
                    U_s[u, col] = 1; V_s[v, col] = 1; W_s[w, col] = 1
                    col += 1
        best_U, best_V, best_W = U_s, V_s, W_s

    # Try to improve: probe ranks 49, 48, 47 with gradient search
    for R in range(best_rank - 1, max(30, best_rank - 6), -1):
        if time.time() - start_time > time_limit - 5:
            break
        max_restarts = 4
        found = False
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit - 5:
                break
            # Initialize near the known solution structure where possible
            if restart == 0:
                U = U50[:, :R] + np.random.randn(U_dim, R) * 0.05
                V = V50[:, :R] + np.random.randn(V_dim, R) * 0.05
                W = W50[:, :R] + np.random.randn(W_dim, R) * 0.05
            else:
                scale = 0.1
                U = np.random.randn(U_dim, R) * scale
                V = np.random.randn(V_dim, R) * scale
                W = np.random.randn(W_dim, R) * scale
            U, V, W, err = _polish(U, V, W, T, start_time, time_limit - 5, max_steps=1500)
            if err < 1e-10:
                Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
                ok_r, _ = _verify(Ur, Vr, Wr, T)
                if ok_r:
                    best_rank = R
                    best_U, best_V, best_W = Ur, Vr, Wr
                    found = True
                    break
        if found:
            break

    # Final safety: ensure we return something valid
    if best_U is None:
        U_s = np.zeros((U_dim, 54)); V_s = np.zeros((V_dim, 54)); W_s = np.zeros((W_dim, 54))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j; v = j * p + k; w = k * n + i
                    U_s[u, col] = 1; V_s[v, col] = 1; W_s[w, col] = 1
                    col += 1
        best_U, best_V, best_W = U_s, V_s, W_s

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
