import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def snap_to_integer(arr):
        return np.round(arr)

    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = snap_to_integer(U)
        V_i = snap_to_integer(V)
        W_i = snap_to_integer(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = np.round(U * 4) / 4
        V_q = np.round(V * 4) / 4
        W_q = np.round(W * 4) / 4
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1

            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1

            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1

            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1

            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1

        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1

        return U, V, W

    def newton_polish(U, V, W, max_iter=200, tol=1e-14):
        U = U.copy()
        V = V.copy()
        W = W.copy()

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < tol:
                break

            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break

            # Adaptive learning rate
            lr = 0.01 * min(1.0, 1e-6 / (grad_norm + 1e-12))

            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW

        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=400, lr=0.03, beta1=0.9, beta2=0.999, eps=1e-8, lr_decay=0.9995):
        R = target_rank
        U = U_init[:, :R].copy()
        V = V_init[:, :R].copy()
        W = W_init[:, :R].copy()

        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)

        prev_loss = float('inf')
        no_improve = 0
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        current_lr = lr
        restart_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 15:
                break

            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if loss < 1e-10:
                break

            # Restart logic if stuck
            if loss > prev_loss * 1.01 and no_improve > 20:
                restart_count += 1
                if restart_count > 8:
                    break
                U += np.random.randn(*U.shape) * 0.02
                V += np.random.randn(*V.shape) * 0.02
                W += np.random.randn(*W.shape) * 0.02
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
                current_lr = lr
                continue
            else:
                no_improve += 1
                prev_loss = loss

            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)

            t = it + 1
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= current_lr * mU_hat / (np.sqrt(vU_hat) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= current_lr * mV_hat / (np.sqrt(vV_hat) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= current_lr * mW_hat / (np.sqrt(vW_hat) + eps)

            current_lr *= lr_decay

        return best_U, best_V, best_W, best_loss

    def try_rank_reduction(U_init, V_init, W_init, target_rank, time_budget=30):
        """
        Try to reduce rank by removing columns and re-optimizing.
        """
        current_rank = U_init.shape[1]
        if current_rank == target_rank:
            return None, None, None

        col_start_time = time.time()
        max_attempts = current_rank

        # Prioritize border columns (last 18) for removal first
        border_cols = list(range(28, current_rank))
        strassen_cols = list(range(0, 28))
        # Interleave: try border first, then strassen
        cols_to_try = border_cols + strassen_cols

        for col_to_remove in cols_to_try:
            if time.time() > deadline - 20:
                break
            if time.time() - col_start_time > time_budget:
                break

            mask = [i for i in range(current_rank) if i != col_to_remove]
            U_sub = U_init[:, mask].copy()
            V_sub = V_init[:, mask].copy()
            W_sub = W_init[:, mask].copy()

            # Add slight noise to break symmetry
            U_sub += np.random.randn(*U_sub.shape) * 0.005
            V_sub += np.random.randn(*V_sub.shape) * 0.005
            W_sub += np.random.randn(*W_sub.shape) * 0.005

            # Run optimization with limited iterations based on remaining time
            remaining = deadline - 25 - time.time()
            max_iter = max(50, min(300, int(remaining * 5)))

            U_opt, V_opt, W_opt, loss = fast_optimize(U_sub, V_sub, W_sub, target_rank, max_iter=max_iter)

            if loss < 1e-5:
                # If we get close, polish it
                U_pol, V_pol, W_pol = newton_polish(U_opt, V_opt, W_pol if False else W_opt)
                valid, U_s, V_s, W_s = try_snap_validate(U_pol, V_pol, W_pol)
                if valid:
                    return U_s, V_s, W_s

        return None, None, None

    # Build and verify rank-46
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to naive rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_U, best_V, best_W = U46, V46, W46
    best_rank = 46

    np.random.seed(42)

    # Try to reduce to 45
    if time.time() < deadline - 35:
        U45, V45, W45 = try_rank_reduction(U46, V46, W46, 45, time_budget=40)
        if U45 is not None:
            best_U, best_V, best_W = U45, V45, W45
            best_rank = 45

    # If still have time, try to reduce to 44
    if best_rank == 45 and time.time() < deadline - 20:
        U44, V44, W44 = try_rank_reduction(best_U, best_V, best_W, 44, time_budget=15)
        if U44 is not None:
            best_U, best_V, best_W = U44, V44, W44
            best_rank = 44

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
