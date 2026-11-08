import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

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

    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4

    def snap_to_eighth(arr):
        return np.round(arr * 8) / 8

    def try_snap_validate(U, V, W):
        # Try half-integers first
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        # Try integers
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
        # Try quarters
        U_q = snap_to_quarter(U)
        V_q = snap_to_quarter(V)
        W_q = snap_to_quarter(W)
        if validate(U_q, V_q, W_q):
            return True, U_q, V_q, W_q
        # Try eighths
        U_e = snap_to_eighth(U)
        V_e = snap_to_eighth(V)
        W_e = snap_to_eighth(W)
        if validate(U_e, V_e, W_e):
            return True, U_e, V_e, W_e
        return False, None, None, None

    def build_rank50():
        """Standard schoolbook rank-50 decomposition."""
        rank = n * m * p  # 50
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
        return U, V, W

    def build_rank46():
        """Strassen-based rank-46 decomposition for 2x5 * 5x5."""
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
        # Handle the middle column/row interactions
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

    def newton_polish(U, V, W, max_iter=100, tol=1e-16):
        """High-precision Newton-style polish."""
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
            if grad_norm < 1e-18:
                break
            
            # Adaptive step size for stability
            lr = 0.005 * min(1.0, 1e-8 / max(grad_norm, 1e-20))

            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW

        return U, V, W

    def fast_optimize(U_init, V_init, W_init, target_rank, max_iter=200, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer with adaptive restarts."""
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
        restart_count = 0
        max_restarts = 5

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

            # Stagnation detection with restart
            if loss > prev_loss * 1.005 and no_improve > 15:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Perturb to escape local minimum
                U += np.random.randn(*U.shape) * 0.01
                V += np.random.randn(*V.shape) * 0.01
                W += np.random.randn(*W.shape) * 0.01
                mU = np.zeros_like(U)
                vU = np.zeros_like(U)
                mV = np.zeros_like(V)
                vV = np.zeros_like(V)
                mW = np.zeros_like(W)
                vW = np.zeros_like(W)
                no_improve = 0
                prev_loss = loss
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
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)

            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)

            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        return best_U, best_V, best_W, best_loss

    def try_reduce(base_U, base_V, base_W, target_rank, max_trials=10):
        """Try to reduce rank by removing columns and optimizing."""
        base_rank = base_U.shape[1]
        num_remove = base_rank - target_rank
        if num_remove <= 0:
            return None

        # Score columns by support size
        col_scores = []
        for c in range(base_rank):
            support_size = (np.count_nonzero(base_U[:, c]) +
                           np.count_nonzero(base_V[:, c]) +
                           np.count_nonzero(base_W[:, c]))
            col_scores.append((support_size, c))
        col_scores.sort()

        # Try removing the num_remove columns with smallest support
        cols_to_remove = [c for _, c in col_scores[:num_remove]]
        mask = [i for i in range(base_rank) if i not in cols_to_remove]

        if len(mask) != target_rank:
            return None

        U_test = base_U[:, mask].copy()
        V_test = base_V[:, mask].copy()
        W_test = base_W[:, mask].copy()

        for trial in range(max_trials):
            if time.time() > deadline - 30:
                break

            np.random.seed(42 + trial * 17)
            noise_scale = 0.01 * (trial + 1)
            U_try = U_test + np.random.randn(*U_test.shape) * noise_scale
            V_try = V_test + np.random.randn(*V_test.shape) * noise_scale
            W_try = W_test + np.random.randn(*W_test.shape) * noise_scale

            U_opt, V_opt, W_opt, loss = fast_optimize(
                U_try, V_try, W_try, target_rank,
                max_iter=200, lr=0.06
            )

            if loss < 1e-4:
                valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                if valid:
                    return U_s, V_s, W_s

            # Also try Newton polish
            U2, V2, W2 = newton_polish(U_opt, V_opt, W_opt, max_iter=50)
            valid, U_s, V_s, W_s = try_snap_validate(U2, V2, W2)
            if valid:
                return U_s, V_s, W_s

        return None

    # Build rank-46 baseline
    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        # Fallback to rank-50
        U50, V50, W50 = build_rank50()
        return {
            "U": [[float(x) for x in row] for row in U50],
            "V": [[float(x) for x in row] for row in V50],
            "W": [[float(x) for x in row] for row in W50]
        }

    # Try to reduce from 46 to lower ranks
    best_result = None

    # Try rank 45
    if time.time() < deadline - 60:
        result = try_reduce(U46, V46, W46, 45, max_trials=8)
        if result is not None:
            best_result = result

    # Try rank 44
    if best_result is None and time.time() < deadline - 50:
        result = try_reduce(U46, V46, W46, 44, max_trials=6)
        if result is not None:
            best_result = result

    # Try rank 43
    if best_result is None and time.time() < deadline - 40:
        result = try_reduce(U46, V46, W46, 43, max_trials=4)
        if result is not None:
            best_result = result

    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }

    # Return rank-46
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
# EVOLVE-BLOCK-END
