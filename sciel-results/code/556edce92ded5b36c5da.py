import time
import random
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 140.0

    # Target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if (i_a, j_a, k_a) matches (i_b, j_b, k_b) appropriately
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Condition: i_a == i_c, j_a == j_b, k_b == k_c
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_exact(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    def tensor_error(U, V, W):
        err = np.einsum('ar,br,cr->abc', U, V, W) - T
        return np.linalg.norm(err)

    def als_solve(U, V, W, R, max_iter=100, tol=1e-14):
        """
        Accelerated ALS: precompute design matrices once per factor-update step,
        then solve all rows at once via batched least-squares.
        """
        for it in range(max_iter):
            if time.time() > deadline - 10.0:
                break

            # ---- Solve U ----
            # M_U[b, c, r] = V[b, r] * W[c, r]  ->  shape (m*p, p*n, R)
            M_U = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
            # Stack all target vectors: T[a, b, c] for each a -> shape (n*m, m*p*p*n)
            T_U = T.reshape(n * m, -1)  # (n*m, m*p*p*n)
            # Solve M_U^T X^T = T_U^T for all a simultaneously? No.
            # For each a: M_U @ U[a] = T_U[a]
            # Batched: U = lstsq(M_U, T_U.T).T  -> use np.linalg.lstsq on transposed
            # Better: solve (M_U^T M_U) U^T = M_U^T T_U^T
            # Use np.linalg.lstsq with multiple RHS
            U_new, _, _, _ = np.linalg.lstsq(M_U, T_U.T, rcond=None)
            U = U_new.T

            # ---- Solve V ----
            M_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
            T_V = np.transpose(T, (1, 0, 2)).reshape(m * p, -1)  # (m*p, n*m*p*n)
            V_new, _, _, _ = np.linalg.lstsq(M_V, T_V.T, rcond=None)
            V = V_new.T

            # ---- Solve W ----
            M_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
            T_W = np.transpose(T, (2, 0, 1)).reshape(p * n, -1)  # (p*n, n*m*m*p)
            W_new, _, _, _ = np.linalg.lstsq(M_W, T_W.T, rcond=None)
            W = W_new.T

            err = tensor_error(U, V, W)
            if err < tol:
                break

        return U, V, W, err

    def refine_exact(U, V, W, R, max_iter=50):
        """
        High-precision polish: use exact rational arithmetic in a local Newton-like step.
        We treat the residual as a small correction and solve for delta in each factor.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Compute current tensor product
            current = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T - current

            # Solve for delta_U: minimize ||residual - einsum(da, V, W)||
            # This is linear in delta_U: M_U @ delta_U[a] = residual[a]
            M_U = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
            T_U = residual.reshape(n * m, -1)
            dU, _, _, _ = np.linalg.lstsq(M_U, T_U.T, rcond=None)
            U = U + dU.T

            M_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
            T_V = np.transpose(residual, (1, 0, 2)).reshape(m * p, -1)
            dV, _, _, _ = np.linalg.lstsq(M_V, T_V.T, rcond=None)
            V = V + dV.T

            M_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
            T_W = np.transpose(residual, (2, 0, 1)).reshape(p * n, -1)
            dW, _, _, _ = np.linalg.lstsq(M_W, T_W.T, rcond=None)
            W = W + dW.T

            err = tensor_error(U, V, W)
            if err < 1e-15:
                break

        return U, V, W, tensor_error(U, V, W)

    def svd_init(R, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Random init for V and W with better scaling
        scale = 1.0 / np.sqrt(R)
        V_init = np.random.randn(m * p, R) * scale
        W_init = np.random.randn(p * n, R) * scale
        # Solve for U exactly
        M_U = (V_init[:, None, :] * W_init[None, :, :]).reshape(m * p * p * n, R)
        T_U = T.reshape(n * m, -1)
        U_init, _, _, _ = np.linalg.lstsq(M_U, T_U.T, rcond=None)
        U_init = U_init.T
        return U_init, V_init, W_init

    def try_quantize_and_verify(U, V, W):
        """Try multiple quantization strategies and verify."""
        # Strategy 1: round to half-integers
        for denom in [1, 2, 4]:
            U_q = np.round(U * denom) / denom
            V_q = np.round(V * denom) / denom
            W_q = np.round(W * denom) / denom

            U_list = U_q.tolist()
            V_list = V_q.tolist()
            W_list = W_q.tolist()

            if verify_exact(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}

        # Strategy 2: round to nearest integer only
        U_q = np.round(U)
        V_q = np.round(V)
        W_q = np.round(W)
        U_list = U_q.tolist()
        V_list = V_q.tolist()
        W_list = W_q.tolist()
        if verify_exact(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}

        return None

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks from 41 down to 30
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 20.0:
            break

        found = False
        # Try multiple random initializations with different seeds
        num_attempts = 30
        for attempt in range(num_attempts):
            if time.time() > deadline - 10.0:
                break

            seed = 42 + attempt * 7919
            U, V, W = svd_init(target_rank, seed=seed)

            # Coarse ALS
            U, V, W, err = als_solve(U, V, W, target_rank, max_iter=60)

            if err < 1e-8:
                # Fine polish
                U, V, W, err2 = refine_exact(U, V, W, target_rank, max_iter=30)

                if err2 < 1e-10:
                    result = try_quantize_and_verify(U, V, W)
                    if result is not None:
                        best = result
                        best_rank = target_rank
                        found = True
                        break

        if found:
            break

    return best
