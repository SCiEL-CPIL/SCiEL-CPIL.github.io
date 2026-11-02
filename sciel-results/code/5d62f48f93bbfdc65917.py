"""Tensor decomposition for 2x3x9 matrix multiplication."""
DIMENSIONS = [2, 3, 9]

import numpy as np
import time
from fractions import Fraction


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank 54)
    rank_school = n * m * p
    U_s = np.zeros((n * m, rank_school))
    V_s = np.zeros((m * p, rank_school))
    W_s = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1
                V_s[j * p + k, col] = 1
                W_s[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_s.copy()
    best_V = V_s.copy()
    best_W = W_s.copy()

    def _verify_exact(U, V, W, rank, tol=1e-10):
        """Verify that the decomposition is exact."""
        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        if np.max(np.abs(D - T)) < tol:
            return True
        return False

    def _try_fractional(U, V, W, rank, max_denom=100000, tol=1e-8):
        """Try to convert to exact fractions and verify."""
        try:
            U_f = np.zeros((n * m, rank), dtype=object)
            V_f = np.zeros((m * p, rank), dtype=object)
            W_f = np.zeros((p * n, rank), dtype=object)

            for i in range(n * m):
                for j in range(rank):
                    f = Fraction(U[i, j]).limit_denominator(max_denom)
                    if abs(float(f) - U[i, j]) > tol:
                        return None
                    U_f[i, j] = f

            for i in range(m * p):
                for j in range(rank):
                    f = Fraction(V[i, j]).limit_denominator(max_denom)
                    if abs(float(f) - V[i, j]) > tol:
                        return None
                    V_f[i, j] = f

            for i in range(p * n):
                for j in range(rank):
                    f = Fraction(W[i, j]).limit_denominator(max_denom)
                    if abs(float(f) - W[i, j]) > tol:
                        return None
                    W_f[i, j] = f

            # Verify with exact arithmetic using float approximation
            U_np = np.array([[float(x) for x in row] for row in U_f.tolist()], dtype=float)
            V_np = np.array([[float(x) for x in row] for row in V_f.tolist()], dtype=float)
            W_np = np.array([[float(x) for x in row] for row in W_f.tolist()], dtype=float)
            D = np.einsum('ar,br,cr->abc', U_np, V_np, W_np, optimize=True)
            if np.max(np.abs(D - T)) < 1e-12:
                return {
                    "U": [[float(x) for x in row] for row in U_f.tolist()],
                    "V": [[float(x) for x in row] for row in V_f.tolist()],
                    "W": [[float(x) for x in row] for row in W_f.tolist()]
                }
            return None
        except Exception:
            return None

    def _optimize_rank(rank, seed, n_steps=2000, lr_init=0.1):
        """Optimize a decomposition of given rank using Adam."""
        np.random.seed(seed)

        U = np.random.randn(n * m, rank) * 0.3
        V = np.random.randn(m * p, rank) * 0.3
        W = np.random.randn(p * n, rank) * 0.3

        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = lr_init

        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for step in range(n_steps):
            if time.time() > deadline - 3:
                break

            D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = D - T
            err_norm = np.linalg.norm(err)

            if err_norm < best_err:
                best_err = err_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if err_norm < 1e-10:
                break

            # Adaptive learning rate
            if step % 200 == 0 and step > 0:
                lr *= 0.8

            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V, optimize=True)

            t = step + 1
            # Adam updates
            m_U[:] = beta1 * m_U + (1 - beta1) * grad_U
            v_U[:] = beta2 * v_U + (1 - beta2) * (grad_U ** 2)
            m_U_hat = m_U / (1 - beta1 ** t)
            v_U_hat = v_U / (1 - beta2 ** t)
            U[:] -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

            m_V[:] = beta1 * m_V + (1 - beta1) * grad_V
            v_V[:] = beta2 * v_V + (1 - beta2) * (grad_V ** 2)
            m_V_hat = m_V / (1 - beta1 ** t)
            v_V_hat = v_V / (1 - beta2 ** t)
            V[:] -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

            m_W[:] = beta1 * m_W + (1 - beta1) * grad_W
            v_W[:] = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            W[:] -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

        return best_U, best_V, best_W, best_err

    def _polish(U, V, W, rank, n_steps=500, tol=1e-12):
        """High-precision polish pass using Newton-like iterations."""
        for step in range(n_steps):
            if time.time() > deadline - 3:
                break

            D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            err = D - T
            err_norm = np.max(np.abs(err))

            if err_norm < tol:
                break

            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', err, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', err, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', err, U, V, optimize=True)

            # Simple gradient descent with small step
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

        D = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return U, V, W, np.max(np.abs(D - T))

    # Try to find lower rank
    # Theoretical lower bound is 27
    ranks_to_try = list(range(53, 26, -1))

    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break

        found = False
        n_trials = 5
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                break

            seed = 1000 + rank * 100 + trial
            U_r, V_r, W_r, final_err = _optimize_rank(rank, seed, n_steps=1500, lr_init=0.05)

            if final_err < 1e-8:
                # Found a good solution, try to make it exact
                result = _try_fractional(U_r, V_r, W_r, rank)
                if result is not None:
                    best_rank = rank
                    best_U = np.array(result["U"])
                    best_V = np.array(result["V"])
                    best_W = np.array(result["W"])
                    found = True
                    break

            if time.time() > deadline - 5:
                break

        if found:
            # Continue to see if we can go lower
            continue

    # Final verification
    if time.time() < deadline - 2:
        D_final = np.einsum('ar,br,cr->abc', best_U, best_V, best_W, optimize=True)
        if np.max(np.abs(D_final - T)) > 1e-8:
            # Fallback to schoolbook
            best_U = U_s
            best_V = V_s
            best_W = W_s
            best_rank = rank_school

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
