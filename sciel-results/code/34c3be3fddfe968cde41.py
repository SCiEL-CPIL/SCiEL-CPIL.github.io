import numpy as np
import time
import sys

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    # Build the Hamiltonians
    n = 6
    dim = 2 ** n
    delta = 0.8
    h = 0.3

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    Jz = 1.0
    Jxy = delta * Jz

    def add_bond(H, i, j):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)

        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)

    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)

    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)

    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term

    # Precompute eigendecompositions for A, B, and A+B
    vals_A, vecs_A = np.linalg.eigh(A)
    vals_B, vecs_B = np.linalg.eigh(B)
    vals_AB, vecs_AB = np.linalg.eigh(A + B)

    times = [0.25, 0.5, 1.0]

    def expm_A(t):
        """exp(-i * t * A)"""
        return vecs_A @ np.diag(np.exp(-1j * t * vals_A)) @ vecs_A.conj().T

    def expm_B(t):
        """exp(-i * t * B)"""
        return vecs_B @ np.diag(np.exp(-1j * t * vals_B)) @ vecs_B.conj().T

    def expm_AB(t):
        """exp(-i * t * (A+B))"""
        return vecs_AB @ np.diag(np.exp(-1j * t * vals_AB)) @ vecs_AB.conj().T

    # Precompute exact propagators
    U_exact = {}
    for t in times:
        U_exact[t] = expm_AB(t)

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        diff = U - U_exact[t]
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a_full = params[:12]
        b_full = params[12:24]

        # Normalize to sum to 1
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-15:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-15:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum

        # Check bounds
        if np.any(a_full < -50) or np.any(a_full > 50):
            return 1e10
        if np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10

        total_err = 0.0
        for t in times:
            total_err += compute_error(a_full, b_full, t)
        return total_err / len(times)

    # Initialize with equal weights
    x0 = np.ones(24) / 12.0
    best_x = x0.copy()
    best_obj = objective(x0)

    # Gradient Descent with Momentum using Central Difference Gradients
    def gradient(params, h=1e-5):
        g = np.zeros_like(params)
        for i in range(len(params)):
            pp = params.copy()
            pp[i] += h
            pm = params.copy()
            pm[i] -= h
            g[i] = (objective(pp) - objective(pm)) / (2 * h)
        return g

    def gradient_descent(x_start, max_iter=200):
        x = x_start.copy()
        fx = objective(x)
        v = np.zeros_like(x)
        mu = 0.9
        lr = 0.01
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            g = gradient(x)
            v = mu * v + lr * g
            x_new = x - v
            fx_new = objective(x_new)
            if fx_new < fx:
                x = x_new
                fx = fx_new
                lr *= 0.99
            else:
                lr *= 0.5
                v = mu * v + lr * g
                x_new = x - v
                fx_new = objective(x_new)
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                else:
                    break
        return x, fx

    np.random.seed(42)

    # Run Gradient Descent from uniform
    try:
        res_x, res_f = gradient_descent(x0, max_iter=300)
        if res_f < best_obj:
            best_obj = res_f
            best_x = res_x
    except Exception:
        pass

    # Try a few random restarts
    max_restarts = 10
    for _ in range(max_restarts):
        if time.time() > deadline - 10:
            break
        x_rand = best_x + 0.05 * np.random.randn(24)
        try:
            res_x, res_f = gradient_descent(x_rand, max_iter=100)
            if res_f < best_obj:
                best_obj = res_f
                best_x = res_x
        except Exception:
            pass

    # Extract final coefficients
    a_full = best_x[:12]
    b_full = best_x[12:24]

    # Clip and renormalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum

    return [float(x) for x in a_full], [float(x) for x in b_full]
