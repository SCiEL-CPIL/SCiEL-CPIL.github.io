import numpy as np
import time
import sys
from scipy.optimize import minimize

def coefficients(stages):
    """
    Compute 12-stage product formula coefficients for Heisenberg chain time evolution.
    Returns (a, b) where each is a list of 12 floats summing to 1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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

    def compute_error(a, b, t):
        """Compute spectral norm of U(t) - exp(-i t (A+B))"""
        U = np.eye(dim, dtype=complex)
        for k in range(len(a)):
            U = expm_B(b[k] * t) @ expm_A(a[k] * t) @ U
        U_exact = expm_AB(t)
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        return s[0]

    def objective(params):
        """
        params: 24-element array. First 12 are a, last 12 are b.
        """
        if time.time() > deadline:
            return 1e10

        a = params[:12]
        b = params[12:24]

        # Normalize to sum to 1
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if abs(a_sum) < 1e-15:
            a = np.ones(12) / 12
        else:
            a = a / a_sum
        if abs(b_sum) < 1e-15:
            b = np.ones(12) / 12
        else:
            b = b / b_sum

        total_err = 0.0
        for t in times:
            total_err += compute_error(a, b, t)
        return total_err / len(times)

    # Initial guess: Uniform
    x0 = np.ones(24) / 12
    best_x = x0.copy()
    best_obj = objective(x0)

    # Run L-BFGS-B optimization
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
    except Exception:
        pass

    # Random restarts with Powell if time permits
    np.random.seed(42)
    max_restarts = 3
    for _ in range(max_restarts):
        if time.time() > deadline - 30:
            break
        x_rand = best_x + 0.1 * np.random.randn(24)
        try:
            res = minimize(objective, x_rand, method='Powell', options={'maxiter': 100, 'maxfun': 200})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
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
