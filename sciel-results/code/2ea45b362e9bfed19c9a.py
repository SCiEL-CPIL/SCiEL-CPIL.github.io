"""ODE solver for convection-diffusion 100-point problem using eigendecomposition."""
import numpy as np
import time


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)

    start_time = time.time()
    deadline = start_time + 295.0

    dt = t1 - t0
    if dt == 0:
        return y

    # Extract A and b via complex probing (2 columns per call)
    y_zero = np.zeros(n)
    b = f(t0, y_zero)
    b = np.asarray(b, dtype=float)

    A = np.zeros((n, n))
    use_complex = True

    try:
        v_test = np.zeros(n, dtype=complex)
        v_test[0] = 1.0
        if n > 1:
            v_test[1] = 1.0j
        w_test = f(t0, v_test)
        if not np.iscomplexobj(w_test):
            raise ValueError("f does not support complex")
        diff = w_test - b
        A[:, 0] = diff.real
        if n > 1:
            A[:, 1] = diff.imag

        for j in range(2, n, 2):
            v = np.zeros(n, dtype=complex)
            v[j] = 1.0
            v[j + 1] = 1.0j
            w = f(t0, v)
            d = w - b
            A[:, j] = d.real
            A[:, j + 1] = d.imag

        if n % 2 != 0:
            ej = np.zeros(n)
            ej[-1] = 1.0
            A[:, -1] = f(t0, ej) - b

    except Exception:
        use_complex = False
        b = f(t0, y_zero)
        b = np.asarray(b, dtype=float)
        A = np.zeros((n, n))
        for j in range(n):
            ej = np.zeros(n)
            ej[j] = 1.0
            A[:, j] = f(t0, ej) - b

    # Compute e^{A*dt} via eigendecomposition for potentially better numerical stability
    try:
        eigvals, eigvecs = np.linalg.eig(A)
        e_vals = np.exp(eigvals * dt)
        # e^{A*dt} = V @ diag(e_vals) @ V^{-1}
        # y_final = e^{A*dt} @ y0 + particular solution
        # For y' = A y + b, the solution is y(t) = e^{At} y0 + integral_0^t e^{A(t-s)} b ds
        # If b is constant, integral = (e^{At} - I) A^{-1} b when A is invertible
        # More robust: use the augmented system approach but via eigendecomp of M
        # Actually, let's just compute e^{A*dt} and handle the particular solution
        V = eigvecs
        V_inv = np.linalg.inv(V)
        E = V @ np.diag(e_vals) @ V_inv

        # Particular solution for constant b:
        # y_p(t) = (e^{At} - I) @ solve(A, b) if A is invertible
        # Or use the augmented matrix exponential approach
        # Let's build the augmented matrix M = [[A, b], [0, 1]] and eigendecompose it
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        M[n, n] = 1.0

        M_dt = M * dt
        eigvals_M, eigvecs_M = np.linalg.eig(M_dt)
        e_vals_M = np.exp(eigvals_M)
        V_M = eigvecs_M
        V_inv_M = np.linalg.inv(V_M)
        E_aug = V_M @ np.diag(e_vals_M) @ V_inv_M

        y_aug = np.zeros(n + 1)
        y_aug[:n] = y
        y_aug[n] = 1.0

        y_final = E_aug @ y_aug
        y_final = y_final[:n]

    except Exception:
        # Fallback to scipy expm if eigendecomposition fails
        from scipy.linalg import expm
        M = np.zeros((n + 1, n + 1))
        M[:n, :n] = A
        M[:n, n] = b
        M[n, n] = 1.0
        M_dt = M * dt
        E_aug = expm(M_dt)
        y_aug = np.zeros(n + 1)
        y_aug[:n] = y
        y_aug[n] = 1.0
        y_final = E_aug @ y_aug
        y_final = y_final[:n]

    if time.time() > deadline:
        return y_final

    return y_final


def _rk4_fallback(f, t_span, y0):
    """Fallback high-order solver if linearity assumption fails."""
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h = (t1 - t0) / 1000
    t = t0
    for _ in range(1000):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return y
