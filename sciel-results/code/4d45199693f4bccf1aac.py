import numpy as np
import time
import scipy.linalg

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = y0.shape[0]
    dt = t1 - t0

    if dt == 0.0:
        return y0.copy()

    start_time = time.time()
    deadline = start_time + 285.0

    # Probe the affine linear map: f(t, y) = A @ y + b
    # For a linear autonomous system, b = 0 and A is constant.
    # Evaluate at t0 to get initial structure.
    b = np.asarray(f(t0, np.zeros(n)), dtype=float)

    # Probe A columns: f(t0, e_i) - b = A e_i = column i of A
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        col = np.asarray(f(t0, ei), dtype=float) - b
        A[:, i] = col

    # Check time remaining
    if time.time() > deadline:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # Compute matrix exponential of A * dt using high-precision method
    E = scipy.linalg.expm(A * dt)

    # Check if b is effectively zero
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        y_end = E @ y0
    else:
        # Use block matrix exponential for affine system
        # Block = [[A, b], [0, 1]]  (size (n+1) x (n+1))
        block = np.zeros((n + 1, n + 1))
        block[:n, :n] = A
        block[:n, n] = b
        block[n, n] = 1.0

        block_exp = scipy.linalg.expm(block * dt)
        y_end = block_exp[:n, :n] @ y0 + block_exp[:n, n]

    # Optional: verify linearity by checking if f(t1, y) matches A@y + b
    # If not, the system is time-varying and we need a different approach
    # But for the given task (fixed 15-dim stiff linear system), assume autonomous
    # We could add a quick check here if time permits
    
    if time.time() < deadline - 5.0:
        # Quick verification: check if the system is truly autonomous
        # by evaluating f at t1 with a test vector and comparing
        test_y = np.random.randn(n)
        f_t1 = np.asarray(f(t1, test_y), dtype=float)
        expected = A @ test_y + b
        diff = np.linalg.norm(f_t1 - expected)
        scale = np.linalg.norm(expected) + 1.0
        if diff / scale > 1e-8:
            # System is time-varying; fall back to numerical integration
            from scipy.integrate import solve_ivp
            sol = solve_ivp(f, t_span, y0, method='Radau', rtol=1e-12, atol=1e-14)
            return sol.y[:, -1]

    return y_end
