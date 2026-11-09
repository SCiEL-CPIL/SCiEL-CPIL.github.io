"""ODE solver: 15-dimensional stiff linear system using linear structure detection."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    
    Attempts to detect if the system is linear (y' = A(t)y + b(t)).
    If linear, provides the Jacobian to the solver for efficiency and accuracy.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    call_count = [0]
    
    def f_counted(t, y):
        call_count[0] += 1
        return f(t, y)

    # --- Phase 1: Linearity Detection ---
    # Check if f(t, y) is linear in y.
    # f(t, y) = A(t) @ y + b(t)
    
    # Sample b(t0)
    b_t0 = f_counted(t0, np.zeros(n))
    
    # Estimate A(t0) using basis vectors
    # We need n calls for A columns
    A_t0 = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        # f(t0, e_j) = A(t0)[:, j] + b(t0)
        # A(t0)[:, j] = f(t0, e_j) - b(t0)
        A_t0[:, j] = f_counted(t0, e_j) - b_t0
        
    # Check linearity at t0 with a random vector
    y_test = np.random.randn(n)
    f_test = f_counted(t0, y_test)
    f_pred = A_t0 @ y_test + b_t0
    err_t0 = np.linalg.norm(f_test - f_pred)
    
    # Check at another time point if possible
    t_mid = 0.5 * (t0 + t1)
    b_tmid = f_counted(t_mid, np.zeros(n))
    A_tmid = np.zeros((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A_tmid[:, j] = f_counted(t_mid, e_j) - b_tmid
        
    y_test2 = np.random.randn(n)
    f_test2 = f_counted(t_mid, y_test2)
    f_pred2 = A_tmid @ y_test2 + b_tmid
    err_tmid = np.linalg.norm(f_test2 - f_pred2)
    
    # If errors are small, assume linear
    is_linear = (err_t0 < 1e-10 * (1 + np.linalg.norm(f_test)) and 
                 err_tmid < 1e-10 * (1 + np.linalg.norm(f_test2)))
    
    if is_linear:
        # Check if A is constant
        const_A = np.linalg.norm(A_t0 - A_tmid) < 1e-10 * (1 + np.linalg.norm(A_t0))
        
        if const_A:
            # Use constant Jacobian
            def jac(t, y):
                return A_t0
            # We don't count jac calls in the contract usually, but f calls are counted.
            # solve_ivp with jac will use it.
            try:
                sol = solve_ivp(
                    f_counted,
                    [t0, t1],
                    y0,
                    method='BDF',
                    jac=jac,
                    rtol=1e-10,
                    atol=1e-12,
                    max_step=0.05,
                    first_step=1e-4
                )
                if sol.success:
                    return sol.y[:, -1]
            except Exception:
                pass
                
        else:
            # A(t) is variable. We can provide a jac that computes A(t) on the fly.
            # This is expensive (n+1 calls per jac eval), but might be worth it if stiff.
            # Alternatively, just use BDF without jac but with tight tolerances.
            # Let's try providing a dynamic jac.
            def jac_dynamic(t, y):
                # Compute A(t)
                b_t = f_counted(t, np.zeros(n))
                A_t = np.zeros((n, n))
                for j in range(n):
                    e_j = np.zeros(n)
                    e_j[j] = 1.0
                    A_t[:, j] = f_counted(t, e_j) - b_t
                return A_t

            try:
                sol = solve_ivp(
                    f_counted,
                    [t0, t1],
                    y0,
                    method='BDF',
                    jac=jac_dynamic,
                    rtol=1e-10,
                    atol=1e-12,
                    max_step=0.05,
                    first_step=1e-4
                )
                if sol.success:
                    return sol.y[:, -1]
            except Exception:
                pass

    # --- Phase 2: Fallback for non-linear or failed linear detection ---
    # Use Radau or BDF with adaptive Jacobian estimation (default behavior)
    # But tighter controls.
    try:
        sol = solve_ivp(
            f_counted,
            [t0, t1],
            y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
            first_step=1e-4
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    # Final fallback: Custom implicit Euler with linear structure if detected, else generic
    # If we detected linear but solvers failed, try manual linear BDF
    if is_linear and const_A:
        y = y0.copy()
        t = t0
        h = 1e-3
        # Pre-factorize (I - h A) is not possible for variable h, but we can solve
        while t < t1:
            h = min(h, t1 - t)
            # Implicit Euler: y_new = y + h * (A @ y_new + b(t+h))
            # y_new - h A y_new = y + h b(t+h)
            # (I - h A) y_new = y + h b(t+h)
            b_next = f_counted(t + h, np.zeros(n))
            rhs = y + h * b_next
            A_mat = np.eye(n) - h * A_t0
            try:
                y = solve(A_mat, rhs)
            except Exception:
                y = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            t += h
            # Increase step if stable
            h = min(h * 2, 0.1)
        return y

    # Generic fallback
    y = y0.copy()
    t = t0
    h = 1e-4
    while t < t1:
        h = min(h, t1 - t)
        y_new = y.copy()
        for _ in range(50):
            fz = f_counted(t + h, y_new)
            # FD Jacobian
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1e-8, 1e-8 * abs(y_new[j]))
                e[j] = scale
                J[:, j] = (f_counted(t + h, y_new + e) - fz) / scale
            A_mat = np.eye(n) - h * J
            rhs = -(y_new - y - h * fz)
            try:
                delta = solve(A_mat, rhs)
            except Exception:
                delta = np.linalg.lstsq(A_mat, rhs, rcond=None)[0]
            y_new = y_new + delta
            if np.linalg.norm(delta) < 1e-12:
                break
        y = y_new
        t += h
        if not np.all(np.isfinite(y)):
            h = max(h * 0.5, 1e-8)
    return y
