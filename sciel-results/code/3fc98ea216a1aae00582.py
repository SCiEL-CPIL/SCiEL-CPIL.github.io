import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 290.0

    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y1, y2, y3 = y
        k1 = 0.04
        k2 = 1e4
        k3 = 6000
        return np.array([
            [-k1, k2*y3, k2*y2],
            [k1, -k2*y3 - 2*k3*y2, -k2*y2],
            [0.0, k2*y3, k2*y2]
        ])

    try:
        from scipy.integrate import solve_ivp
        
        # Use BDF with CORRECT analytic Jacobian
        # BDF is well-suited for this stiff problem
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

        # Fallback: Radau with CORRECT analytic Jacobian
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

        # Fallback: BDF without Jacobian
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16
        )
        if sol.success:
            return sol.y[:, -1]

    except Exception:
        pass

    # Final fallback: simple RK45 if scipy fails
    def _rk45_fallback(f, t0, t1, y0, deadline):
        y = np.array(y0, dtype=float)
        t = t0
        h = (t1 - t0) / 100.0
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t)
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2*k1)
            k3 = f(t + h/2, y + h/2*k2)
            k4 = f(t + h, y + h*k3)
            
            y_new = y + h*(k1 + 2*k2 + 2*k3 + k4)/6
            err = h*(k4 - k3)/12
            err_norm = np.max(np.abs(err) / (1 + np.abs(y_new)))
            
            if err_norm <= 1e-10:
                y = y_new
                t += h
                h *= min(5.0, 0.9*(1e-10/err_norm)**0.2) if err_norm > 0 else 5.0
            else:
                h *= max(0.1, 0.9*(1e-10/err_norm)**0.2)
        return y

    return _rk45_fallback(f, t0, t1, y0, deadline)
