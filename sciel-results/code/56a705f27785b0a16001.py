import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    start_time = time.time()
    deadline = start_time + 290.0

    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        return np.array([
            [-0.04, 1e4*y[2], 1e4*y[1]],
            [0.04, -1e4*y[2] - 6000*y[1], -1e4*y[1]],
            [0.0, 6000*y[1], 0.0]
        ])

    # Phase 1: Explicit RK45 with adaptive step on the original system
    # This handles the fast transient efficiently with large steps once it settles
    def _rk45_explicit(f, t0, t1, y0, deadline, rtol=1e-10, atol=1e-12):
        y = np.array(y0, dtype=float)
        t = t0
        h = 1e-4  # Start with small step for fast transient
        max_h = (t1 - t0) * 0.01
        min_h = 1e-12
        
        while t < t1 and time.time() < deadline:
            h = min(h, t1 - t, max_h)
            if h < min_h:
                h = min_h
            
            # RK45 (Dormand-Prince) coefficients
            a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0
            b2, b3, b4, b5, b6 = 1/5, 3/40, 44/45, 19372/6561, 9/2
            c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
            d2, d3, d4, d5, d6 = 0, 9/40, 32/9, 3392/27, 285/26
            e2, e3, e4, e5, e6 = 0, 9/40, 32/9, 3392/27, 285/26
            # 5th order weights
            w2, w3, w4, w5, w6 = 19372/6561, -25060/27783, 645/632, -212/729, 1/9
            # 4th order weights
            v2, v3, v4, v5, v6 = 35/384, 0, 500/1113, 125/192, -2187/6784
            
            k1 = f(t, y)
            k2 = f(t + a2*h, y + a2*h*k1)
            k3 = f(t + a3*h, y + a3*h*(b2*k1 + b3*k2))
            k4 = f(t + a4*h, y + a4*h*(b3*k1 + b4*k2 + b5*k3))
            k5 = f(t + a5*h, y + a5*h*(b4*k1 + b5*k2 + b6*k3))
            k6 = f(t + a6*h, y + a6*h*(b5*k1 + b6*k2 + b7*k3 + b8*k4 + b9*k5))
            
            # 5th order solution
            y_new = y + h*(w1*k1 + w2*k2 + w3*k3 + w4*k4 + w5*k5 + w6*k6)
            # 4th order solution for error estimation
            y_4th = y + h*(v1*k1 + v2*k2 + v3*k3 + v4*k4 + v5*k5 + v6*k6)
            
            err = np.max(np.abs(y_new - y_4th) / (atol + rtol*np.maximum(np.abs(y), np.abs(y_new))))
            
            if err <= 1.0:
                y = y_new
                t += h
                # Increase step size
                if err > 0:
                    h *= min(5.0, 0.9 * err**(-0.2))
                else:
                    h *= 5.0
                h = min(h, max_h)
            else:
                # Decrease step size
                h *= max(0.1, 0.9 * err**(-0.25))
        
        return y, t

    # Phase 2: Use BDF with tight tolerances for the remaining interval
    def _bdf_phase(f, t0, t1, y0, deadline, rtol=1e-13, atol=1e-15):
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f, [t0, t1], y0,
                method='BDF',
                rtol=rtol,
                atol=atol,
                jac=jac
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
        return y0

    try:
        from scipy.integrate import solve_ivp
        
        # Strategy: Use BDF with very tight tolerances directly
        # BDF is 5th order and handles stiffness well with analytic Jacobian
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

        # Fallback: Radau with analytic Jacobian
        sol = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]

    except Exception:
        pass

    # Final fallback: RK45
    y_final, _ = _rk45_explicit(f, t0, t1, y0, deadline)
    return y_final
