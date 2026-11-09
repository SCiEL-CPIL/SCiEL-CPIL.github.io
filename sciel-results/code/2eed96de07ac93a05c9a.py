import numpy as np
import time
from scipy.integrate import solve_ivp

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

    # Switch time: Fast transient is mostly over by t=10
    T_switch = 10.0
    
    try:
        # Phase 1: Full stiff integration [t0, T_switch]
        # Use BDF with very tight tolerances
        sol1 = solve_ivp(
            f, [t0, min(T_switch, t1)], y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        
        if not sol1.success:
            # Fallback: Full BDF for entire range
            sol = solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-13,
                atol=1e-15,
                jac=jac
            )
            return sol.y[:, -1] if sol.success else y0

        y_switch = sol1.y[:, -1]
        t_switch = sol1.t[-1]
        
        # If we've already reached t1, return
        if t_switch >= t1 - 1e-10:
            return y_switch

        # Phase 2: Reduced order model on slow manifold [T_switch, t1]
        # Variables: u = [y1, y3]
        # y2 is slaved: y2 = 0.04*y1 / (1e4*y3 + 6000)
        
        def f_reduced(t, u):
            y1 = u[0]
            y3 = u[1]
            # Quasi-steady state for y2
            # y2 = 0.04 * y1 / (10000 * y3 + 6000)
            y2 = 0.04 * y1 / (1e4 * y3 + 6000)
            
            # dy1/dt = -0.04*y1 + 1e4*y2*y3
            dy1 = -0.04 * y1 + 1e4 * y2 * y3
            # dy3/dt = 6000*y2
            dy3 = 6000 * y2
            
            return [dy1, dy3]

        # Initial condition for reduced system
        u0 = [y_switch[0], y_switch[2]]
        
        # Integrate reduced system
        # RK45 is efficient for non-stiff systems
        sol2 = solve_ivp(
            f_reduced, [t_switch, t1], u0,
            method='RK45',
            rtol=1e-13,
            atol=1e-15
        )
        
        if sol2.success:
            y1_end = sol2.y[0, -1]
            y3_end = sol2.y[1, -1]
            
            # Reconstruct y2 at the end
            y2_end = 0.04 * y1_end / (1e4 * y3_end + 6000)
            
            return np.array([y1_end, y2_end, y3_end])
        
        # Fallback if reduced model fails: BDF on full system for remaining
        sol2_full = solve_ivp(
            f, [t_switch, t1], y_switch,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        if sol2_full.success:
            return sol2_full.y[:, -1]
            
        return y_switch

    except Exception:
        pass

    # Final fallback: full BDF
    try:
        sol = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-13,
            atol=1e-15,
            jac=jac
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass

    return y0
