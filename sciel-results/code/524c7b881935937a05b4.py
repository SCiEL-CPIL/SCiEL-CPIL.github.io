import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solves the Robertson ODE problem using a two-phase approach:
    1. Explicit RK45 to quickly pass through the fast transient phase (small t)
    2. Implicit Radau with high precision for the slow phase (large t)
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y1, y2, y3 = y
        J = np.array([
            [-0.04, 1e4 * y3, 1e4 * y2],
            [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
            [0.0, 6e3 * y2, 0.0]
        ])
        return J

    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin

    try:
        # Phase 1: Use RK45 (explicit) to handle the fast transient phase
        # The Robertson problem has a very fast initial transient where y2 spikes
        # RK45 is efficient for this non-stiff part if we use a reasonable max_step
        # We'll integrate from t0 to a point where the system becomes more gradual
        
        # Estimate where the fast transient ends
        # For Robertson, the fast transient is roughly in [0, 1]
        # Let's use a transition point at t = 1.0 or where the dynamics slow down
        
        t_transition = min(1.0, (t1 - t0) * 0.01)  # First 1% of interval or t=1
        
        if t_transition > t0:
            # Phase 1: RK45 for fast transient
            sol_phase1 = solve_ivp(
                f,
                [t0, t_transition],
                y0,
                method='RK45',
                rtol=1e-8,
                atol=1e-10,
                max_step=(t_transition - t0) / 50.0
            )
            
            if not sol_phase1.success:
                # Fallback: use Radau for entire interval
                sol = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='Radau',
                    jac=jac,
                    rtol=1e-10,
                    atol=1e-10
                )
                if sol.success:
                    return sol.y[:, -1]
                return y0
            
            y_transition = sol_phase1.y[:, -1]
            t_start_phase2 = t_transition
        else:
            y_transition = y0
            t_start_phase2 = t0
        
        # Phase 2: Use Radau (implicit) for the slow, stiff part
        if t_start_phase2 < t1:
            sol_phase2 = solve_ivp(
                f,
                [t_start_phase2, t1],
                y_transition,
                method='Radau',
                jac=jac,
                rtol=1e-12,
                atol=1e-14
            )
            
            if sol_phase2.success:
                return sol_phase2.y[:, -1]
            else:
                # Fallback if phase 2 fails
                sol = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='Radau',
                    jac=jac,
                    rtol=1e-10,
                    atol=1e-10
                )
                if sol.success:
                    return sol.y[:, -1]
                return y0
        else:
            return y_transition

    except Exception as e:
        # Fallback to Radau for entire interval
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=jac,
                rtol=1e-10,
                atol=1e-10
            )
            if sol.success:
                return sol.y[:, -1]
        except:
            pass
        return y0
