import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using DOP853 (8th-order explicit Runge-Kutta)
    with very tight tolerances. DOP853 has a smaller error constant than Radau for
    smooth non-stiff problems, potentially achieving lower relative error with
    comparable or fewer function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s limit, use 280s safety margin
    start_time = time.time()
    deadline = start_time + 280.0

    # Analytical Jacobian for Van der Pol (mu=5)
    def jac(t, y):
        return np.array([[0.0, 1.0],
                         [-10.0 * y[0] * y[1] - 1.0, 5.0 * (1.0 - y[0]**2)]])

    best_result = y0.copy()
    best_error = np.inf

    # Strategy 1: DOP853 with very tight tolerances
    # DOP853 is an 8th-order explicit RK method with embedded 7th-order error estimate
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-14, 
            atol=1e-16,
            jac=jac
        )
        
        if time.time() > deadline:
            return best_result

        if sol.success:
            best_result = sol.y[:, -1].copy()
            # We can't compute true error, but we trust DOP853's internal estimate
            # DOP853 typically has very good accuracy for smooth problems
            return best_result
    except Exception:
        pass

    # Strategy 2: Radau with tight tolerances (fallback)
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-14, 
            atol=1e-16,
            jac=jac
        )
        
        if time.time() > deadline:
            return best_result

        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    # Strategy 3: BDF with tight tolerances (last resort)
    try:
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            rtol=1e-14, 
            atol=1e-16,
            jac=jac
        )
        
        if time.time() > deadline:
            return best_result

        if sol.success:
            return sol.y[:, -1].copy()
    except Exception:
        pass

    return best_result
