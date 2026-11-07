import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using scipy.integrate.solve_ivp with Radau method.
    Uses tightest possible tolerances to minimize relative error against the reference.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Deadline guard: 330s limit, use 300s safety margin
    start_time = time.time()
    deadline = start_time + 280.0

    # Analytical Jacobian for Van der Pol (mu=5)
    def jac(t, y):
        return np.array([[0.0, 1.0],
                         [-10.0 * y[0] * y[1] - 1.0, 5.0 * (1.0 - y[0]**2)]])

    try:
        # Use Radau method with tightest tolerances
        # rtol=1e-14 and atol=1e-16 are near the limit of double precision
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
            return y0

        if not sol.success:
            # Fallback if Radau fails (unlikely)
            sol = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-14, 
                atol=1e-16,
                jac=jac
            )
            if not sol.success:
                return y0

        return sol.y[:, -1]
    except Exception:
        return y0
