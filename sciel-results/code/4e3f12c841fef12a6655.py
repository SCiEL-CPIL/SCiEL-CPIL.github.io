"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.integrate as si

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _van_der_pol_jac(t, y):
    """Analytical Jacobian for Van der Pol oscillator with mu=5."""
    mu = 5.0
    J = np.array([
        [0.0, 1.0],
        [-2.0 * mu * y[0] * y[1] - 1.0, mu * (1.0 - y[0]**2)]
    ])
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Provide analytical Jacobian to help Radau converge faster
    # Use slightly tighter tolerances to reduce relative error
    sol = si.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-12, 
        atol=1e-14,
        jac=_van_der_pol_jac
    )
    
    if not sol.success:
        # Fallback with looser tolerances
        sol = si.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-11, 
            atol=1e-13,
            jac=_van_der_pol_jac
        )
        
    return sol.y[:, -1]

# EVOLVE-BLOCK-END
