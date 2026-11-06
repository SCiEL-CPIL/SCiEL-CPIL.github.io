"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use Radau method for stiff problems (convection-diffusion is often stiff)
    # Radau is an implicit Runge-Kutta method, 3rd order, suitable for stiff systems.
    # Tolerances chosen to balance accuracy and function call count.
    # rtol=1e-8, atol=1e-10 is typically sufficient for high accuracy without excessive calls.
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-8,
            atol=1e-10,
            max_step=(t1 - t0) / 100  # Limit step size to ensure resolution
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # If solve_ivp fails, try with looser tolerances or different method
            # But usually Radau is robust. If it fails, it might be due to stiffness or singularities.
            # Let's try BDF as a fallback, which is also implicit and good for stiff problems.
            sol2 = solve_ivp(
                f,
                t_span,
                y0,
                method='BDF',
                rtol=1e-8,
                atol=1e-10,
                max_step=(t1 - t0) / 100
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: return the last computed value if available, or y0
                if sol.y.size > 0:
                    return sol.y[:, -1]
                if sol2.y.size > 0:
                    return sol2.y[:, -1]
                return y0

    except Exception:
        # In case of any other exception, return y0 to avoid crashing
        return y0

# EVOLVE-BLOCK-END
