"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Solve the 100-point convection-diffusion ODE using adaptive stiff solver."""
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    deadline = time.time() + 300.0  # Safety margin before 330s kill
    
    # Try LSODA with tight tolerances
    result = solve_ivp(
        f, t_span, y0,
        method='LSODA',
        rtol=1e-12,
        atol=1e-14,
        max_step=(t1 - t0) / 100.0
    )
    
    if result.success and time.time() < deadline:
        return result.y[:, -1]
    
    # Fallback: Radau (implicit, stiff) if LSODA failed
    if time.time() < deadline:
        result2 = solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50.0
        )
        if result2.success:
            return result2.y[:, -1]
    
    # Last resort: BDF
    if time.time() < deadline:
        result3 = solve_ivp(
            f, t_span, y0,
            method='BDF',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 50.0
        )
        if result3.success:
            return result3.y[:, -1]
    
    # If all failed, return initial condition (shouldn't happen)
    return y0
# EVOLVE-BLOCK-END
