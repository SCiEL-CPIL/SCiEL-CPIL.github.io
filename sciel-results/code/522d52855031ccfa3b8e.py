"""Rewritten ODE integrator for the fixed Lorenz IVP using scipy's DOP853."""
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use scipy's optimized DOP853 implementation
    # This is the same method used to generate the reference solution
    # rtol and atol are set to match or exceed the reference tolerances
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='DOP853',
        rtol=1e-12,
        atol=1e-14,
        dense_output=False
    )
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback: if solve_ivp fails, try with slightly relaxed tolerances
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            dense_output=False
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: return initial condition or raise
            raise RuntimeError("Integration failed")
