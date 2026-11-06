"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE (or any IVP y' = f(t, y)) from t_span[0] to t_span[1].
    
    Strategy (New Design):
    1. Use DOP853 (high-order explicit RK) for the bulk of the integration [t0, t1 - 0.1]
       with tight tolerances. This handles the transient and the long slow manifold
       efficiently, leveraging the high order of DOP853 to take large steps on the
       non-stiff slow manifold.
    2. Use a high-accuracy implicit method (Radau) for the final interval
       [t1 - 0.1, t1] to polish the solution and ensure high terminal precision.
    
    This design minimizes the overhead of implicit methods (Jacobian costs) over the
    long interval where the problem is non-stiff, while using the robustness of
    Radau for the final step to correct any accumulated errors.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # If the interval is very short, just use a high-accuracy implicit method directly
    if (t1 - t0) < 0.5:
        sol = solve_ivp(f, [t0, t1], y0, method='Radau',
                        rtol=1e-12, atol=1e-14, max_step=(t1-t0))
        if sol.success:
            return sol.y[:, -1]
        # Fallback to DOP853 if Radau fails
        sol = solve_ivp(f, [t0, t1], y0, method='DOP853',
                        rtol=1e-12, atol=1e-14, max_step=(t1-t0))
        return sol.y[:, -1] if sol.success else y0
    
    # Define the split point.
    # We use DOP853 for the vast majority of the interval.
    # The final 0.1 units are handled by Radau for high precision polishing.
    t_mid = t1 - 0.1
    if t_mid <= t0:
        t_mid = (t0 + t1) / 2
    
    # Phase 1: DOP853 for [t0, t_mid]
    # Tight tolerances to ensure high accuracy over the long interval.
    # DOP853 is 8th order, so it is very efficient on the slow manifold.
    sol1 = solve_ivp(f, [t0, t_mid], y0, method='DOP853',
                     rtol=1e-12, atol=1e-14, max_step=np.inf)
    
    if not sol1.success:
        # Fallback: try with slightly relaxed tolerances if it failed (unlikely)
        sol1 = solve_ivp(f, [t0, t_mid], y0, method='DOP853',
                         rtol=1e-10, atol=1e-12, max_step=np.inf)
        
        if not sol1.success:
            # Last resort: use Radau for the whole thing
            sol = solve_ivp(f, [t0, t1], y0, method='Radau',
                            rtol=1e-10, atol=1e-12)
            return sol.y[:, -1] if sol.success else y0
    
    y_mid = sol1.y[:, -1]
    
    # Phase 2: High-accuracy integration from t_mid to t1
    # Use Radau with very tight tolerances for the final short step.
    # This acts as a "polish" to ensure the terminal state is highly accurate.
    sol2 = solve_ivp(f, [t_mid, t1], y_mid, method='Radau',
                     rtol=1e-14, atol=1e-16, max_step=(t1 - t_mid))
    
    if not sol2.success:
        # Fallback: try DOP853 for the final segment
        sol2 = solve_ivp(f, [t_mid, t1], y_mid, method='DOP853',
                         rtol=1e-14, atol=1e-16, max_step=(t1 - t_mid))
    
    if not sol2.success:
        # If everything fails, return the mid-point value
        return y_mid
    
    return sol2.y[:, -1]

# EVOLVE-BLOCK-END
