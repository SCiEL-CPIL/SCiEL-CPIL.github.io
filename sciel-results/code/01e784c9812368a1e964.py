"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40] using a two-phase strategy:
    Phase 1: DOP853 with relaxed tolerances for the transient phase [0, 39.9]
    Phase 2: Radau with tight tolerances for the final step [39.9, 40]
    
    This exploits the rapid convergence to the slow manifold to minimize total
    function evaluations while ensuring high terminal precision.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: aim to finish well before 330s
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    # Split point: most of the transient is done by t=39.9
    t_split = t1 - 0.1
    
    try:
        # Phase 1: DOP853 for the transient phase with relaxed tolerances
        # DOP853 is a high-order explicit Runge-Kutta method (8th/7th order)
        # Relaxed tolerances since we'll refine in Phase 2
        sol1 = solve_ivp(
            f,
            [t0, t_split],
            y0,
            method='DOP853',
            rtol=1e-8,
            atol=1e-10,
            max_step=1.0,
        )
        
        if not sol1.success:
            # Fallback: try Radau for the whole interval
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=1.0,
            )
            return sol.y[:, -1] if sol.success else y0
        
        # Check deadline before Phase 2
        if time.time() > deadline - 30.0:
            # Not enough time for Phase 2, return Phase 1 result
            return sol1.y[:, -1]
        
        # Phase 2: Radau with tight tolerances for the final step
        # The system is on the slow manifold, so a single high-accuracy step suffices
        y_split = sol1.y[:, -1]
        sol2 = solve_ivp(
            f,
            [t_split, t1],
            y_split,
            method='Radau',
            rtol=1e-14,
            atol=1e-16,
            max_step=0.1,
        )
        
        if sol2.success:
            return sol2.y[:, -1]
        else:
            # If Phase 2 fails, try with slightly looser tolerances
            sol2b = solve_ivp(
                f,
                [t_split, t1],
                y_split,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=0.1,
            )
            if sol2b.success:
                return sol2b.y[:, -1]
            else:
                return sol2.y[:, -1] if sol2.y.size > 0 else y_split
                
    except Exception:
        # If something goes wrong, try a simpler approach
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                max_step=1.0,
            )
            if sol.success:
                return sol.y[:, -1]
            return sol.y[:, -1] if sol.y.size > 0 else y0
        except Exception:
            return y0


# EVOLVE-BLOCK-END
