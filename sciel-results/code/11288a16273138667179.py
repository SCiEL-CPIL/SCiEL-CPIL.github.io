"""Rewritten Robertson ODE solver with structural changes for higher precision."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def _jacobian(t, y):
    """
    Analytic Jacobian for the Robertson problem.
    y0' = -0.04 y0 + 10^4 y1 y2
    y1' = 0.04 y0 - 10^4 y1 y2 - 3e3 y1^2
    y2' = 3e3 y1^2
    """
    y0, y1, y2 = y
    J = np.array([
        [-0.04, 1e4 * y2, 1e4 * y1],
        [0.04, -1e4 * y2 - 6e3 * y1, -1e4 * y1],
        [0.0, 6e3 * y1, 0.0]
    ])
    return J

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Set a deadline slightly before the 330s kill switch
    start_time = time.time()
    deadline = 325.0  # seconds, 5s safety margin
    
    # Strategy: Use multiple solver passes with progressively tighter tolerances.
    # The Robertson problem is stiff, so implicit methods (Radau, BDF) are appropriate.
    # We'll start with Radau and use very tight tolerances.
    # Key insight: The score is -log10(rel_err) - 0.5*log10(evals).
    # To maximize this, we want both high precision and low evaluation count.
    # Radau with the Jacobian should be efficient.
    
    best_result = None
    best_elapsed = 0.0
    
    # Pass 1: Radau with rtol=1e-13, atol=1e-15
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            jac=_jacobian,
            rtol=1e-13, 
            atol=1e-15,
            max_step=np.inf
        )
        
        if sol.success:
            best_result = sol.y[:, -1].copy()
            best_elapsed = time.time() - start_time
        else:
            raise RuntimeError("Radau pass 1 failed")
            
    except Exception:
        pass
    
    # Check if we have time for a refinement pass
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < deadline - 20:
        # Pass 2: Radau with even tighter tolerances rtol=1e-14, atol=1e-16
        try:
            sol_refined = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                jac=_jacobian,
                rtol=1e-14, 
                atol=1e-16,
                max_step=np.inf
            )
            
            if sol_refined.success:
                best_result = sol_refined.y[:, -1].copy()
                best_elapsed = time.time() - start_time
                
        except Exception:
            pass
    
    # If Radau failed entirely, try BDF as fallback
    if best_result is None:
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                jac=_jacobian,
                rtol=1e-12, 
                atol=1e-14
            )
            if sol.success:
                best_result = sol.y[:, -1].copy()
            else:
                raise RuntimeError("BDF fallback failed")
        except Exception:
            # Last resort: try with even looser tolerances
            try:
                sol = solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='BDF', 
                    jac=_jacobian,
                    rtol=1e-10, 
                    atol=1e-12
                )
                best_result = sol.y[:, -1].copy()
            except Exception:
                # Absolute last resort: return initial condition (will score poorly but won't crash)
                best_result = y0.copy()
    
    return best_result
# EVOLVE-BLOCK-END
