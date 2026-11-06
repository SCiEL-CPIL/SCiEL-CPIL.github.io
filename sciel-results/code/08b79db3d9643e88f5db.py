"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    """
    Integrates the Robertson ODE system from t_span[0] to t_span[1].
    Returns the final state y(t_end).
    
    Strategy: Hybrid multi-stage integration.
    The Robertson problem has three distinct timescales:
    1. Fast transient: t in [0, ~0.1] where y0 drops and y1 spikes.
    2. Slow decay: t in [~0.1, ~100] where y1 decays slowly.
    3. Equilibrium: t in [~100, 40] (actually t goes to 40, so this is just the tail).
    
    Wait, t_span is [0, 40].
    Phase 1: t=0 to t=0.1 (Stiff, fast changes)
    Phase 2: t=0.1 to t=40 (Stiff, slow changes)
    
    Actually, for Robertson, the stiffness is high throughout, but the fast transient
    is very localized. Using an implicit method like BDF or Radau with an analytic
    Jacobian is standard.
    
    The key to improving the score is reducing the number of function evaluations (calls to f)
    while maintaining high accuracy.
    
    The current program uses Radau. BDF is often more efficient for very stiff problems
    because it has better stability properties for extremely stiff systems and can take
    larger steps. LSODA is also a candidate.
    
    Let's try BDF with an analytic Jacobian.
    Also, we can try to be smarter about tolerances.
    
    Score = -log10(relative error) - 0.5 * log10(calls)
    
    If we reduce calls by a factor of 10, we gain 0.5 * log10(10) = 0.5 points.
    If we maintain the same error, we gain 0.5 points.
    
    Let's use BDF.
    """
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill

    # Robertson system analytic Jacobian
    # y' = f(t, y)
    # y0' = -0.04*y0 + 1e4*y1*y2
    # y1' =  0.04*y0 - 1e4*y1*y2 - 3e7*y1^2
    # y2' =  3e7*y1^2
    def jac(t, y):
        y0_val, y1_val, y2_val = y
        return np.array([
            [-0.04, 1e4*y2_val, -1e4*y1_val],
            [0.04, -1e4*y2_val - 6e7*y1_val, 1e4*y1_val],
            [0, 6e7*y1_val, 0]
        ])

    try:
        if time.time() > deadline:
            return np.array(y0)

        # Try BDF first, as it is often more efficient for stiff problems than Radau
        # in terms of function evaluations per step.
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='BDF', 
            jac=jac,
            rtol=1e-10, 
            atol=1e-12
        )
        
        if sol.success:
            return sol.y[:, -1]
        else:
            # Fallback to Radau if BDF fails
            if time.time() > deadline:
                return np.array(y0)
            
            sol2 = scipy.integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau', 
                jac=jac,
                rtol=1e-10, 
                atol=1e-12
            )
            
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: LSODA
                if time.time() > deadline:
                    return np.array(y0)
                
                sol3 = scipy.integrate.solve_ivp(
                    f, 
                    t_span, 
                    y0, 
                    method='LSODA', 
                    jac=jac,
                    rtol=1e-10, 
                    atol=1e-12
                )
                
                if sol3.success:
                    return sol3.y[:, -1]
                return np.array(y0)
                
    except Exception:
        # In case of any unexpected error
        return np.array(y0)

# EVOLVE-BLOCK-END
