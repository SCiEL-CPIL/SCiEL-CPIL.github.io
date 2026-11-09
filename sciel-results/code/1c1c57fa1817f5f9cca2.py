"""ODE solver for convection-diffusion 100-point problem using matrix exponential strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg
import scipy.integrate
import time

def _compute_jacobian(f, t, y, eps=1e-7):
    """Compute Jacobian using forward differences."""
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(t, y)
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps * max(1.0, abs(y[j]))
        J[:, j] = (f(t, y + e) - f0) / e[j]
    return J

def _expm_multiply(M, x):
    """Compute exp(M) @ x efficiently."""
    # scipy.linalg.expm_multiply is more efficient than full expm for single vector
    return scipy.linalg.expm_multiply(M, x)

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # 1. Compute Jacobian at t0
    J = _compute_jacobian(f, t0, y)
    
    # Strategy: Try exponential stepping first
    # If the system is linear or nearly linear, this is very efficient.
    # We use a fixed number of steps initially, then check accuracy.
    
    def try_exponential_solve():
        nonlocal y, t0, t1, J
        # Start with a moderate number of steps
        steps = 20
        h = (t1 - t0) / steps
        t = t0
        y_curr = y.copy()
        
        # Pre-compute expm(J*h) if J is constant
        # Note: J might change if we update it, but we assume constant for this phase
        exp_J_h = scipy.linalg.expm(J * h)
        
        for _ in range(steps):
            # Check deadline
            if time.time() > deadline:
                return None
            
            y_next = exp_J_h @ y_curr
            t += h
            
            # Check residual to ensure linearity assumption holds
            # Residual = f(t, y_next) - J @ y_next
            # If small, the linear approximation is good
            f_val = f(t, y_next)
            residual = np.max(np.abs(f_val - J @ y_next))
            
            if not np.all(np.isfinite(y_next)) or np.max(np.abs(y_next)) > 1e12:
                return None
            
            # Heuristic: if residual is too large, the linear model is invalid
            # Threshold depends on scale of y
            scale = np.max(np.abs(y_next)) + 1.0
            if residual > 1e-8 * scale:
                # Linear assumption failed, abort exponential method
                return None
                
            y_curr = y_next
        
        return y_curr

    # Attempt 1: Exponential method
    y_result = try_exponential_solve()
    
    if y_result is not None:
        # Verify accuracy with a finer step or different method?
        # For now, trust it if residual was low.
        # To be safer, we can do a quick check with solve_ivp if time permits
        # but the score penalty for calls is high, so we want to stop if possible.
        
        # Optional: Refine with a few Radau steps if time permits and we want higher precision
        # But exponential with low residual should be ~machine precision for linear systems.
        
        return y_result

    # Fallback: Use scipy.integrate.solve_ivp with Radau (implicit, stiff)
    # Radau is 5th order, L-stable.
    # We want high accuracy, so tight tolerances.
    # But we want few calls. Radau is efficient for stiff problems.
    
    def solve_ivp_fallback():
        # Use Radau method
        # max_step can be used to limit calls, but adaptive is better
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-10,
            atol=1e-12,
            max_step=(t1 - t0) / 10.0  # Limit step size to prevent huge steps if needed
        )
        if not sol.success:
            # Try BDF
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-10,
                atol=1e-12,
                max_step=(t1 - t0) / 10.0
            )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: RK45 with tight tolerance
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='RK45',
                rtol=1e-10,
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            raise RuntimeError("All solvers failed")

    if time.time() > deadline:
        # If we are out of time, return the last best guess?
        # We don't have one. Raise error or return y0?
        # Better to try a very fast explicit step if we have time, but we are at deadline.
        # Return y0 is bad. Let's try a single big step with RK4?
        # No, just return what we have or raise.
        # Actually, the exponential solver might have partially run.
        # But we returned None on failure.
        # Let's just try the fallback, it might be fast enough.
        pass

    y_result = solve_ivp_fallback()
    return y_result
# EVOLVE-BLOCK-END
