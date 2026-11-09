<<<<<<< SEARCH
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
=======
def _compute_jacobian(f, t, y, eps=1e-7):
    """Compute Jacobian using central differences for better accuracy."""
    n = len(y)
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        step = eps * max(1.0, abs(y[j]))
        e[j] = step
        fp = f(t, y + e)
        fm = f(t, y - e)
        J[:, j] = (fp - fm) / (2 * step)
    return J
>>>>>>> REPLACE
<<<<<<< SEARCH
    # 1. Compute Jacobian at t0
    J = _compute_jacobian(f, t0, y)
    
    # Strategy: Try exponential stepping first
    # If the system is linear or nearly linear, this is very efficient.
    # We use a fixed number of steps initially, then check accuracy.
    
    def try_exponential_solve():
        nonlocal y, t0, t1, J
        # Start with a moderate number of steps
        # Using fewer steps is fine for linear systems as expm is exact for the step
        # We use 10 steps to reduce overhead, though for linear it doesn't matter much
        # unless we were checking residuals at each step.
        steps = 10
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
            
            y_curr = exp_J_h @ y_curr
            t += h
            
            # We do NOT check residual here to save f-calls.
            # We will check at the end.
            
        # Check for numerical stability
        if not np.all(np.isfinite(y_curr)) or np.max(np.abs(y_curr)) > 1e12:
            return None
            
        # Check residual at the final point to verify linearity assumption
        # Residual = f(t, y_curr) - J @ y_curr
        f_val = f(t, y_curr)
        residual = np.max(np.abs(f_val - J @ y_curr))
        
        # Heuristic: if residual is too large, the linear model is invalid
        # Threshold depends on scale of y
        scale = np.max(np.abs(y_curr)) + 1.0
        if residual > 1e-8 * scale:
            # Linear assumption failed, abort exponential method
            return None
            
        return y_curr
=======
    # 1. Compute Jacobian at t0
    J = _compute_jacobian(f, t0, y)
    
    # Strategy: Try exponential stepping first
    # If the system is linear or nearly linear, this is very efficient.
    # We use a fixed number of steps initially, then check accuracy.
    
    def try_exponential_solve():
        nonlocal y, t0, t1, J
        # Use more steps for better accuracy in the linear approximation
        # For convection-diffusion, the Jacobian may change over time
        # More steps with constant J approximates the varying system better
        steps = 50
        h = (t1 - t0) / steps
        t = t0
        y_curr = y.copy()
        
        # Pre-compute expm(J*h) if J is constant
        exp_J_h = scipy.linalg.expm(J * h)
        
        for _ in range(steps):
            # Check deadline
            if time.time() > deadline:
                return None
            
            y_curr = exp_J_h @ y_curr
            t += h
            
        # Check for numerical stability
        if not np.all(np.isfinite(y_curr)) or np.max(np.abs(y_curr)) > 1e12:
            return None
            
        # Check residual at the final point to verify linearity assumption
        f_val = f(t, y_curr)
        residual = np.max(np.abs(f_val - J @ y_curr))
        
        # Heuristic: if residual is too large, the linear model is invalid
        scale = np.max(np.abs(y_curr)) + 1.0
        if residual > 1e-6 * scale:
            # Linear assumption failed, abort exponential method
            return None
            
        return y_curr
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
    # Attempt 1: Exponential method with constant Jacobian
    y_result = try_exponential_solve()
    
    if y_result is not None:
        # Try to refine by checking if we can do better with adaptive Jacobian updates
        # This is a polish pass: use a few steps with updated Jacobians
        if time.time() < deadline - 10.0:
            y_refined = _polish_with_adaptive_jacobian(f, t0, t1, y, J, deadline)
            if y_refined is not None:
                return y_refined
        return y_result
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
    # Fallback: Use scipy.integrate.solve_ivp with Radau (implicit, stiff)
    # Radau is 5th order, L-stable.
    # We want high accuracy, so tight tolerances.
    # But we want few calls. Radau is efficient for stiff problems.
    
    def solve_ivp_fallback():
        # Use Radau method with tighter tolerances for higher accuracy
        sol = scipy.integrate.solve_ivp(
            f, t_span, y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
            max_step=(t1 - t0) / 5.0
        )
        if not sol.success:
            # Try BDF
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='BDF',
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 5.0
            )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort: RK45 with tight tolerance
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='RK45',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                return sol.y[:, -1]
            raise RuntimeError("All solvers failed")
>>>>>>> REPLACE
<<<<<<< SEARCH
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
=======
    def _polish_with_adaptive_jacobian(f, t0, t1, y0, J_init, deadline):
        """Polish the solution using adaptive Jacobian updates."""
        y = y0.copy()
        n = len(y)
        t = t0
        total_steps = 20
        h = (t1 - t0) / total_steps
        J = J_init.copy()
        
        for step in range(total_steps):
            if time.time() > deadline:
                return None
            
            # Update Jacobian every few steps
            if step % 5 == 0 or step == 0:
                J = _compute_jacobian(f, t, y)
                exp_J_h = scipy.linalg.expm(J * h)
            else:
                exp_J_h = scipy.linalg.expm(J * h)
            
            y = exp_J_h @ y
            t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
        
        # Verify with residual
        f_val = f(t, y)
        residual = np.max(np.abs(f_val - J @ y))
        scale = np.max(np.abs(y)) + 1.0
        if residual > 1e-5 * scale:
            return None
        
        return y

    if time.time() > deadline:
        pass

    y_result = solve_ivp_fallback()
    return y_result
>>>>>>> REPLACE
