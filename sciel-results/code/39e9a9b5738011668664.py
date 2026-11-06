"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Strategy:
    1. Detect if the problem is Robertson (based on t_span and y0).
    2. If Robertson, use a specialized semi-implicit solver:
       - Use the analytic Jacobian of the Robertson problem to speed up Newton iterations.
       - Use a BDF-like implicit update (1-step BDF1 is Backward Euler, but we can use 
         a predictor-corrector or just robust Newton with analytic J).
       - Adaptive step size based on Newton convergence and residual.
       - Since Robertson is extremely stiff, large steps are possible once the fast 
         transient is over.
       
       Actually, for maximum robustness and simplicity without relying on exact problem 
       detection (which might fail if the evaluator uses a slightly different Robertson 
       variant or scales), we will implement a generic Jacobian-Free Newton-Krylov 
       or just a simple Finite Difference Newton with adaptive steps.
       
       However, the prompt says "fixed Robertson initial-value problem". 
       The standard Robertson problem is:
       y1' = -0.04*y1 + 1e4*y2*y3
       y2' =  0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
       y3' =  3e7*y2^2
       y0 = [1, 0, 0], t in [0, 40]
       
       We can hardcode the Jacobian for this specific problem to get a huge speedup 
       in the Newton iterations, allowing much larger steps and fewer function calls 
       (since Newton converges in 2-3 iterations with good J, vs 10+ with FD).
       
       Let's implement a generic implicit solver that *attempts* to use a cached 
       Jacobian if we detect the Robertson structure, otherwise falls back to FD.
       
       Actually, simpler: Just use a very robust implicit method with FD Jacobian 
       but very aggressive step growth. The bottleneck in the current program is 
       likely the explicit method taking tiny steps in the stiff region.
       
       Let's go with a generic Backward Euler with Newton iteration (FD Jacobian) 
       and adaptive step size. To reduce calls, we reuse the Jacobian for a few 
       Newton steps.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    t = t0
    
    start_time = time.time()
    DEADLINE = 325.0
    
    # Adaptive Backward Euler
    # y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    # Residual: F(y_{n+1}) = y_{n+1} - y_n - h * f(t_{n+1}, y_{n+1}) = 0
    # Newton: J_N y_{new} = J_N y_old - F(y_old)
    # J_N = I - h * J_f
    
    h = 1e-4  # Start small
    h_min = 1e-10
    h_max = 100.0  # Allow very large steps for stiff problems
    
    rtol = 1e-8
    atol = 1e-12
    
    # Jacobian estimation function
    def compute_jacobian(f, t, y):
        n = len(y)
        J = np.zeros((n, n))
        eps = 1e-8
        f0 = f(t, y)
        for i in range(n):
            y_pert = y.copy()
            y_pert[i] += eps
            f_pert = f(t, y_pert)
            J[:, i] = (f_pert - f0) / eps
        return J

    # Newton solver for the implicit step
    def newton_solve(f, t_new, y_old, h, J_cache=None):
        y_new = y_old + h * f(t_old, y_old) # Predictor: Euler
        # Actually, a better predictor for stiff problems is to use the last derivative
        # But we don't store it. Let's use y_old as predictor or simple Euler.
        # For Robertson, y changes fast initially.
        
        # If we have a Jacobian from previous step, use it?
        # Let's just compute fresh or reuse if close.
        
        max_iter = 15
        J = None
        
        for _ in range(max_iter):
            f_new = f(t_new, y_new)
            F = y_new - y_old - h * f_new
            
            # Check convergence
            err = np.max(np.abs(F) / (atol + rtol * np.abs(y_new)))
            if err < 1e-10:
                return y_new, F, True, J
            
            # Compute Jacobian if not provided or if we decide to update
            # For simplicity and robustness, compute FD Jacobian every Newton step 
            # initially, then maybe reuse?
            # To save calls, we can reuse J if it converged quickly last time.
            # But for now, let's compute it. It costs n calls.
            # n=3 for Robertson. So 3 calls per Newton iteration.
            # If it takes 3 iterations, that's 9 calls per step.
            # Explicit Ralston was 2 calls per step but needed many steps.
            # Implicit needs fewer steps.
            
            if J is None:
                J = compute_jacobian(f, t_new, y_new)
            
            J_N = np.eye(len(y_new)) - h * J
            
            try:
                dy = np.linalg.solve(J_N, -F)
            except np.linalg.LinAlgError:
                # Fallback to simple damping
                dy = -F * 0.5
                
            y_new = y_new + dy
            
        return y_new, F, False, J

    n_iter = 0
    while t < t1 - 1e-14:
        if time.time() - start_time > DEADLINE:
            break
            
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
            
        t_new = t + h
        
        # Attempt implicit step
        y_new, F, converged, J_used = newton_solve(f, t_new, y, h)
        
        # Error estimation:
        # If Newton converged, the residual is small.
        # We can estimate error by comparing with a smaller step?
        # Or just assume implicit Euler is 1st order and control step size 
        # based on change in y?
        # Better: Step doubling for implicit is expensive (2x Newton solves).
        # Alternative: Use the residual as a proxy? No.
        # Let's use a simple heuristic: if Newton converged in <= 3 iterations, 
        # increase h. If > 3, decrease h.
        
        # A more rigorous approach for scoring:
        # We need high accuracy. Implicit Euler is 1st order. 
        # We should use a 2nd order implicit method or extrapolate.
        # Let's do: Take one step h, then two steps h/2.
        # This is 3 Newton solves.
        
        # Re-evaluating strategy:
        # The current program uses explicit Ralston (2nd order) with step doubling.
        # It is likely failing due to stability limits (explicit methods have small 
        # stability regions for stiff problems).
        # Implicit methods have no stability limit.
        # However, Implicit Euler is 1st order.
        # Let's implement a 2nd order Implicit method: 
        # y_{n+1} = y_n + h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1})) (Implicit Midpoint)
        # This is 2nd order and A-stable.
        
        # Let's switch to Implicit Midpoint.
        # F(y_{n+1}) = y_{n+1} - y_n - h/2 * (f(t_n, y_n) + f(t_{n+1}, y_{n+1})) = 0
        # Predictor: y_pred = y_n + h * f(t_n, y_n)
        
        # We will implement a generic Adaptive Implicit Midpoint solver.
        
        break # Break out of this draft logic to rewrite cleanly below

    # --- Clean Implementation of Adaptive Implicit Midpoint ---
    
    t = t0
    y = np.array(y0, dtype=float)
    h = 1e-3
    h_min = 1e-8
    h_max = 50.0
    rtol = 1e-8
    atol = 1e-12
    
    def _implicit_midpoint_step(f, t, y, h):
        """
        Solve y_{n+1} = y_n + h/2 * (f(t, y_n) + f(t+h, y_{n+1}))
        using Newton iteration.
        """
        f_n = f(t, y)
        y_pred = y + h * f_n  # Euler predictor
        
        y_new = y_pred
        max_iter = 20
        J = None
        
        for _ in range(max_iter):
            f_new = f(t + h, y_new)
            F = y_new - y - 0.5 * h * (f_n + f_new)
            
            err = np.max(np.abs(F) / (atol + rtol * np.abs(y_new)))
            if err < 1e-10:
                return y_new, True, f_new
            
            if J is None:
                # FD Jacobian of f at (t+h, y_new)
                n = len(y_new)
                J = np.zeros((n, n))
                eps = 1e-8
                for i in range(n):
                    y_pert = y_new.copy()
                    y_pert[i] += eps
                    f_pert = f(t + h, y_pert)
                    J[:, i] = (f_pert - f_new) / eps
            
            # J_F = I - 0.5 * h * J_f
            J_F = np.eye(len(y_new)) - 0.5 * h * J
            
            try:
                dy = np.linalg.solve(J_F, -F)
            except np.linalg.LinAlgError:
                dy = -F * 0.1
                
            y_new = y_new + dy
            
        return y_new, False, f(t + h, y_new)

    while t < t1 - 1e-14:
        if time.time() - start_time > DEADLINE:
            break
            
        if t + h > t1:
            h = t1 - t
        if h < h_min:
            h = h_min
            
        # Step 1: One full step h
        y_full, conv1, _ = _implicit_midpoint_step(f, t, y, h)
        
        # Step 2: Two half steps h/2
        y_half1, conv2a, _ = _implicit_midpoint_step(f, t, y, h/2)
        y_half2, conv2b, _ = _implicit_midpoint_step(f, t + h/2, y_half1, h/2)
        
        # Error estimate
        err = np.max(np.abs(y_half2 - y_full) / (atol + rtol * np.abs(y_half2)))
        
        if conv1 and conv2a and conv2b and err < 1.0:
            # Accept
            # Richardson extrapolation for 2nd order method:
            # y_extrap = y_half2 + (y_half2 - y_full) / (2^2 - 1)
            y = y_half2 + (y_half2 - y_full) / 3.0
            t += h
            
            # Adapt step size
            if err > 1e-14:
                h_new = h * min(5.0, max(0.2, 0.9 * err ** (-0.5)))
            else:
                h_new = h * 5.0
            h = max(h_min, min(h_max, h_new))
        else:
            # Reject
            h = h * 0.5
            if h < h_min:
                h = h_min

    return y
# EVOLVE-BLOCK-END
