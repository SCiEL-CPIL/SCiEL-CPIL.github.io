"""ODE solver: Robertson problem - optimized for accuracy vs. evaluation cost."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Robertson problem: y' = f(t, y)
# y0 = [1, 0, 0]
# y' = [-a1*y0 + a3*y1*y2, a1*y0 - a2*y1 - a3*y1*y2, a2*y1]
# a1 = 0.04, a2 = 3e7, a3 = 1e10
# t in [0, 40]

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Uses an adaptive stiff solver approach for the Robertson problem.
    Returns the final state y(t_end).
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Count evaluations
    call_count = 0
    
    # Robertson problem parameters (standard)
    a1 = 0.04
    a2 = 3e7
    a3 = 1e10
    
    # Jacobian of the Robertson problem
    def jac(t, y):
        nonlocal call_count
        # For Robertson: J = [[-a1, a3*y2], [a1, -a2-a3*y2], [0, a2]]
        return np.array([
            [-a1, a3 * y[2]],
            [a1, -a2 - a3 * y[2]],
            [0.0, a2]
        ])
    
    # We'll use a stiff ODE solver approach: BDF (Backward Differentiation Formula)
    # or implicit Runge-Kutta. Let's use a simple implicit method with Newton iteration.
    
    # Strategy: Use adaptive step-size implicit Euler / BDF with Newton's method
    # and Jacobian reuse.
    
    # For the Robertson problem, we can use the known structure.
    # Let's implement a 2nd-order BDF (Backward Differentiation Formula) method.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Use a fixed-step implicit method with Newton iteration
    # 2. Start with small steps and increase as the solution stabilizes
    
    # The Robertson problem has three phases:
    # Phase 1: t in [0, ~0.1] - fast transient, y0 drops from 1 to ~1e-6
    # Phase 2: t in [0.1, ~10] - slow evolution, y1 peaks then decays
    # Phase 3: t in [10, 40] - very slow, y2 approaches 1
    
    # Use adaptive stepping with implicit Euler + Newton
    
    h = 1e-4  # Initial small step
    t = t0
    y_prev = y.copy()
    y_curr = y.copy()
    
    # For BDF2, we need the previous step
    # Let's use implicit Euler first, then switch to BDF2
    
    max_newton_iter = 50
    tol = 1e-12
    
    # Precompute some constants
    eye = np.eye(n)
    
    while t < t1:
        if time.time() > deadline:
            break
        
        # Don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Implicit Euler: (y_{n+1} - y_n) / h = f(t_{n+1}, y_{n+1})
        # => y_{n+1} - h * f(t_{n+1}, y_{n+1}) = y_n
        
        t_new = t + h
        y_new = y_curr.copy()
        
        # Newton iteration
        converged = False
        for _ in range(max_newton_iter):
            if time.time() > deadline:
                break
            
            fz = f(t_new, y_new)
            call_count += 1
            
            # Compute Jacobian using finite differences (only when needed)
            J = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n)
                scale = max(1.0, abs(y_new[j]))
                e[j] = 1e-8 * scale
                J[:, j] = (f(t_new, y_new + e) - fz) / e[j]
                call_count += 1
            
            # Solve (I - h*J) * delta = -(y_new - y_curr - h*fz)
            A = eye - h * J
            b = -(y_new - y_curr - h * fz)
            
            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fallback to least squares
                delta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            y_new = y_new + delta
            
            if np.linalg.norm(delta) <= tol * (1.0 + np.linalg.norm(y_new)):
                converged = True
                break
        
        if not converged:
            # If Newton didn't converge, reduce step size
            h *= 0.5
            continue
        
        # Check for step size adjustment based on error estimate
        # Simple heuristic: if the step was very small relative to the solution change,
        # we can increase the step
        
        y_prev = y_curr
        y_curr = y_new
        t = t_new
        
        # Adaptive step size control
        # Estimate local error by comparing with a smaller step
        # For simplicity, use a heuristic based on the Newton convergence
        # If Newton converged in few iterations, we can increase h
        # Otherwise, keep h or decrease
        
        # Simple adaptive strategy:
        # If t is in the fast transient region (t < 0.1), use smaller steps
        # Otherwise, use larger steps
        if t < 0.01:
            h_target = 1e-4
        elif t < 0.1:
            h_target = 1e-3
        elif t < 1.0:
            h_target = 1e-2
        elif t < 10.0:
            h_target = 0.1
        else:
            h_target = 1.0
        
        # Adjust h towards h_target
        if h < h_target:
            h = min(h * 2.0, h_target)
        elif h > h_target:
            h = max(h * 0.5, h_target)
    
    return y_curr
# EVOLVE-BLOCK-END
