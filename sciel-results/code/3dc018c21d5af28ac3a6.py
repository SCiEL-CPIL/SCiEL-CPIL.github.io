import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Solve the Robertson problem using a custom Newton-Backward Euler method
    with adaptive step size and analytical Jacobian exploitation.
    
    Strategy:
    1. Detect if the problem is the Robertson problem (3D, specific structure).
    2. If so, use an analytical Jacobian J(t, y) = [[0, 1e4*y2, 1e4*y1],
                                                    [0, -1e4*y2, -1e4*y1],
                                                    [0, 3e3*y2, -3e3*y2]]
       Note: The standard Robertson system is:
       y1' = -1e4 y1 y2
       y2' = 1e4 y1 y2 - 3e3 y2^2
       y3' = 3e3 y2^2
       
       Wait, the standard Robertson problem is:
       y1' = k1*y1*y2
       y2' = -k1*y1*y2 - k2*y2^2
       y3' = k2*y2^2
       with k1=1e4, k2=3e3.
       
       The Jacobian is:
       J = [ -k1*y2,  k1*y1,  0 ]
           [ -k1*y2, -k1*y1 - 2*k2*y2, 0 ]
           [ 0,      2*k2*y2,          0 ]
       
       3. Use Newton-Backward Euler for stability.
       4. Estimate local error using a Richardson extrapolation or by comparing
          with a smaller step, or simply use a very tight tolerance on the Newton
          residual and a fixed, sufficiently small step size if adaptive logic is
          complex. However, adaptive is better for score.
       
       Let's implement a robust adaptive Newton-Backward Euler.
       Error estimation: Compare BE step h with two half-steps h/2.
       Local error ~ |y_{h/2+h/2} - y_h|.
       Adjust h based on error.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # 50s safety margin
    
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Constants for Robertson
    k1 = 1e4
    k2 = 3e3
    
    # Check if this is Robertson
    is_robertson = False
    if n == 3:
        # Heuristic: check initial conditions or function behavior
        # We can assume it is if y0 is close to [1,0,0] and t_span is [0,40]
        # But to be safe, we can just try to use the analytical Jacobian if it matches
        # the finite difference one at the first step.
        # For now, let's assume it is Robertson if n==3 and t1-t0 is around 40.
        # Or better, just implement the generic case with analytical Jacobian if possible.
        # Since the task says "fixed Robertson initial-value problem", we can hardcode the structure
        # if we are confident.
        # Let's verify by calling f once.
        try:
            f_test = f(t0, y)
            # If y0 = [1, 0, 0], f should be [0, -1e4, 3e3]
            # Actually, y1' = -k1*y1*y2 = 0
            # y2' = k1*y1*y2 - k2*y2^2 = 0
            # y3' = k2*y2^2 = 0
            # Wait, standard Robertson:
            # A -> B -> C
            # A' = -k1 A B
            # B' = k1 A B - k2 B^2
            # C' = k2 B^2
            # If y0 = [1, 0, 0], then A=1, B=0, C=0.
            # A' = 0
            # B' = 0
            # C' = 0
            # This is a singular start for the Jacobian? No, J is fine.
            # J[0,1] = -k1*y1 = -1e4
            # J[1,0] = k1*y2 = 0
            # J[1,1] = -k1*y1 - 2*k2*y2 = -1e4
            # J[2,1] = 2*k2*y2 = 0
            # So J = [[0, -1e4, 0], [0, -1e4, 0], [0, 0, 0]]
            # This is singular! The system is stiff and has a fast transient.
            # Actually, the Jacobian of the RHS is:
            # df1/dy1 = -k1*y2
            # df1/dy2 = -k1*y1
            # df1/dy3 = 0
            # df2/dy1 = k1*y2
            # df2/dy2 = -k1*y1 - 2*k2*y2
            # df2/dy3 = 0
            # df3/dy1 = 0
            # df3/dy2 = 2*k2*y2
            # df3/dy3 = 0
            # At y=[1,0,0]:
            # J = [[0, -1e4, 0], [0, -1e4, 0], [0, 0, 0]]
            # This matrix is singular (rank 1). Newton's method might struggle.
            # However, the system is well-posed.
            
            # Let's stick to the generic robust approach but optimized.
            # If we detect Robertson, we can use the analytical Jacobian.
            # To handle the singularity at the start, we can take a very small first step
            # or use a modified Newton step.
            
            # Let's implement the analytical Jacobian function.
            def robertson_jac(t, y):
                J = np.zeros((3, 3))
                J[0, 0] = -k1 * y[1]
                J[0, 1] = -k1 * y[0]
                J[1, 0] = k1 * y[1]
                J[1, 1] = -k1 * y[0] - 2 * k2 * y[1]
                J[2, 1] = 2 * k2 * y[1]
                return J
                
            # Verify if f matches the Robertson structure at a random point
            # to avoid hardcoding if it's not Robertson.
            # But the task says "fixed Robertson". So we can assume it.
            is_robertson = True
        except:
            is_robertson = False

    # If not Robertson, fall back to generic finite difference Jacobian
    if not is_robertson:
        def jac(t, y):
            J = np.zeros((n, n))
            eps = 1e-7
            y_base = y.copy()
            f_base = f(t, y_base)
            for j in range(n):
                step = eps * max(1.0, abs(y_base[j]))
                y_pert = y_base.copy()
                y_pert[j] += step
                J[:, j] = (f(t, y_pert) - f_base) / step
            return J
    else:
        jac = robertson_jac

    # Newton-Backward Euler Step
    def newton_be_step(t, y, h):
        t_new = t + h
        # Solve: y_new = y + h * f(t_new, y_new)
        # g(y_new) = y_new - y - h * f(t_new, y_new) = 0
        y_new = y.copy()
        for _ in range(20):
            f_new = f(t_new, y_new)
            J = jac(t_new, y_new)
            # A = I - h * J
            A = np.eye(n) - h * J
            rhs = y_new - y - h * f_new
            try:
                delta = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError:
                # If singular, add small regularization
                A += 1e-10 * np.eye(n)
                delta = np.linalg.solve(A, rhs)
            
            y_new -= delta
            if np.linalg.norm(delta) < 1e-12 * np.linalg.norm(y_new):
                break
        return y_new

    # Adaptive Step Control
    rtol = 1e-12
    atol = 1e-15
    h = 1e-3
    t = t0
    y_prev = y.copy()
    
    # First step: take a small step to get off the singular start
    if is_robertson:
        h = 1e-6
        t_new = t + h
        y = newton_be_step(t, y, h)
        t = t_new
        h = 1e-4 # Increase step size after first step
    
    max_steps = 10000
    step_count = 0
    
    while t < t1 and step_count < max_steps:
        if time.time() > deadline:
            break
            
        # Ensure we don't step past t1
        if t + h > t1:
            h = t1 - t
        if h <= 0:
            break
            
        # Take a full step
        y_full = newton_be_step(t, y, h)
        
        # Take two half steps for error estimation
        y_mid = newton_be_step(t, y, h/2)
        y_half = newton_be_step(t + h/2, y_mid, h/2)
        
        # Error estimate
        err = np.linalg.norm(y_half - y_full)
        scale = atol + rtol * np.linalg.norm(y_half)
        
        if err > scale:
            # Reduce step size
            factor = 0.5 * (scale / err) ** 0.5
            if factor < 0.1:
                factor = 0.1
            h *= factor
            continue
        else:
            # Accept step
            t += h
            y = y_half # Use the more accurate half-step result
            # Increase step size if error is small
            if err < 0.1 * scale:
                h *= 1.5
            else:
                h *= 1.0
            step_count += 1
            y_prev = y.copy()
            
    return y
