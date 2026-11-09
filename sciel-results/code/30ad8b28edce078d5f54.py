import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    h_total = t1 - t0
    y0 = np.array(y0, dtype=float)
    N = len(y0)
    
    start_time = time.time()
    deadline = start_time + 290.0 # 10s margin
    
    # 1. Probe the system to see if it's a constant-coefficient linear system
    # y' = A y + b
    A = None
    b = None
    
    try:
        # Call f(0, 0) to get b
        b = f(t0, np.zeros(N))
        
        # Call f(0, e_j) to get columns of A
        A = np.zeros((N, N))
        for j in range(N):
            e_j = np.zeros(N)
            e_j[j] = 1.0
            val = f(t0, e_j)
            A[:, j] = val - b
            
        # Check if the model holds at the end point t1
        # y'(t1) should be A y(t1) + b? We don't have y(t1) yet.
        # But we can check if f(t1, y0) approx A y0 + b.
        # If the system is autonomous, f(t, y) doesn't depend on t.
        # So f(t1, y0) should equal A y0 + b.
        
        # Check if the model holds at multiple points for robustness
        # Test at t0 with y0, and at t1 with a different vector
        f_t0_y0 = f(t0, y0)
        pred_t0_y0 = A @ y0 + b
        err1 = np.linalg.norm(f_t0_y0 - pred_t0_y0) / (1.0 + np.linalg.norm(pred_t0_y0))
        
        # Test at t1 with y0 (checks time-independence)
        f_t1_y0 = f(t1, y0)
        pred_t1_y0 = A @ y0 + b
        err2 = np.linalg.norm(f_t1_y0 - pred_t1_y0) / (1.0 + np.linalg.norm(pred_t1_y0))
        
        # Test at t_mid with a scaled y0 to catch non-linearity
        t_mid = t0 + 0.5 * h_total
        y_test2 = 2.0 * y0
        f_mid_test2 = f(t_mid, y_test2)
        pred_mid_test2 = A @ y_test2 + b
        err3 = np.linalg.norm(f_mid_test2 - pred_mid_test2) / (1.0 + np.linalg.norm(pred_mid_test2))
        
        max_err = max(err1, err2, err3)
        
        # If the model is consistent (autonomous linear), use exact solution
        if max_err < 1e-8:
            # Construct augmented system for exact solution
            # z' = [ A  b ]
            #     [ 0  0 ]
            # z = [y]
            #     [1]
            A_aug = np.zeros((N + 1, N + 1))
            A_aug[:N, :N] = A
            A_aug[:N, N] = b
            # Last row is 0
            
            z0 = np.zeros(N + 1)
            z0[:N] = y0
            z0[N] = 1.0
            
            try:
                if time.time() > deadline - 30.0:
                    raise TimeoutError("deadline approaching")
                # Compute matrix exponential
                exp_Ah = la.expm(A_aug * h_total)
                z_end = exp_Ah @ z0
                y_end = z_end[:N]
                
                if np.all(np.isfinite(y_end)):
                    return y_end
            except Exception:
                pass # Fall through to time stepping
                
    except Exception:
        # If probing fails completely, A and b might be None or partial
        A = None
        b = None

    # 2. Fallback: Use scipy.integrate.solve_ivp with Radau for stiff systems
    # If we computed A during probing, pass it as the Jacobian to save evaluations
    jac_func = None
    if A is not None:
        # We assume A is the Jacobian. If the system is linear, this is exact.
        # If the system is non-linear, A is a linearization at t0, which might be good enough
        # or at least better than finite differences for the first step.
        # However, for non-linear systems, the Jacobian changes. 
        # But the task is "linear system", so A should be constant.
        # If the check failed, maybe it's not constant? Or maybe numerical noise?
        # Given the task description "fixed 15-dimensional stiff linear system", 
        # it is highly likely A is constant. 
        # Passing it is a safe bet to reduce calls.
        jac_func = lambda t, y: A
        
    try:
        # Try Radau first
        sol = integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12, 
            t_eval=[t1],
            jac=jac_func
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau fails, try BDF
            sol = integrate.solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-12, 
                atol=1e-12, 
                t_eval=[t1],
                jac=jac_func
            )
            if sol.success:
                return sol.y[:, -1]
    except Exception:
        pass
    
    # 3. Last resort fallback: Simple Backward Euler (kept from original but simplified)
    # This should rarely be reached
    steps = 100
    h = h_total / steps
    t = t0
    y = y0.copy()
    
    for i in range(steps):
        t_new = t + h
        y_new = y.copy()
        
        for _newton in range(10):
            f_val = f(t_new, y_new)
            # Use the probed A as Jacobian if available, else finite diff
            if A is not None:
                J = A
            else:
                J = np.zeros((N, N))
                for j in range(N):
                    e_j = np.zeros(N)
                    step_size = 1e-7 * max(1.0, abs(y_new[j]))
                    e_j[j] = step_size
                    J[:, j] = (f(t_new, y_new + e_j) - f_val) / step_size
            
            F_val = y_new - h * f_val - y
            M = np.eye(N) - h * J
            
            try:
                delta = la.solve(M, -F_val)
            except la.LinAlgError:
                delta = la.lstsq(M, -F_val)[0]
                
            y_new = y_new + delta
            
            if np.linalg.norm(delta) < 1e-10 * (1 + np.linalg.norm(y_new)):
                break
        
        y = y_new
        t = t_new
        
        if time.time() > deadline:
            break
            
    return y
