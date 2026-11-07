import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s wall clock if that's the hard kill)
    # The prompt says "evaluator KILLS the program after 150s wall-clock"
    # But also "Each evaluation must finish within 120 s".
    # Let's aim for 100s to be safe.
    
    max_calls = 400
    call_count = 0
    
    def F_wrapper(x):
        nonlocal call_count
        call_count += 1
        if call_count > max_calls:
            raise RuntimeError("Max calls exceeded")
        return np.array(F(x), dtype=float)

    x = np.array(x0, dtype=float)
    
    # Check initial point
    Fx = F_wrapper(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Initialize Jacobian
    # Use finite difference for initial Jacobian to get a good start
    # This costs 2 calls
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F_wrapper(x + e) - Fx) / h
    
    # Broyden's method loop
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            s, _, _, _ = np.linalg.lstsq(J, -Fx, rcond=None)
            
        # Line search
        alpha = 1.0
        x_new = x + alpha * s
        Fx_new = F_wrapper(x_new)
        
        # Check if step improved
        if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
            # Update Jacobian (Broyden's Good Method)
            # J_new = J + (y - J s) s^T / (s^T s)
            y = Fx_new - Fx
            ds = s.T @ s
            if ds > 1e-12:
                J = J + np.outer(y - J @ s, s) / ds
            x = x_new
            Fx = Fx_new
        else:
            # Backtracking line search
            accepted = False
            for _ in range(10):
                alpha *= 0.5
                x_new = x + alpha * s
                Fx_new = F_wrapper(x_new)
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    y = Fx_new - Fx
                    s_actual = alpha * s
                    ds = s_actual.T @ s_actual
                    if ds > 1e-12:
                        J = J + np.outer(y - J @ s_actual, s_actual) / ds
                    x = x_new
                    Fx = Fx_new
                    accepted = True
                    break
            if not accepted:
                # If line search fails, try a different direction or break
                # Maybe try inverse Hessian step or random perturbation?
                # For now, break and rely on fallback
                break

        if np.linalg.norm(Fx) <= 1e-8:
            return x

    # Fallback: Try SciPy root
    try:
        from scipy.optimize import root
        res = root(F_wrapper, x, method='hybr', tol=1e-8)
        if res.success and np.linalg.norm(res.fun) <= 1e-8:
            return res.x
    except:
        pass

    return x
