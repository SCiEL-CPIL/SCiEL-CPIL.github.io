"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use Broyden's method (rank-1 update) which needs only 1 F-call per iteration
       after the initial Jacobian setup (3 calls).
    2. Include safeguards: if step diverges, use line search or restart.
    3. Track call count and time to stay within budget.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    call_count = 0
    
    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.flatten()[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result
    
    # Initial evaluation
    try:
        Fx = safe_F(x)
    except TimeoutError:
        return x
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Initial Jacobian via finite differences (3 calls: Fx + 2 more)
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_e = safe_F(x + e)
        except TimeoutError:
            return x
        J[:, j] = (Fx_e - Fx) / h
    
    # Broyden's method
    best_x = x
    best_Fx = Fx
    best_norm = np.linalg.norm(Fx)
    
    for iteration in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break
        
        # Check convergence
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm < best_norm:
            best_norm = Fx_norm
            best_x = x.copy()
            best_Fx = Fx.copy()
        
        if Fx_norm <= 1e-10:
            return x
        
        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Limit step size
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        
        x_new = x + dx
        
        # Evaluate F at new point
        try:
            Fx_new = safe_F(x_new)
        except TimeoutError:
            break
        
        Fx_new_norm = np.linalg.norm(Fx_new)
        
        # Check if we made progress
        if not np.all(np.isfinite(x_new)) or Fx_new_norm > Fx_norm * 1.1:
            # Step didn't help, try smaller step
            for factor in [0.5, 0.25, 0.1]:
                x_try = x + factor * dx
                try:
                    Fx_try = safe_F(x_try)
                except TimeoutError:
                    break
                Fx_try_norm = np.linalg.norm(Fx_try)
                if Fx_try_norm < Fx_norm:
                    # Update Broyden with smaller step
                    s = x_try - x
                    y = Fx_try - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s) if s @ s > 1e-14 else J
                    x = x_try
                    Fx = Fx_try
                    break
            else:
                # No good step found, restart with new Jacobian
                break
        else:
            # Good step, update Broyden
            s = x_new - x
            y = Fx_new - Fx
            if s @ s > 1e-14:
                J = J + np.outer(y - J @ s, s) / (s @ s)
            x = x_new
            Fx = Fx_new
    
    # Return the best point found
    if best_norm <= 1e-8:
        return best_x
    
    # If we didn't converge well, try a few more iterations from best point
    x = best_x.copy()
    Fx = best_Fx.copy()
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        if call_count >= 495:
            break
        
        Fx_norm = np.linalg.norm(Fx)
        if Fx_norm <= 1e-8:
            return x
        
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            try:
                Fx_e = safe_F(x + e)
            except TimeoutError:
                break
            J[:, j] = (Fx_e - Fx) / h
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        step_norm = np.linalg.norm(dx)
        if step_norm > 0.5:
            dx = dx / step_norm * 0.5
        
        x_new = x + dx
        try:
            Fx_new = safe_F(x_new)
        except TimeoutError:
            break
        
        if np.linalg.norm(Fx_new) < Fx_norm:
            x = x_new
            Fx = Fx_new
        else:
            break
    
    return x
