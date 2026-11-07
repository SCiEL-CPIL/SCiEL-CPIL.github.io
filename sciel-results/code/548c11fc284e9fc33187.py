"""Root finder for 3 fixed 2-D systems: Global Grid Search + Damped Newton."""
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Global Search: Scan a grid of points in a large box around x0 to find 
       a point with small ||F(x)||. This handles cases where x0 is far from the root.
    2. Local Refinement: Use Damped Newton's method starting from the best 
       grid point to achieve high precision (1e-8) with few iterations.
    
    Budget:
    - Time limit: 120s (using 115s internal deadline).
    - Call limit: < 500 calls.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    def time_ok():
        return time.time() - start_time < time_limit

    def safe_eval_F(x):
        """Evaluate F(x) safely, returning (value, norm, success)."""
        try:
            val = F(np.asarray(x, dtype=float))
            val = np.asarray(val, dtype=float)
            if val.shape != (2,) or not np.all(np.isfinite(val)):
                return None, np.inf, False
            norm = np.linalg.norm(val)
            return val, norm, True
        except Exception:
            return None, np.inf, False

    # 1. Global Search Phase
    # We try to find a starting point that is closer to the root.
    # We scan a set of radii and angles around x0.
    # This is cheap enough to do within the budget if we are careful.
    
    x0 = np.asarray(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    best_x = x0.copy()
    best_F = None
    best_norm = np.inf
    
    # Check x0 first
    Fx, norm, ok = safe_eval_F(x0)
    if ok:
        best_F = Fx
        best_norm = norm
        best_x = x0.copy()
        
    if best_norm <= 1e-8:
        return best_x

    # Define search grid
    # Radii: logarithmic spacing from small to large
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    # Limit global search to ~100 calls max to leave room for Newton
    max_global_calls = 100
    calls_made = 1 # x0
    
    for r in radii:
        if not time_ok():
            break
        for theta in angles:
            if calls_made >= max_global_calls:
                break
            if not time_ok():
                break
            
            # Generate candidate point
            # If x0 is far from origin, maybe search relative to x0? 
            # The problem says x0 is given. Usually x0 is somewhat reasonable or random.
            # Searching around x0 is safer than assuming origin is root.
            dx = r * np.cos(theta)
            dy = r * np.sin(theta)
            x_cand = x0 + np.array([dx, dy])
            
            # Optional: also check points relative to origin if x0 is 0 or large?
            # Let's stick to x0-relative for now, but also check a few absolute points if x0 is near 0
            # Actually, let's just add a few absolute points if x0 is small
            if np.linalg.norm(x0) < 1.0:
                 x_cand2 = np.array([dx, dy])
            else:
                 x_cand2 = None
                 
            # Evaluate x_cand
            Fx_c, norm_c, ok_c = safe_eval_F(x_cand)
            calls_made += 1
            
            if ok_c and norm_c < best_norm:
                best_norm = norm_c
                best_F = Fx_c
                best_x = x_cand.copy()
                
                if best_norm <= 1e-8:
                    return best_x
            
            # Evaluate x_cand2 if different
            if x_cand2 is not None and not np.allclose(x_cand, x_cand2):
                Fx_c2, norm_c2, ok_c2 = safe_eval_F(x_cand2)
                calls_made += 1
                if ok_c2 and norm_c2 < best_norm:
                    best_norm = norm_c2
                    best_F = Fx_c2
                    best_x = x_cand2.copy()
                    
                    if best_norm <= 1e-8:
                        return best_x
                        
            if not time_ok():
                break
        if not time_ok():
            break

    # 2. Local Refinement Phase: Damped Newton
    # Start from best_x found in global search
    
    x = best_x.copy()
    Fx = best_F
    
    # If we didn't find a valid Fx, try x0 again or just return best_x
    if Fx is None:
        Fx, norm, ok = safe_eval_F(x)
        if not ok:
            return x # Return whatever we have, likely failure but avoids exception
        
    # Newton iterations
    # We use a finite difference Jacobian.
    # To save calls, we reuse Fx.
    # J is 2x2. We need 2 more calls for J columns if we don't have them.
    # Total 3 calls per iteration (1 for Fx check, 2 for J).
    # We have ~400 calls left. 100 iterations max.
    
    max_iters = 80
    h = 1e-6
    
    for i in range(max_iters):
        if not time_ok():
            break
            
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-12:
            return x
            
        # Compute Jacobian
        J = np.zeros((2, 2))
        
        # Column 0
        e0 = np.zeros(2)
        e0[0] = h
        x_p0 = x + e0
        Fx_p0, _, ok0 = safe_eval_F(x_p0)
        if not ok0:
            break
        J[:, 0] = (Fx_p0 - Fx) / h
        
        # Column 1
        e1 = np.zeros(2)
        e1[1] = h
        x_p1 = x + e1
        Fx_p1, _, ok1 = safe_eval_F(x_p1)
        if not ok1:
            break
        J[:, 1] = (Fx_p1 - Fx) / h
        
        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, try a simple gradient step on ||F||^2?
            # grad = J^T F
            grad = J.T @ Fx
            dx = -grad / (np.linalg.norm(grad) + 1e-12)
            
        # Damping: find lambda in [1, 0.5, 0.25, ...] such that ||F(x + lambda dx)|| < ||F(x)||
        lambda_ = 1.0
        Fx_new = None
        x_new = x
        
        # Max 5 line search steps
        for _ in range(5):
            x_trial = x + lambda_ * dx
            Fx_trial, norm_trial, ok_trial = safe_eval_F(x_trial)
            
            if ok_trial and norm_trial < np.linalg.norm(Fx):
                Fx_new = Fx_trial
                x_new = x_trial
                break
            lambda_ *= 0.5
            
        if Fx_new is None:
            # If no improvement found with damping, break
            break
            
        x = x_new
        Fx = Fx_new
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x
            
    return x
