"""Root finder for 3 fixed 2D systems using Broyden's method with optimized restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method.
    Optimized for low call count and robustness.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    best_x = x.copy()
    # Initial evaluation
    try:
        F_x0 = F(x)
        best_norm = np.linalg.norm(F_x0)
    except:
        return np.zeros(2)
        
    if best_norm <= tol:
        return best_x

    max_restarts = 20
    
    # Strategy for restarts:
    # 0: Original x0
    # 1-4: Small perturbations
    # 5+: Larger random jumps
    
    for restart in range(max_restarts):
        if time_left() < 2.0:
            break
        
        if restart == 0:
            x_local = x.copy()
            F_local = F_x0.copy()
            norm_local = best_norm
        else:
            if restart < 5:
                # Small deterministic perturbations
                # Directions: e1, e2, e1+e2, e1-e2
                dirs = [np.array([1,0]), np.array([0,1]), np.array([1,1]), np.array([1,-1])]
                d = dirs[(restart-1) % 4]
                step = 0.1 * restart
                x_local = x + step * d
            else:
                # Random restarts
                # Mix of global and local
                if restart < 10:
                    # Local around best_x
                    radius = 0.5 * (restart - 4)
                    dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec)
                    x_local = best_x + radius * dir_vec
                else:
                    # Global random
                    x_local = np.random.uniform(-10, 10, size=2)
            
            if not np.all(np.isfinite(x_local)):
                continue
            
            try:
                F_local = F(x_local)
                norm_local = np.linalg.norm(F_local)
            except:
                continue
            
            if norm_local < best_norm:
                best_norm = norm_local
                best_x = x_local.copy()
            
            if norm_local <= tol:
                return x_local
        
        # Broyden's Good Method
        # Compute initial Jacobian
        # Use a robust h
        h = 1e-7
        # Scale h with x magnitude if x is large
        scale = np.max(np.abs(x_local))
        if scale > 1.0:
            h = 1e-7 * scale
        
        J = np.zeros((2, 2))
        try:
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = F(x_local + e)
                J[:, j] = (Fx_e - F_local) / h
        except:
            continue
        
        # Check condition
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            continue
        
        max_broyden_iters = 30
        for it in range(max_broyden_iters):
            if time_left() < 1.0:
                break
            
            if norm_local <= tol:
                return x_local
            
            # Solve J * s = -F_local
            try:
                s = np.linalg.solve(J, -F_local)
            except np.linalg.LinAlgError:
                break
            
            # Step size control
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-12:
                break
            if s_norm > 10.0:
                s = s * 10.0 / s_norm
            
            # Line search: try to find alpha such that ||F(x + alpha*s)|| < ||F(x)||
            # We use a simple backtracking
            alpha = 1.0
            f_prev = norm_local ** 2
            accepted = False
            
            # Limit line search calls to save budget
            for _ls in range(10):
                x_try = x_local + alpha * s
                try:
                    F_try = F(x_try)
                    norm_try = np.linalg.norm(F_try)
                except:
                    alpha *= 0.5
                    continue
                
                if norm_try < best_norm:
                    best_norm = norm_try
                    best_x = x_try.copy()
                
                if norm_try <= tol:
                    return x_try
                
                # Accept if we make sufficient progress
                if norm_try ** 2 < f_prev * 0.5 + 1e-12:
                    # Update Broyden Jacobian
                    delta_F = F_try - F_local - J @ s
                    s_norm_sq = np.dot(s, s)
                    if s_norm_sq > 1e-16:
                        J = J + np.outer(delta_F, s) / s_norm_sq
                    
                    x_local = x_try
                    F_local = F_try
                    norm_local = norm_try
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        if best_norm <= tol:
            return best_x
    
    return best_x
# EVOLVE-BLOCK-END
