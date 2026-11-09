# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    x0 = np.array(x0, dtype=float)
    best_x = x0.copy()
    best_norm = np.inf
    total_calls = 0

    def time_ok():
        return time.time() < deadline

    def compute_F(x):
        nonlocal total_calls
        if not time_ok():
            return None
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            total_calls += 1
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return None
            return Fx
        except Exception:
            return None

    def newton_solve(x_start):
        nonlocal best_x, best_norm, total_calls
        
        x = x_start.copy()
        Fx = compute_F(x)
        if Fx is None:
            return x, np.inf
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
            
        if norm_Fx <= 1e-8:
            return x, norm_Fx

        max_iter = 30
        h = 1e-5
        J = None

        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Compute Jacobian if not available or if we want to update it (Newton uses fresh J)
            # For 2D, finite difference is cheap.
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = compute_F(x + e)
                if Fx_pert is None:
                    return x, norm_Fx
                J[:, j] = (Fx_pert - Fx) / h
            
            # Solve J s = -Fx
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Regularization
                try:
                    s = np.linalg.solve(J + 1e-8 * np.eye(2), -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Line search
            alpha = 1.0
            found_step = False
            x_new = x
            Fx_new = Fx
            
            for _ in range(20):
                if not time_ok():
                    return x, norm_Fx
                    
                x_trial = x + alpha * s
                Fx_trial = compute_F(x_trial)
                if Fx_trial is None:
                    alpha *= 0.5
                    continue
                
                norm_trial = np.linalg.norm(Fx_trial)
                
                if norm_trial < norm_Fx:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    found_step = True
                    break
                
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            if not found_step:
                break
                
            # Update best
            if norm_trial < best_norm:
                best_norm = norm_trial
                best_x = x_new.copy()
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_trial
            
            if norm_Fx <= 1e-8:
                return x, norm_Fx
                
        return x, norm_Fx

    # 1. Try from x0
    x_sol, norm_sol = newton_solve(x0)
    if norm_sol <= 1e-8:
        return x_sol

    # 2. If failed, try restarts
    # Generate random points in a box around x0
    # Scale of the box: try 1.0, 0.1, 0.01
    scales = [1.0, 0.1, 0.01]
    np.random.seed(42) # Deterministic for reproducibility in eval? No, better to use system random or fixed seed for stability.
    
    for scale in scales:
        if not time_ok():
            break
        # Try a few random points
        for _ in range(5):
            if not time_ok():
                break
            # Random point in box [-scale, scale] around x0
            offset = np.random.uniform(-scale, scale, size=2)
            x_new_start = x0 + offset
            
            x_sol, norm_sol = newton_solve(x_new_start)
            if norm_sol <= 1e-8:
                return x_sol

    return best_x
# EVOLVE-BLOCK-END
