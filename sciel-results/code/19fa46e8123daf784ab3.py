"""Root finder for 3 fixed 2D systems using Powell's secant method with Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Powell's method (secant-based quasi-Newton) which maintains the Jacobian
    inverse directly, requiring only 1 F-call per iteration after initial setup.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 140.0
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    def compute_jacobian(x_pt, Fx_pt):
        """Compute Jacobian via forward differences."""
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_pt + e)
            J[:, j] = (Fx_e - Fx_pt) / h
        return J
    
    # Track best solution
    best_x = x.copy()
    best_norm_sq = float('inf')
    
    # Initial evaluation
    try:
        Fx0 = F(x)
        best_norm_sq = float(np.dot(Fx0, Fx0))
    except:
        return np.zeros(2)
    
    if best_norm_sq <= tol_sq:
        return best_x
    
    # Powell's method: secant-based quasi-Newton for 2D
    def powell_solve(x_start, Fx_start, max_iters=40):
        """
        Powell's method for 2D systems.
        Maintains H = J^{-1} directly using the secant equation.
        Requires exactly 1 F-call per iteration after initial setup.
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True
        
        # Initial Jacobian estimate
        J = compute_jacobian(x_cur, Fx_cur)
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            return x_cur, Fx_cur, False
        
        # Invert J to get H
        H = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det
        
        # Store previous state for secant update
        x_prev = x_cur.copy()
        Fx_prev = Fx_cur.copy()
        
        for it in range(max_iters):
            if time_left() < 2.0:
                break
            
            # Compute step: s = -H * F(x)
            s = -H @ Fx_cur
            s_norm = np.linalg.norm(s)
            
            if s_norm < 1e-15:
                break
            
            # Limit step size for stability
            max_step = 100.0
            if s_norm > max_step:
                s = s * (max_step / s_norm)
                s_norm = max_step
            
            # Try full step first
            x_try = x_cur + s
            Fx_try = F(x_try)
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            if norm_sq_try <= tol_sq:
                return x_try, Fx_try, True
            
            # If step didn't reduce norm, try damped step
            if norm_sq_try > norm_sq_cur:
                # Damped step
                alpha = 0.5
                for _ in range(10):
                    x_try = x_cur + alpha * s
                    Fx_try = F(x_try)
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    if norm_sq_try <= norm_sq_cur:
                        break
                    alpha *= 0.5
                
                # If still not improving, try smaller step
                if norm_sq_try > norm_sq_cur:
                    x_try = x_cur + 0.1 * s
                    Fx_try = F(x_try)
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    if norm_sq_try > norm_sq_cur:
                        break
            
            # Secant update for H (Powell's method)
            # s = x_new - x_old, y = F(x_new) - F(x_old)
            s_vec = x_try - x_prev
            y_vec = Fx_try - Fx_prev
            
            s_norm_sq = float(np.dot(s_vec, s_vec))
            if s_norm_sq > 1e-20:
                # H_new = H + (s - H*y) * s^T / (s^T * s)
                Hy = H @ y_vec
                correction = (s_vec - Hy) * np.outer(s_vec, s_vec) / s_norm_sq
                H = H + correction
            
            # Update state
            x_prev = x_cur.copy()
            Fx_prev = Fx_cur.copy()
            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
        
        return x_cur, Fx_cur, norm_sq_cur <= tol_sq
    
    # Newton polish with fresh Jacobians
    def newton_polish(x_start, Fx_start, max_iters=15):
        """Newton's method with fresh Jacobian for high-precision convergence."""
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True
        
        for it in range(max_iters):
            if time_left() < 1.0:
                break
            
            J = compute_jacobian(x_cur, Fx_cur)
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-12:
                break
            
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            max_step = 50.0
            if s_norm > max_step:
                s = s * (max_step / s_norm)
            
            x_try = x_cur + s
            Fx_try = F(x_try)
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            if norm_sq_try <= tol_sq:
                return x_try, Fx_try, True
            
            # Check for improvement
            if norm_sq_try > norm_sq_cur:
                # Damped step
                alpha = 0.5
                for _ in range(10):
                    x_try = x_cur + alpha * s
                    Fx_try = F(x_try)
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    if norm_sq_try <= norm_sq_cur:
                        break
                    alpha *= 0.5
                
                if norm_sq_try > norm_sq_cur:
                    break
            
            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
        
        return x_cur, Fx_cur, norm_sq_cur <= tol_sq
    
    # Phase 1: Powell's method from x0
    x_sol, Fx_sol, converged = powell_solve(x, Fx0, max_iters=40)
    if converged:
        return x_sol
    
    if float(np.dot(Fx_sol, Fx_sol)) < best_norm_sq:
        best_norm_sq = float(np.dot(Fx_sol, Fx_sol))
        best_x = x_sol.copy()
    
    # Phase 2: Newton polish from best point
    x_polished, Fx_polished, converged = newton_polish(best_x, Fx_sol, max_iters=15)
    if converged:
        return x_polished
    
    if float(np.dot(Fx_polished, Fx_polished)) < best_norm_sq:
        best_norm_sq = float(np.dot(Fx_polished, Fx_polished))
        best_x = x_polished.copy()
    
    # Phase 3: Multi-start with Powell's method
    np.random.seed(42)
    max_restarts = 15
    
    for restart in range(max_restarts):
        if time_left() < 5.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x
        
        # Generate starting point
        if restart % 3 == 0:
            # Random in box around x0
            x_start = x + np.random.uniform(-10, 10, size=2)
        elif restart % 3 == 1:
            # Ring around best_x with growing radius
            radius = 1.0 + 2.0 * (restart // 3)
            angle = np.random.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Small perturbation around best_x
            x_start = best_x + np.random.randn(2) * 0.5
        
        if not np.all(np.isfinite(x_start)):
            continue
        
        try:
            Fx_start = F(x_start)
        except:
            continue
        
        norm_sq_start = float(np.dot(Fx_start, Fx_start))
        if norm_sq_start < best_norm_sq:
            best_norm_sq = norm_sq_start
            best_x = x_start.copy()
        
        if norm_sq_start <= tol_sq:
            return x_start
        
        # Powell's method from new start
        x_sol, Fx_sol, converged = powell_solve(x_start, Fx_start, max_iters=30)
        if converged:
            return x_sol
        
        norm_sq_sol = float(np.dot(Fx_sol, Fx_sol))
        if norm_sq_sol < best_norm_sq:
            best_norm_sq = norm_sq_sol
            best_x = x_sol.copy()
        
        # Newton polish
        x_polished, Fx_polished, converged = newton_polish(best_x, Fx_sol, max_iters=10)
        if converged:
            return x_polished
        
        norm_sq_polished = float(np.dot(Fx_polished, Fx_polished))
        if norm_sq_polished < best_norm_sq:
            best_norm_sq = norm_sq_polished
            best_x = x_polished.copy()
    
    return best_x
# EVOLVE-BLOCK-END
