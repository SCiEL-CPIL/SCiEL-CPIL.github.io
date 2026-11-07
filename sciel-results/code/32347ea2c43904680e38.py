"""Root finder for 3 fixed 2-D systems: multi-strategy with careful budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with rank-1 update and trust-region damping.
    2. If Broyden fails, try damped Newton with finite-difference Jacobian.
    3. If that fails, try a global search (grid/random restarts).
    
    Budget:
    - Time limit: 110s internal deadline (150s hard limit).
    - Call limit: < 500 calls per system.
    """
    t_start = time.time()
    deadline = t_start + 110.0
    x = np.array(x0, dtype=float)
    
    def time_ok():
        return time.time() < deadline
    
    # Initial evaluation
    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float)
    if not (Fx.shape == (2,) and np.all(np.isfinite(Fx))):
        return x
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
    
    # Strategy 1: Broyden's method
    result = _broyden(F, x, Fx, best_x, best_norm, time_ok)
    if result[1] <= 1e-8:
        return result[0]
    best_x, best_norm = result
    
    if not time_ok():
        return best_x
    
    # Strategy 2: Damped Newton with finite-difference Jacobian
    result = _damped_newton(F, best_x, best_norm, time_ok)
    if result[1] <= 1e-8:
        return result[0]
    best_x, best_norm = result
    
    if not time_ok():
        return best_x
    
    # Strategy 3: Global search with random restarts
    result = _global_search(F, best_x, best_norm, time_ok)
    if result[1] <= 1e-8:
        return result[0]
    best_x, best_norm = result
    
    return best_x


def _broyden(F, x0, Fx0, best_x, best_norm, time_ok):
    """Broyden's method with rank-1 update and trust-region damping."""
    x = np.array(x0, dtype=float)
    Fx = np.array(Fx0, dtype=float)
    
    # Initial Jacobian: use finite differences if we have budget
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = F(x + e)
        Fx_e = np.asarray(Fx_e, dtype=float)
        if Fx_e.shape == (2,) and np.all(np.isfinite(Fx_e)):
            J[:, j] = (Fx_e - Fx) / h
        else:
            J[:, j] = np.eye(2)[:, j]  # fallback to identity column
    
    max_iter = 50
    for it in range(max_iter):
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use steepest descent direction
            dx = -Fx
            dx_norm = np.linalg.norm(dx)
            if dx_norm < 1e-14:
                break
            dx = dx / dx_norm * 0.1
        
        # Trust-region damping
        step_norm = np.linalg.norm(dx)
        max_step = 1.0
        if step_norm > max_step:
            dx = dx * (max_step / step_norm)
            step_norm = max_step
        
        x_new = x + dx
        Fx_new = F(x_new)
        Fx_new = np.asarray(Fx_new, dtype=float)
        if not (Fx_new.shape == (2,) and np.all(np.isfinite(Fx_new))):
            break
        norm_new = np.linalg.norm(Fx_new)
        
        # Update best
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
        
        # Check convergence
        if norm_new <= 1e-8:
            x = x_new
            Fx = Fx_new
            break
        
        # Broyden rank-1 update
        y = Fx_new - Fx
        denom = dx @ dx
        if denom > 1e-14:
            J = J + np.outer(y - J @ dx, dx) / denom
        
        # Accept step if it reduces the residual
        if norm_new < best_norm:
            x = x_new
            Fx = Fx_new
        else:
            # Reject step, try smaller step
            dx = dx * 0.5
            x_new = x + dx
            Fx_new = F(x_new)
            Fx_new = np.asarray(Fx_new, dtype=float)
            if not (Fx_new.shape == (2,) and np.all(np.isfinite(Fx_new))):
                break
            norm_new = np.linalg.norm(Fx_new)
            
            if norm_new < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = norm_new
                if norm_new <= 1e-8:
                    break
                # Update Broyden
                y = Fx_new - Fx
                denom = dx @ dx
                if denom > 1e-14:
                    J = J + np.outer(y - J @ dx, dx) / denom
            else:
                # Stagnation: break and try fallback
                break
    
    return best_x, best_norm


def _damped_newton(F, x0, best_norm, time_ok):
    """Damped Newton with finite-difference Jacobian and line search."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    Fx = np.asarray(Fx, dtype=float)
    if not (Fx.shape == (2,) and np.all(np.isfinite(Fx))):
        return x, best_norm
    
    best_x = x.copy()
    best_norm = min(best_norm, np.linalg.norm(Fx))
    
    max_iter = 30
    for it in range(max_iter):
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        # Finite-difference Jacobian
        h = 1e-6
        J_fd = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x + e)
            Fx_e = np.asarray(Fx_e, dtype=float)
            if Fx_e.shape == (2,) and np.all(np.isfinite(Fx_e)):
                J_fd[:, j] = (Fx_e - Fx) / h
            else:
                J_fd[:, j] = np.eye(2)[:, j]
        
        try:
            dx = np.linalg.solve(J_fd, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Damped Newton: line search
        alpha = 1.0
        for _ in range(15):
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            Fx_new = np.asarray(Fx_new, dtype=float)
            if not (Fx_new.shape == (2,) and np.all(np.isfinite(Fx_new))):
                alpha *= 0.5
                continue
            norm_new = np.linalg.norm(Fx_new)
            
            if norm_new < best_norm:
                x = x_new
                Fx = Fx_new
                best_x = x.copy()
                best_norm = norm_new
                if norm_new <= 1e-8:
                    break
                break
            else:
                alpha *= 0.5
        
        if best_norm <= 1e-8:
            break
    
    return best_x, best_norm


def _global_search(F, x0, best_norm, time_ok):
    """Global search with random restarts and local refinement."""
    x0 = np.array(x0, dtype=float)
    best_x = x0.copy()
    best_norm = best_norm
    
    # Try several random starting points
    np.random.seed(42)
    for trial in range(20):
        if not time_ok():
            break
        if best_norm <= 1e-8:
            break
        
        # Random starting point near x0
        scale = 1.0 * (1 + trial * 0.5)
        x = x0 + np.random.randn(2) * scale
        
        # Quick Broyden from this starting point
        result = _broyden(F, x, F(x), best_x, best_norm, time_ok)
        if result[1] < best_norm:
            best_x, best_norm = result
        if best_norm <= 1e-8:
            break
    
    return best_x, best_norm
# EVOLVE-BLOCK-END
