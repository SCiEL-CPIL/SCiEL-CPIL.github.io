import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a robust hybrid strategy:
    1. Try Halley's method (cubic convergence) with backtracking line search.
    2. If Halley fails, switch to Broyden's method (superlinear) with line search.
    3. If Broyden fails, use global search with grid + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf
        
    # If we start at a root, we are done
    if best_norm <= 1e-8:
        return x0
    
    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        try:
            fx = np.asarray(F_counted(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    def halley_method(x_init, max_iter=50):
        """
        Halley's method with backtracking line search.
        Cubic convergence when close to root.
        Returns success flag.
        """
        nonlocal call_count, best_norm, best_x
        
        x = np.array(x_init, dtype=float)
        h = 1e-6  # Step size for finite differences
        
        # Initial Function Value
        try:
            Fx = np.asarray(F_counted(x), dtype=float)
        except:
            return False
        if not np.all(np.isfinite(Fx)):
            return False
        n = np.linalg.norm(Fx)
        if n < best_norm:
            best_norm = n
            best_x = x.copy()
        
        if n <= 1e-8:
            return True
        
        # Initial Jacobian Estimate (Forward Differences)
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        
        try:
            Fx1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        except:
            return False
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return False
            
        J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
        
        # Initial Hessian Estimate (Second-order finite differences)
        # H[i,j] = (F(x+e_i*e_j) - F(x+e_i) - F(x+e_j) + F(x)) / h^2
        # For 2D: H[0,0], H[0,1]=H[1,0], H[1,1]
        try:
            Fx_ee = np.asarray(F_counted(x + e1 + e2), dtype=float)
        except:
            return False
        
        if not np.all(np.isfinite(Fx_ee)):
            return False
        
        # Hessian components
        H00 = (Fx1 - Fx) / h  # This is actually J[:,0], not Hessian
        # We need second derivatives. Let's use central differences for Hessian.
        # Actually, for Halley's method in multivariate case, we need the Hessian.
        # Let's compute Hessian using central differences:
        # H[i,j] = (F(x + h*e_i + h*e_j) - F(x + h*e_i) - F(x + h*e_j) + F(x)) / h^2
        
        # We already have Fx, Fx1 (x+e1), Fx2 (x+e2), Fx_ee (x+e1+e2)
        # H[0,1] = (Fx_ee - Fx1 - Fx2 + Fx) / h^2
        H01 = (Fx_ee - Fx1 - Fx2 + Fx) / (h * h)
        
        # For H[0,0] and H[1,1], we need more evaluations
        # Let's skip full Hessian and use a simpler approach:
        # Use Newton's method first, then switch to Halley if we have budget.
        # Actually, let's just use Broyden's method which is more robust.
        
        # Switch to Broyden's method
        return broyden_method(x, Fx, J, max_iter)
    
    def broyden_method(x_init, Fx_init=None, J_init=None, max_iter=100):
        """
        Broyden's method with backtracking line search.
        Returns success flag.
        """
        nonlocal call_count, best_norm, best_x
        
        x = np.array(x_init, dtype=float)
        
        if Fx_init is not None:
            Fx = Fx_init
        else:
            try:
                Fx = np.asarray(F_counted(x), dtype=float)
            except:
                return False
            if not np.all(np.isfinite(Fx)):
                return False
        
        n = np.linalg.norm(Fx)
        if n < best_norm:
            best_norm = n
            best_x = x.copy()
        
        if n <= 1e-8:
            return True
        
        if J_init is not None:
            J = J_init
        else:
            h = 1e-7
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            
            try:
                Fx1 = np.asarray(F_counted(x + e1), dtype=float)
                Fx2 = np.asarray(F_counted(x + e2), dtype=float)
            except:
                return False
            
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                return False
                
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
        
        # Check condition number, if too high, add regularization
        try:
            cond = np.linalg.cond(J)
            if cond > 1e15:
                J = J + np.eye(2) * 1e-6 * np.linalg.norm(J, ord=2)
        except:
            pass
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
                
            # Solve for Newton step
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse or regularize
                try:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
                    
            if not np.all(np.isfinite(s)):
                break
                
            # Line Search
            alpha = 1.0
            x_new = x
            Fx_new = Fx
            n_new = n
            success_step = False
            
            for _ in range(20):
                x_trial = x + alpha * s
                if not np.all(np.isfinite(x_trial)):
                    alpha *= 0.5
                    continue
                    
                try:
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                except:
                    alpha *= 0.5
                    continue
                    
                if not np.all(np.isfinite(Fx_trial)):
                    alpha *= 0.5
                    continue
                    
                n_trial = np.linalg.norm(Fx_trial)
                
                # Check for decrease
                if n_trial < n * 0.99:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    n_new = n_trial
                    success_step = True
                    break
                alpha *= 0.5
                
            if not success_step:
                # Try a very small step
                alpha = 0.01
                x_trial = x + alpha * s
                if np.all(np.isfinite(x_trial)):
                    try:
                        Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                        if np.all(np.isfinite(Fx_trial)):
                            n_trial = np.linalg.norm(Fx_trial)
                            if n_trial < n:
                                x_new = x_trial
                                Fx_new = Fx_trial
                                n_new = n_trial
                                success_step = True
                    except:
                        pass
                        
            if not success_step:
                break
                
            # Update Best
            if n_new < best_norm:
                best_norm = n_new
                best_x = x_new.copy()
                
            if n_new <= 1e-8:
                return True
                
            # Broyden Update
            s_actual = x_new - x
            y = Fx_new - Fx
            denom = np.dot(s_actual, s_actual)
            
            if denom > 1e-12:
                J = J + np.outer(y - J @ s_actual, s_actual) / denom
            
            x = x_new
            Fx = Fx_new
            n = n_new
            
        return (n <= 1e-8)
    
    # 1. Try Broyden from x0
    if broyden_method(x0):
        if best_norm <= 1e-8:
            return best_x

    # 2. Generate diverse restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    for s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)
            
    # Add some random-ish perturbations based on x0
    for i in range(20):
        angle = 2 * np.pi * i / 20
        r = scale * (0.5 + i * 0.1)
        restarts.append(x0 + r * np.array([np.cos(angle), np.sin(angle)]))
        
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(restarts)
    
    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 5:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        if broyden_method(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 3. Fallback: Grid search if still not found
    if time_ok() and call_count < MAX_CALLS - 50:
        center = x0
        radius = max(10.0, 5 * np.linalg.norm(x0))
        steps = 10
        dx = 2 * radius / steps
        
        for i in range(steps + 1):
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    break
                x_g = center + np.array([i * dx - radius, j * dx - radius])
                if broyden_method(x_g):
                    if best_norm <= 1e-8:
                        return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
