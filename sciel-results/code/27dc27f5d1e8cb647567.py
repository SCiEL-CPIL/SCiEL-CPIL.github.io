"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Strategy: Global Homotopy Path Following with Adaptive Step Control
    # 1. Define a homotopy H(t, x) = (1-t)*F(x) + t*(x - x_target) where x_target is a simple anchor (e.g., 0 or x0).
    # 2. Solve for x(t) starting from t=0 (where x=x_target is an exact root) and marching to t=1.
    # 3. Use a predictor-corrector scheme: 
    #    - Predictor: Linear extrapolation of the path.
    #    - Corrector: Damped Newton steps on H(t, x) = 0.
    # 4. This provides a global convergence mechanism that is robust to local minima and singularities of F alone.
    # 5. If homotopy fails, fallback to random restarts with Damped Newton on F.
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
        
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False
            
    def finite_diff_jac(F, x, eps=1e-6):
        J = np.zeros((2, 2))
        Fx = F(x)
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                J[:, j] = (F(x + e) - Fx) / eps
            except Exception:
                raise
        return J
        
    def newton_damped(F, x0, max_iter=50, tol=1e-12, jac_func=None):
        x = np.array(x0, dtype=float)
        best_x = x.copy()
        best_norm = np.inf
        
        for i in range(max_iter):
            if time_left() < 0.5:
                break
                
            try:
                Fx = F(x)
            except Exception:
                break
                
            if not np.all(np.isfinite(Fx)):
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            if current_norm <= tol:
                return x
                
            try:
                if jac_func:
                    J = jac_func(x)
                else:
                    J = finite_diff_jac(F, x)
            except Exception:
                break
                
            if not np.all(np.isfinite(J)):
                break
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
                
            if not np.all(np.isfinite(dx)):
                break
                
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                except Exception:
                    continue
                    
                if not np.all(np.isfinite(F_new)):
                    continue
                    
                new_norm = norm(F_new)
                if new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break
                    
            if not step_found:
                break
                
            if norm(dx) < 1e-10:
                break
                
        return best_x

    def homotopy_solver(F, x0, max_iter=100):
        x_target = np.array([0.0, 0.0])
        t = 0.0
        x = x_target.copy()
        
        # Define H(t, x) and its Jacobian
        def H(t, x):
            return (1 - t) * F(x) + t * (x - x_target)
            
        def J_H(t, x):
            # J_H = (1-t) * J_F(x) + t * I
            try:
                J_F = finite_diff_jac(F, x)
            except Exception:
                return None
            return (1 - t) * J_F + t * np.eye(2)
        
        # We want to march from t=0 to t=1.
        # At t=0, H(0, x) = F(x). We need x such that F(x)=0? 
        # Wait, the standard homotopy is H(t, x) = (1-t)G(x) + tF(x) where G has a known root.
        # Let G(x) = x - x_target. Root is x_target.
        # H(t, x) = (1-t)(x - x_target) + tF(x).
        # At t=0, x = x_target.
        # At t=1, F(x) = 0.
        
        def H_new(t, x):
            return (1 - t) * (x - x_target) + t * F(x)
            
        def J_H_new(t, x):
            try:
                J_F = finite_diff_jac(F, x)
            except Exception:
                return None
            return (1 - t) * np.eye(2) + t * J_F
            
        x = x_target.copy()
        dt = 0.1
        prev_x = x.copy()
        
        for i in range(max_iter):
            if time_left() < 1.0:
                break
                
            t_next = min(1.0, t + dt)
            
            # Predictor: Linear extrapolation
            # dx/dt approx = x - prev_x
            dx_dt = x - prev_x
            x_pred = x + dx_dt * (t_next - t)
            
            # Corrector: Newton on H(t_next, x) = 0
            x_corr = x_pred.copy()
            success = False
            
            for k in range(10):
                if time_left() < 0.5:
                    break
                try:
                    H_val = H_new(t_next, x_corr)
                except Exception:
                    break
                    
                if not np.all(np.isfinite(H_val)):
                    break
                    
                if norm(H_val) < 1e-10:
                    success = True
                    break
                    
                try:
                    J_val = J_H_new(t_next, x_corr)
                except Exception:
                    break
                    
                if J_val is None or not np.all(np.isfinite(J_val)):
                    break
                    
                try:
                    dx = np.linalg.solve(J_val, -H_val)
                except np.linalg.LinAlgError:
                    break
                    
                if not np.all(np.isfinite(dx)):
                    break
                    
                x_corr = x_corr + dx
                
            if success:
                prev_x = x
                x = x_corr
                t = t_next
                # Increase dt if we succeeded easily? Keep it simple for now.
                if norm(x - x_pred) < 1e-6:
                    dt = min(0.5, dt * 1.5)
                else:
                    dt = 0.1
            else:
                # Failed to correct. Try smaller step.
                dt = max(0.01, dt * 0.5)
                if dt < 0.001:
                    break
        
        return x

    # Phase 1: Try Homotopy
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    x_hom = homotopy_solver(F, x0)
    
    if is_good(x_hom):
        return x_hom
        
    # Phase 2: Fallback to Random Restarts with Damped Newton
    x_best = x_hom
    best_norm = np.inf
    
    try:
        F_best = F(x_best)
        if np.all(np.isfinite(F_best)):
            best_norm = norm(F_best)
    except Exception:
        pass
        
    candidates = []
    scale = max(1.0, norm(x0))
    
    candidates.append(np.array([0.0, 0.0]))
    candidates.append(x0 * 0.5)
    candidates.append(x0 * 2.0)
    
    rng = np.random.default_rng(42)
    for r in [scale * 0.1, scale * 0.5, scale * 1.0, scale * 2.0]:
        for _ in range(5):
            u = rng.normal(size=2)
            u = u / norm(u)
            p = x0 + r * u
            candidates.append(p)
            
    for c in candidates:
        if time_left() < 1.0:
            break
            
        if not np.all(np.isfinite(c)):
            continue
            
        x_try = newton_damped(F, c)
        
        if is_good(x_try):
            return x_try
            
        try:
            F_try = F(x_try)
            if np.all(np.isfinite(F_try)):
                n_try = norm(F_try)
                if n_try < best_norm:
                    best_norm = n_try
                    x_best = x_try
        except Exception:
            pass
            
    return x_best
# EVOLVE-BLOCK-END
