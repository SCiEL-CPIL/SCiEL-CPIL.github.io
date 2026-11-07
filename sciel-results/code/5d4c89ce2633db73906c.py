"""Root finder for 3 fixed 2D systems using homotopy continuation with Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using a homotopy/continuation
    approach combined with Broyden's method.
    
    Strategy:
    1. First try direct Broyden from x0
    2. If that fails, use homotopy: track root of F_t(x) = t*F(x) + (1-t)*(x - x0)
       from t=0 (trivial root at x0) to t=1 (original problem)
    3. Multi-start fallback with diverse initial points
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
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
    
    def broyden_solve(x_start, Fx_start, max_iters=40):
        """
        Broyden's good method with merit function line search.
        Returns (x, Fx, converged, iters_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        
        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, 0
        
        # Compute initial Jacobian via forward differences
        h = 1e-7
        scale = np.max(np.abs(x_cur))
        if scale > 1.0:
            h = 1e-7 * scale
        
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_cur + e)
            J[:, j] = (Fx_e - Fx_cur) / h
        
        # Check if Jacobian is well-conditioned
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det) < 1e-12:
            return x_cur, Fx_cur, False, 0
        
        iters = 0
        max_step = 50.0
        
        for it in range(max_iters):
            iters = it + 1
            if time_left() < 1.0:
                break
            
            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break
            
            # Limit step size
            if s_norm > max_step:
                s = s * (max_step / s_norm)
                s_norm = max_step
            
            # Line search on merit function phi(x) = 0.5 * ||F(x)||^2
            alpha = 1.0
            phi_prev = 0.5 * norm_sq_cur
            slope = float(np.dot(Fx_cur, J @ s))
            
            if slope >= 0:
                s = -J.T @ Fx_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                if s_norm > max_step:
                    s = s * (max_step / s_norm)
                slope = -float(np.dot(s, J.T @ Fx_cur))
                if slope >= 0:
                    break
            
            accepted = False
            c1 = 1e-4
            
            for _ls in range(30):
                x_try = x_cur + alpha * s
                Fx_try = F(x_try)
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                phi_try = 0.5 * norm_sq_try
                
                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, iters
                
                if phi_try <= phi_prev + c1 * alpha * slope:
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                break
            
            # Update Broyden
            delta_F = Fx_try - Fx_cur - J @ s
            s_norm_sq = np.dot(s, s)
            if s_norm_sq > 1e-16:
                J = J + np.outer(delta_F, s) / s_norm_sq
            
            # Update state
            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try
        
        return x_cur, Fx_cur, False, iters
    
    # Step 1: Try direct Broyden from x0
    x_sol, Fx_sol, converged, _ = broyden_solve(x, Fx0, max_iters=40)
    if converged:
        return x_sol
    
    if float(np.dot(Fx_sol, Fx_sol)) < best_norm_sq:
        best_norm_sq = float(np.dot(Fx_sol, Fx_sol))
        best_x = x_sol.copy()
    
    # Step 2: Homotopy continuation
    # F_t(x) = t*F(x) + (1-t)*(x - x0)
    # At t=0, root is at x0. We track the root as t increases to 1.
    
    def homotopy_F(x_h, t):
        """Evaluate F_t(x) = t*F(x) + (1-t)*(x - x0)"""
        try:
            Fx = F(x_h)
        except:
            return None
        return t * Fx + (1 - t) * (x_h - x)
    
    def homotopy_jacobian(x_h, t, J_F):
        """Jacobian of F_t: t*J_F + (1-t)*I"""
        return t * J_F + (1 - t) * np.eye(2)
    
    def compute_jacobian_F(x_h):
        """Compute Jacobian of F via forward differences"""
        h = 1e-7
        scale = np.max(np.abs(x_h))
        if scale > 1.0:
            h = 1e-7 * scale
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = F(x_h + e)
            J[:, j] = (Fx_e - F(x_h)) / h
        return J
    
    # Homotopy solver: track root from t=0 to t=1
    def homotopy_solve(x_init, t_start=0.0, t_end=1.0, n_steps=20, max_iters_per_step=20):
        """
        Track the root of F_t from t_start to t_end.
        Returns (x_final, F_final, converged).
        """
        x_cur = x_init.copy()
        t_cur = t_start
        
        # Initial Jacobian of F at x_cur
        try:
            J_F = compute_jacobian_F(x_cur)
        except:
            return x_cur, F(x_cur), False
        
        for step in range(n_steps):
            if time_left() < 1.0:
                break
            
            t_next = t_start + (step + 1) * (t_end - t_start) / n_steps
            dt = t_next - t_cur
            
            # Solve F_t(x) = 0 using Newton-like method with Broyden updates
            # We need to solve: t*F(x) + (1-t)*(x - x0) = 0
            
            # Evaluate at current x with new t
            Fx_cur = F(x_cur)
            F_t_cur = t_next * Fx_cur + (1 - t_next) * (x_cur - x)
            norm_sq_t = float(np.dot(F_t_cur, F_t_cur))
            
            if norm_sq_t <= tol_sq:
                # Check if this is also a root of the original F
                if t_next >= 0.99:
                    norm_sq_orig = float(np.dot(Fx_cur, Fx_cur))
                    if norm_sq_orig <= tol_sq:
                        return x_cur, Fx_cur, True
            elif norm_sq_t > 1e6:
                # Homotopy path diverged, try smaller step
                continue
            
            # Newton step for F_t
            J_t = t_next * J_F + (1 - t_next) * np.eye(2)
            
            try:
                s = np.linalg.solve(J_t, -F_t_cur)
            except np.linalg.LinAlgError:
                # Singular, try pseudo-inverse
                try:
                    s = np.linalg.lstsq(J_t, -F_t_cur, rcond=None)[0]
                except:
                    break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                # Already at root of F_t
                if t_next >= 0.99:
                    norm_sq_orig = float(np.dot(Fx_cur, Fx_cur))
                    if norm_sq_orig <= tol_sq:
                        return x_cur, Fx_cur, True
                continue
            
            # Limit step
            if s_norm > 10.0:
                s = s * (10.0 / s_norm)
                s_norm = 10.0
            
            # Line search on ||F_t(x)||^2
            alpha = 1.0
            phi_prev = 0.5 * norm_sq_t
            slope = float(np.dot(F_t_cur, J_t @ s))
            
            if slope >= 0:
                # Use steepest descent
                s = -J_t.T @ F_t_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                if s_norm > 10.0:
                    s = s * (10.0 / s_norm)
                slope = -float(np.dot(s, J_t.T @ F_t_cur))
                if slope >= 0:
                    break
            
            accepted = False
            c1 = 1e-4
            
            for _ls in range(20):
                x_try = x_cur + alpha * s
                Fx_try = F(x_try)
                F_t_try = t_next * Fx_try + (1 - t_next) * (x_try - x)
                norm_sq_try = float(np.dot(F_t_try, F_t_try))
                phi_try = 0.5 * norm_sq_try
                
                if phi_try <= phi_prev + c1 * alpha * slope:
                    accepted = True
                    break
                
                alpha *= 0.5
            
            if not accepted:
                # Try smaller homotopy step
                continue
            
            # Update
            x_cur = x_try
            Fx_cur = Fx_try
            
            # Update Jacobian of F
            try:
                J_F_new = compute_jacobian_F(x_cur)
                J_F = J_F_new
            except:
                pass
            
            t_cur = t_next
        
        # Final check
        Fx_final = F(x_cur)
        norm_sq_final = float(np.dot(Fx_final, Fx_final))
        converged = norm_sq_final <= tol_sq
        return x_cur, Fx_final, converged
    
    # Try homotopy from x0
    if time_left() > 5.0:
        x_h, Fx_h, conv_h = homotopy_solve(x, t_start=0.0, t_end=1.0, n_steps=20, max_iters_per_step=20)
        if conv_h:
            return x_h
        if float(np.dot(Fx_h, Fx_h)) < best_norm_sq:
            best_norm_sq = float(np.dot(Fx_h, Fx_h))
            best_x = x_h.copy()
    
    # Step 3: Multi-start with diverse initial points
    np.random.seed(42)
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time_left() < 3.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x
        
        # Generate starting point
        if restart % 4 == 0:
            # Random in box around x0
            x_start = x + np.random.uniform(-10, 10, size=2)
        elif restart % 4 == 1:
            # Ring around best_x with growing radius
            radius = 1.0 + 2.0 * (restart // 4)
            angle = np.random.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 4 == 2:
            # Small perturbation around best_x
            x_start = best_x + np.random.randn(2) * 0.5
        else:
            # Try homotopy from a perturbed start
            x_start = x + np.random.randn(2) * 2.0
        
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
        
        # Try direct Broyden
        x_sol, Fx_sol, converged, _ = broyden_solve(x_start, Fx_start, max_iters=35)
        if converged:
            return x_sol
        
        norm_sq_sol = float(np.dot(Fx_sol, Fx_sol))
        if norm_sq_sol < best_norm_sq:
            best_norm_sq = norm_sq_sol
            best_x = x_sol.copy()
        
        # If not converged, try homotopy from this point
        if not converged and time_left() > 5.0 and restart % 2 == 0:
            x_h, Fx_h, conv_h = homotopy_solve(x_start, t_start=0.0, t_end=1.0, n_steps=15, max_iters_per_step=15)
            if conv_h:
                return x_h
            norm_sq_h = float(np.dot(Fx_h, Fx_h))
            if norm_sq_h < best_norm_sq:
                best_norm_sq = norm_sq_h
                best_x = x_h.copy()
    
    return best_x
# EVOLVE-BLOCK-END
