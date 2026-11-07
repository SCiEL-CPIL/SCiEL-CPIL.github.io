import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt, Fx_pt):
        """Compute Jacobian via forward finite differences (2 calls)."""
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            J = np.array([(Fx1 - Fx_pt) / h, (Fx2 - Fx_pt) / h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    # Initial Jacobian estimate
    J = compute_jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    # Homotopy parameters
    t = 0.0
    dt = 0.1
    max_homotopy_steps = 100
    
    # We will track the root of H(x, t) = t*F(x) + (1-t)*(x - x0)
    # At t=0, x=x0 is the root.
    # We step t towards 1.
    
    # If we are already very close to a root of F, we might skip homotopy
    # but let's try homotopy first as it's the structural change.
    
    homotopy_success = False
    
    for step in range(max_homotopy_steps):
        if not time_ok() or call_count > 450:
            break
            
        # Target t
        t_target = min(1.0, t + dt)
        
        # Solve H(x, t_target) = 0 starting from current x
        # Use Newton iterations on H
        converged = False
        for _ in range(10):
            if call_count > 480:
                break
                
            # Evaluate F(x)
            try:
                Fx_curr = F_count(x)
            except:
                break
                
            if not np.all(np.isfinite(Fx_curr)):
                break
                
            # Evaluate H
            H = t_target * Fx_curr + (1.0 - t_target) * (x - x0)
            
            # Check if H is small enough (relative to step size)
            # For t close to 1, we care about F. For t small, we care about H.
            # A good metric: norm(H) < 1e-8 * (1 + norm(x))
            # Or simply check if F is small if t is close to 1.
            
            # Jacobian of H: J_H = t_target * J + (1 - t_target) * I
            J_H = t_target * J + (1.0 - t_target) * np.eye(2)
            
            try:
                # Solve J_H * dx = -H
                dx = np.linalg.solve(J_H, -H)
            except np.linalg.LinAlgError:
                # If singular, regularize
                J_H_reg = J_H + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_H_reg, -H)
                except:
                    break
                    
            x_try = x + dx
            
            # Check if step is too large?
            if np.linalg.norm(dx) > 10.0:
                dx = dx * (10.0 / np.linalg.norm(dx))
                x_try = x + dx
                
            try:
                Fx_try = F_count(x_try)
            except:
                break
                
            if not np.all(np.isfinite(Fx_try)):
                break
                
            # Update Broyden for J (based on F)
            # J_new = J + (dF - J*dx) * dx^T / (dx^T dx)
            dF = Fx_try - Fx_curr
            Jdx = J @ dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(dx, dx))
            
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, dx) / dx_norm_sq
                
            # Accept step if H decreases or is small
            H_new = t_target * Fx_try + (1.0 - t_target) * (x_try - x0)
            norm_H = np.linalg.norm(H)
            norm_H_new = np.linalg.norm(H_new)
            
            if norm_H_new < norm_H or norm_H_new < 1e-10:
                x = x_try
                Fx = Fx_try
                if norm_H_new < 1e-10:
                    converged = True
                    break
            else:
                # If not decreasing, maybe step is too big?
                # For homotopy, we usually just accept the Newton step 
                # and rely on the outer loop to adjust dt if it fails.
                # But here we are inside the inner Newton loop for a fixed t_target.
                # If Newton on H doesn't converge, we fail this t_step.
                pass

        if converged:
            # Successfully tracked to t_target
            t = t_target
            # Increase dt slightly for next step
            dt = min(0.5, dt * 1.2)
            
            # Check if we reached t=1 and F is small
            if t >= 0.999:
                # Polish the root
                # We are at x such that F(x) should be small.
                # Run a few standard Newton steps on F
                for _ in range(20):
                    if call_count > 490:
                        break
                    try:
                        Fx_curr = F_count(x)
                    except:
                        break
                    if not np.all(np.isfinite(Fx_curr)):
                        break
                    norm_sq_curr = float(np.dot(Fx_curr, Fx_curr))
                    if norm_sq_curr <= tol_sq:
                        return x
                    if norm_sq_curr < best_norm_sq:
                        best_norm_sq = norm_sq_curr
                        best_x = x.copy()
                        
                    try:
                        dx = np.linalg.solve(J, -Fx_curr)
                    except:
                        J = J + 1e-6 * np.eye(2)
                        try:
                            dx = np.linalg.solve(J, -Fx_curr)
                        except:
                            break
                            
                    x_try = x + dx
                    try:
                        Fx_try = F_count(x_try)
                    except:
                        break
                    if not np.all(np.isfinite(Fx_try)):
                        break
                    norm_sq_try = float(np.dot(Fx_try, Fx_try))
                    
                    dF = Fx_try - Fx_curr
                    Jdx = J @ dx
                    residual = dF - Jdx
                    dx_norm_sq = float(np.dot(dx, dx))
                    if dx_norm_sq > 1e-30:
                        J = J + np.outer(residual, dx) / dx_norm_sq
                        
                    if norm_sq_try < norm_sq_curr:
                        x = x_try
                        Fx = Fx_try
                    else:
                        break
                        
                if best_norm_sq <= tol_sq:
                    return best_x
                break # Exit homotopy loop, we are at t=1
                
        else:
            # Failed to converge to t_target
            # Reduce step size
            dt = dt * 0.5
            if dt < 1e-4:
                # Homotopy stuck, fall back to standard Broyden on F
                break

    # Fallback: Standard Broyden from best_x
    if best_norm_sq > tol_sq and call_count < 480 and time_ok():
        x = best_x.copy()
        try:
            Fx = F_count(x)
        except:
            return best_x
        if not np.all(np.isfinite(Fx)):
            return best_x
        norm_sq = float(np.dot(Fx, Fx))
        
        # Recompute J if it's bad
        # J is already updated, but let's ensure it's good
        J_new = compute_jacobian_fd(x, Fx)
        if J_new is not None:
            J = J_new
            
        for _ in range(50):
            if call_count > 495 or not time_ok():
                break
            if norm_sq <= tol_sq:
                return x
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                J = J + 1e-6 * np.eye(2)
                try:
                    dx = np.linalg.solve(J, -Fx)
                except:
                    break
                    
            x_try = x + dx
            try:
                Fx_try = F_count(x_try)
            except:
                break
            if not np.all(np.isfinite(Fx_try)):
                break
            norm_sq_try = float(np.dot(Fx_try, Fx_try))
            
            dF = Fx_try - Fx
            Jdx = J @ dx
            residual = dF - Jdx
            dx_norm_sq = float(np.dot(dx, dx))
            if dx_norm_sq > 1e-30:
                J = J + np.outer(residual, dx) / dx_norm_sq
                
            if norm_sq_try < norm_sq:
                x = x_try
                Fx = Fx_try
                norm_sq = norm_sq_try
                if norm_sq < best_norm_sq:
                    best_norm_sq = norm_sq
                    best_x = x.copy()
            else:
                break

    return best_x
