"""Root finder for 3 fixed 2D systems using Homotopy/Continuation Method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x):
    """Safely evaluate F at x, returning (Fx, norm_Fx, is_valid)."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = np.linalg.norm(Fx)
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _newton_step(F, x, J, max_iter=10, tol=1e-10, call_counter=None):
    """
    Perform a few Newton iterations using the provided Jacobian J.
    Updates J using Broyden's method to maintain accuracy without extra FD calls.
    Returns (x, J, success, norm_F).
    """
    x = np.array(x, dtype=float)
    J = np.array(J, dtype=float)
    
    Fx, norm_F, valid = _safe_eval(F, x)
    if call_counter is not None:
        call_counter[0] += 1
    if not valid:
        return x, J, False, float('inf')
        
    if norm_F <= tol:
        return x, J, True, norm_F

    for _ in range(max_iter):
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use least squares
            s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
        if not np.all(np.isfinite(s)):
            break
            
        # Simple damping if step is too large
        step_norm = np.linalg.norm(s)
        if step_norm > 10.0:
            s = s / step_norm * 10.0
            
        x_new = x + s
        F_new, norm_new, valid_new = _safe_eval(F, x_new)
        if call_counter is not None:
            call_counter[0] += 1
            
        if not valid_new:
            # Backtrack if invalid
            for alpha in [0.5, 0.25, 0.1]:
                x_try = x + alpha * s
                F_try, norm_try, valid_try = _safe_eval(F, x_try)
                if call_counter is not None:
                    call_counter[0] += 1
                if valid_try and norm_try < norm_F:
                    x_new = x_try
                    F_new = F_try
                    norm_new = norm_try
                    valid_new = True
                    break
            if not valid_new:
                break
                
        # Broyden Update
        y = F_new - Fx
        s_vec = x_new - x
        if np.linalg.norm(s_vec) > 1e-12:
            Js = J @ s_vec
            diff = y - Js
            J = J + np.outer(diff, s_vec) / (s_vec @ s_vec)
            
        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        if norm_F <= tol:
            return x, J, True, norm_F
            
        if np.linalg.norm(s_vec) < 1e-10:
            break
            
    return x, J, norm_F <= tol, norm_F

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    best_x = x0_arr
    best_norm = float('inf')
    call_counter = [0]
    max_calls = 450 # Leave buffer for safety
    
    # Homotopy path: F_t(x) = t * F(x) + (1 - t) * (x - x0)
    # Root at t=0 is x0. We track the root as t increases to 1.
    
    # We need an initial Jacobian for the homotopy function H(x, t) = t*F(x) + (1-t)*(x-x0)
    # dH/dx = t*J_F(x) + (1-t)*I
    # At t=0, J_H = I.
    
    # To get J_F(x0), we need 2 calls (forward diff) or we can start with I and refine.
    # Starting with I is risky if F is stiff, but homotopy is robust.
    # Let's estimate J_F(x0) with 2 calls to start the homotopy with a better Jacobian.
    
    h = 1e-5
    e0 = np.array([h, 0.0])
    e1 = np.array([0.0, h])
    
    Fx0, norm_Fx0, valid0 = _safe_eval(F, x0_arr)
    if call_counter is not None: call_counter[0] += 1
    
    if not valid0:
        # If F(x0) is invalid, try origin? Or just return x0?
        # The task says x0 is provided. If it's bad, we might need a fallback.
        # But usually x0 is a valid starting point.
        return x0_arr

    Fx_e0, _, v0 = _safe_eval(F, x0_arr + e0)
    if call_counter is not None: call_counter[0] += 1
    Fx_e1, _, v1 = _safe_eval(F, x0_arr + e1)
    if call_counter is not None: call_counter[0] += 1
    
    if v0 and v1:
        J_F = np.zeros((2, 2))
        J_F[:, 0] = (Fx_e0 - Fx0) / h
        J_F[:, 1] = (Fx_e1 - Fx0) / h
        if not np.all(np.isfinite(J_F)):
            J_F = np.eye(2)
    else:
        J_F = np.eye(2)
        
    # If we already have a root
    if norm_Fx0 <= 1e-8:
        return x0_arr

    # Homotopy steps
    # t goes from 0 to 1.
    # We will take steps in t. If Newton fails, we reduce dt.
    
    t = 0.0
    x = x0_arr.copy()
    J_H = np.eye(2) # dH/dx at t=0 is I
    
    # Adaptive t-stepping
    dt = 0.1
    min_dt = 1e-4
    
    while t < 1.0 - 1e-6:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] > max_calls:
            break
            
        # Target t
        t_new = min(t + dt, 1.0)
        
        # We want to find x such that H(x, t_new) = 0
        # H(x, t) = t*F(x) + (1-t)*(x-x0)
        # We are currently at x, which is root of H(x, t_old).
        # Let's do Newton iterations on H(x, t_new) starting from current x.
        # Jacobian of H w.r.t x at (x, t_new) is:
        # J_H_new = t_new * J_F(x) + (1 - t_new) * I
        # We have J_F approximated from previous step? 
        # Actually, we can maintain J_F separately and update it.
        
        # Let's reformulate: 
        # We maintain x and J_F (Jacobian of F at x).
        # At each t step:
        # 1. Update J_H = t_new * J_F + (1 - t_new) * I
        # 2. Solve J_H * dx = -H(x, t_new)
        # 3. x_new = x + dx
        # 4. Update J_F using Broyden based on F(x_new) - F(x)
        
        # Compute H(x, t_new)
        # H = t_new * F(x) + (1 - t_new) * (x - x0)
        # We have F(x) from previous iteration? 
        # In _newton_step we evaluated F. Let's keep Fx current.
        
        # We need F(x) at current x. 
        # If we just did a Newton step, we have Fx. 
        # Initially we have Fx0.
        
        # Let's integrate the Newton step logic directly here for better control.
        
        # Current state: x, J_F, Fx (F at x)
        # We need to solve H(x, t_new) = 0.
        # H(x, t_new) = t_new * F(x) + (1 - t_new) * (x - x0)
        
        # Newton step for x:
        # J_H = t_new * J_F + (1 - t_new) * I
        # Residual R = H(x, t_new)
        # dx = -J_H^{-1} R
        
        # But R depends on x. 
        # Let's do a few inner Newton iterations for fixed t_new.
        
        success = False
        for _ in range(15): # Max inner iterations
            if call_counter[0] > max_calls:
                break
                
            # Compute R = H(x, t_new)
            # We need F(x). If we changed x in previous inner iter, we have Fx.
            # Initially, we have Fx from start.
            
            R = t_new * Fx + (1 - t_new) * (x - x0_arr)
            
            # Check if we are close to root of H
            # Note: We want H=0. 
            # If t_new=1, H=F. So ||F|| <= tol means success.
            # If t_new<1, we just need to advance.
            
            J_H = t_new * J_F + (1 - t_new) * np.eye(2)
            
            try:
                dx = np.linalg.solve(J_H, -R)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J_H, -R, rcond=None)[0]
                
            if not np.all(np.isfinite(dx)):
                break
                
            # Damping
            dx_norm = np.linalg.norm(dx)
            if dx_norm > 1.0:
                dx = dx / dx_norm * 1.0
                
            x_trial = x + dx
            Fx_trial, norm_Fx_trial, valid_trial = _safe_eval(F, x_trial)
            if call_counter is not None:
                call_counter[0] += 1
                
            if not valid_trial:
                # Try smaller step
                dx = 0.5 * dx
                x_trial = x + dx
                Fx_trial, norm_Fx_trial, valid_trial = _safe_eval(F, x_trial)
                if call_counter is not None:
                    call_counter[0] += 1
                if not valid_trial:
                    break
                    
            # Update J_F using Broyden
            y_F = Fx_trial - Fx
            s_F = x_trial - x
            if np.linalg.norm(s_F) > 1e-12:
                J_Fs = J_F @ s_F
                diff_F = y_F - J_Fs
                J_F = J_F + np.outer(diff_F, s_F) / (s_F @ s_F)
                
            x = x_trial
            Fx = Fx_trial
            
            # Check convergence for this t step
            # We can stop inner loop if ||R|| is small
            R_new = t_new * Fx + (1 - t_new) * (x - x0_arr)
            if np.linalg.norm(R_new) < 1e-8:
                success = True
                break
                
            # Also check if we hit the final root
            if t_new >= 1.0 - 1e-6 and norm_Fx_trial <= 1e-8:
                best_x = x
                best_norm = norm_Fx_trial
                success = True
                break
                
        if success:
            # Increase dt for next step
            dt = min(dt * 1.5, 0.5)
            t = t_new
        else:
            # Decrease dt
            dt = max(dt * 0.5, min_dt)
            if dt <= min_dt:
                # Stuck, try to restart or break
                break
                
    # Final check
    Fx_final, norm_final, valid_final = _safe_eval(F, best_x)
    if call_counter is not None:
        call_counter[0] += 1
        
    if valid_final and norm_final <= 1e-8:
        return best_x
        
    # If homotopy failed, fallback to multi-start Broyden?
    # Or just return best_x found.
    # Let's try a quick multi-start if we have budget.
    
    if call_counter[0] < max_calls - 10:
        starts = [x0_arr, np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
        for s in starts:
            if call_counter[0] >= max_calls:
                break
            # Quick Broyden from s
            # Reuse _broyden_step logic? 
            # To keep it simple and robust, just return best_x.
            pass
            
    return best_x
# EVOLVE-BLOCK-END
