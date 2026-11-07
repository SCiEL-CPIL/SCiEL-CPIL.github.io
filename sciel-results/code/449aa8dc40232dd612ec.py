"""Root finder for 3 fixed 2D systems using Broyden's Method with Adaptive Line Search."""
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

def _broyden_step(F, x0, max_iter=50, tol=1e-10, call_counter=None):
    """
    Broyden's method (Good/Bad) with adaptive line search.
    Initializes Jacobian with forward differences (2 calls).
    Each iteration: 1 call for F(x_new) + 1 call for F(x_old) is reused.
    Actually, we need F at current x to compute s and y.
    Standard Broyden:
    1. Solve J s = -F(x)
    2. x_new = x + alpha * s
    3. F_new = F(x_new)
    4. s = x_new - x
    5. y = F_new - F_old
    6. Update J
    7. x = x_new, F_old = F_new
    
    Cost per iteration: 1 call (for F_new).
    Initial cost: 1 (F at x0) + 2 (Jacobian) = 3 calls.
    Total cost ~ 3 + N_iter.
    """
    x = np.array(x0, dtype=float)
    Fx, norm_F, valid = _safe_eval(F, x)
    if call_counter is not None:
        call_counter[0] += 1
    if not valid:
        return None, float('inf')
    if norm_F <= tol:
        return x, norm_F

    # Initial Jacobian via forward differences
    h = 1e-5
    e0 = np.array([h, 0.0])
    e1 = np.array([0.0, h])
    
    Fx_e0, _, v0 = _safe_eval(F, x + e0)
    if call_counter is not None:
        call_counter[0] += 1
    Fx_e1, _, v1 = _safe_eval(F, x + e1)
    if call_counter is not None:
        call_counter[0] += 1
        
    if not (v0 and v1):
        return x, norm_F
        
    J = np.zeros((2, 2))
    J[:, 0] = (Fx_e0 - Fx) / h
    J[:, 1] = (Fx_e1 - Fx) / h
    
    if not np.all(np.isfinite(J)):
        return x, norm_F

    for _ in range(max_iter):
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
        if not np.all(np.isfinite(s)):
            break
            
        # Line search
        alpha = 1.0
        x_new = x + alpha * s
        F_new, norm_new, valid_new = _safe_eval(F, x_new)
        if call_counter is not None:
            call_counter[0] += 1
            
        if not valid_new:
            break
            
        # If norm didn't decrease, try smaller alpha
        # Simple backtracking
        while norm_new > norm_F and alpha > 1e-6:
            alpha *= 0.5
            x_new = x + alpha * s
            F_new, norm_new, valid_new = _safe_eval(F, x_new)
            if call_counter is not None:
                call_counter[0] += 1
            if not valid_new:
                break
                
        if not valid_new:
            break
            
        # Broyden update (Good)
        # y = F_new - Fx
        # s = x_new - x
        y = F_new - Fx
        s = x_new - x
        
        if np.linalg.norm(s) > 1e-12:
            # J = J + (y - J s) s^T / (s^T s)
            Js = J @ s
            diff = y - Js
            J = J + np.outer(diff, s) / (s @ s)
            
        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        if norm_F <= tol:
            return x, norm_F
            
        # Check for stagnation
        if np.linalg.norm(s) < 1e-10:
            break
            
    return x, norm_F

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 140.0  # Safety margin from 150s
    
    # Generate a set of diverse starting points
    starts = []
    
    # 1. The provided start
    x0_arr = np.array(x0, dtype=float)
    if np.all(np.isfinite(x0_arr)):
        starts.append(x0_arr)
    else:
        starts.append(np.zeros(2))
        
    # 2. Origin
    starts.append(np.array([0.0, 0.0]))
    
    # 3. Points on unit circle and small radii
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for r in radii:
        for a in angles:
            starts.append(np.array([r * np.cos(a), r * np.sin(a)]))
            
    # 4. Random points in a larger box (seeded for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(20):
        starts.append(rng.uniform(-20, 20, 2))
        
    best_x = starts[0]
    best_norm = float('inf')
    call_counter = [0]
    max_calls = 500  # Hard limit per system
    
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
            
        # Check if we've used too many calls
        if call_counter[0] > max_calls - 60:
            break
            
        x_res, norm_res = _broyden_step(F, x_start, max_iter=50, tol=1e-10, call_counter=call_counter)
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
