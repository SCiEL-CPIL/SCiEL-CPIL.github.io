import numpy as np
import time

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return np.all(np.isfinite(x)) and x.shape == (2,)

def _norm(x):
    return float(np.linalg.norm(x))

def _safe_call(F, x):
    """Call F safely, returning None on failure."""
    try:
        y = F(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        if y.shape != (2,):
            return None
        if not np.all(np.isfinite(y)):
            return None
        return y
    except Exception:
        return None

# Pre-compute the inverse of the design matrix for quadratic fitting
# Points relative to center: (0,0), (h,0), (-h,0), (0,h), (0,-h), (h,h)
# Basis: [x^2, xy, y^2, x, y, 1]
def _get_quadratic_design_inverse(h):
    pts = np.array([
        [0, 0],
        [h, 0],
        [-h, 0],
        [0, h],
        [0, -h],
        [h, h]
    ])
    M = np.zeros((6, 6))
    for i, (x, y) in enumerate(pts):
        M[i, 0] = x*x
        M[i, 1] = x*y
        M[i, 2] = y*y
        M[i, 3] = x
        M[i, 4] = y
        M[i, 5] = 1.0
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return None

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0
    
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    Fx = _safe_call(F, x)
    if Fx is None:
        x = x + np.array([0.1, 0.1])
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-8:
        return x

    best_x = x.copy()
    best_norm = _norm(Fx)
    
    # Strategy: Quadratic Model Newton
    # We try to fit a quadratic model to F and find its root directly.
    # This is more accurate than linear (Broyden) for curved functions.
    
    h = 1e-3 # Step size for model construction
    M_inv = _get_quadratic_design_inverse(h)
    
    if M_inv is None:
        # Fallback to Broyden if design matrix is singular (shouldn't happen with this h)
        return _broyden_fallback(F, x, Fx, deadline)

    # Define the 6 sampling points relative to current x
    offsets = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [-h, 0.0],
        [0.0, h],
        [0.0, -h],
        [h, h]
    ])
    
    max_restarts = 5
    for restart in range(max_restarts):
        if _elapsed() > deadline:
            break
            
        # Collect function values at the 6 points
        vals = np.zeros((6, 2))
        valid = True
        for i in range(6):
            pt = x + offsets[i]
            v = _safe_call(F, pt)
            if v is None:
                valid = False
                break
            vals[i] = v
        
        if not valid:
            # If we can't sample all points, try a smaller h or break
            h *= 0.5
            M_inv = _get_quadratic_design_inverse(h)
            if M_inv is None:
                break
            continue
            
        # Fit quadratic coefficients
        # vals is 6x2. We want coeffs C (6x2) such that M @ C = vals
        # C = M_inv @ vals
        try:
            C = M_inv @ vals
        except Exception:
            break
            
        # Evaluate the polynomial model and its Jacobian at the center (x)
        # P(x) = C[5] (since x=0, y=0 in local coords)
        # Wait, the model is defined in local coordinates (u,v) where x = x_center + u, y = y_center + v.
        # P(u,v) = a u^2 + b uv + c v^2 + d u + e v + f
        # At center (0,0), P(0,0) = f = C[5]
        # We want to solve P(u,v) = 0.
        
        # Jacobian of P at (0,0):
        # dP/du = 2a u + b v + d -> at (0,0) is d = C[3]
        # dP/dv = b u + 2c v + e -> at (0,0) is e = C[4]
        # So J_model at center is:
        # [[C1[3], C1[4]],
        #  [C2[3], C2[4]]]
        
        J_model = np.array([
            [C[0, 3], C[0, 4]],
            [C[1, 3], C[1, 4]]
        ])
        
        P_center = C[:, 5]
        
        # Newton step for the model:
        # Solve J_model * delta = -P_center
        try:
            delta = np.linalg.solve(J_model, -P_center)
        except np.linalg.LinAlgError:
            # If singular, regularize
            try:
                delta = np.linalg.solve(J_model + 1e-8 * np.eye(2), -P_center)
            except np.linalg.LinAlgError:
                break
                
        if not np.all(np.isfinite(delta)):
            break
            
        # The candidate point is x + delta (since delta is in local coords)
        x_new = x + delta
        
        # Check if the step is reasonable (not too large)
        step_norm = _norm(delta)
        if step_norm > 10 * h:
             # If the step is huge, the model might be bad. 
             # But for quadratic, the root is exact. 
             # However, if the function is not quadratic, the root might be far.
             # Let's accept it for now and verify with F.
             pass
             
        F_new = _safe_call(F, x_new)
        
        if F_new is None:
            # The candidate is invalid. Try a smaller step?
            # Or just break and fallback.
            # Let's try a damped step
            x_new = x + 0.5 * delta
            F_new = _safe_call(F, x_new)
            if F_new is None:
                break
                
        norm_new = _norm(F_new)
        
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x_new.copy()
            
        if norm_new <= 1e-8:
            return x_new
            
        # If we didn't reach the root, update center to the new point and repeat
        # This effectively refines the local quadratic model.
        x = x_new
        Fx = F_new
        
        # Adaptive h: if we moved a lot, increase h? If we moved little, decrease?
        # For now, keep h constant or slightly increase to capture global shape.
        # h = max(h * 1.5, 1e-2) 
        # But if we are close, we want precision.
        # Let's stick to the same h for consistency in the design matrix.
        
    # If the quadratic model approach didn't solve it, fallback to Broyden
    # Using the best_x found so far
    return _broyden_fallback(F, best_x, _safe_call(F, best_x), deadline)

def _broyden_fallback(F, x, Fx, deadline):
    """Standard Broyden's method as a fallback."""
    if Fx is None:
        Fx = _safe_call(F, x)
        if Fx is None:
            return x

    if _norm(Fx) <= 1e-8:
        return x

    h = 1e-6
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    
    if Fx1 is None or Fx2 is None:
        return x

    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h

    x_old = x
    Fx_old = Fx
    best_x = x
    best_norm = _norm(Fx)
    
    max_iter = 50
    
    for i in range(max_iter):
        if _elapsed() > deadline:
            break
            
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(d)):
            break
            
        alpha = 1.0
        norm_current = _norm(Fx)
        found_step = False
        
        for _ls in range(10):
            if _elapsed() > deadline:
                break
            x_new = x + alpha * d
            Fx_new = _safe_call(F, x_new)
            
            if Fx_new is None:
                alpha *= 0.5
                continue
                
            norm_new = _norm(Fx_new)
            
            if norm_new < norm_current:
                x = x_new
                Fx = Fx_new
                found_step = True
                break
            else:
                alpha *= 0.5
                
        if not found_step:
            break
            
        if norm_new < best_norm:
            best_norm = norm_new
            best_x = x.copy()
            
        if best_norm <= 1e-8:
            break

        s = x - x_old
        y = Fx - Fx_old
        s_dot_s = np.dot(s, s)
        if s_dot_s < 1e-16:
            break
        J = J + np.outer(y - J @ s, s) / s_dot_s
        x_old = x
        Fx_old = Fx

    return best_x
