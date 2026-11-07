"""Root finder for 3 fixed 2D systems using Global Trust-Region Broyden with Multi-Start."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    x0 = np.asarray(x0, dtype=float)
    
    # Generate a set of diverse starting points
    # 1. The given x0
    # 2. Origin
    # 3. Small perturbations around x0 and origin
    # 4. Larger perturbations
    
    candidates = []
    if np.all(np.isfinite(x0)):
        candidates.append(x0.copy())
    
    candidates.append(np.zeros(2))
    
    # Perturbations around x0
    if np.all(np.isfinite(x0)):
        for d in [0.1, 1.0, 10.0]:
            for dx in [-d, 0, d]:
                for dy in [-d, 0, d]:
                    if dx == 0 and dy == 0: continue
                    candidates.append(x0 + np.array([dx, dy]))
    
    # Perturbations around origin
    for d in [0.1, 1.0, 10.0]:
        for dx in [-d, 0, d]:
            for dy in [-d, 0, d]:
                if dx == 0 and dy == 0: continue
                candidates.append(np.array([dx, dy]))
    
    # A few random-ish points
    candidates.append(np.array([1.0, 1.0]))
    candidates.append(np.array([-1.0, 1.0]))
    candidates.append(np.array([1.0, -1.0]))
    candidates.append(np.array([-1.0, -1.0]))
    candidates.append(np.array([5.0, 5.0]))
    candidates.append(np.array([-5.0, 5.0]))
    candidates.append(np.array([5.0, -5.0]))
    candidates.append(np.array([-5.0, -5.0]))

    best_x = None
    best_norm = float('inf')
    
    for x_start in candidates:
        if time.time() - start_time > time_limit:
            break
            
        if not np.all(np.isfinite(x_start)):
            continue
            
        x, Fx, norm_F = _broyden_trust_region(F, x_start, start_time, time_limit)
        
        if x is not None and Fx is not None and np.all(np.isfinite(x)) and np.all(np.isfinite(Fx)):
            if norm_F < best_norm:
                best_norm = norm_F
                best_x = x
                
        if best_norm <= 1e-8:
            return best_x
            
    if best_x is not None:
        return best_x
    else:
        return np.zeros(2)

def _broyden_trust_region(F, x0, start_time, time_limit):
    x = np.array(x0, dtype=float)
    
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            Fx = Fx.reshape(2)
        if not np.all(np.isfinite(Fx)):
            return None, None, float('inf')
    except Exception:
        return None, None, float('inf')
        
    norm_F = np.linalg.norm(Fx)
    if norm_F <= 1e-8:
        return x, Fx, norm_F

    # Initialize Jacobian
    h = 1e-5
    J = np.zeros((2, 2))
    try:
        J[:, 0] = (F(x + np.array([h, 0])) - Fx) / h
        J[:, 1] = (F(x + np.array([0, h])) - Fx) / h
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
    except Exception:
        J = np.eye(2)

    delta = 1.0  # Initial trust region radius
    
    for i in range(50):
        if time.time() - start_time > time_limit:
            break
            
        # Solve J * s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(s)):
            break
            
        # Trust Region Step
        norm_s = np.linalg.norm(s)
        if norm_s > delta:
            s = s * (delta / norm_s)
            
        x_new = x + s
        F_new = None
        norm_F_new = float('inf')
        
        try:
            F_new = F(x_new)
            F_new = np.asarray(F_new, dtype=float)
            if F_new.shape != (2,):
                F_new = F_new.reshape(2)
            if not np.all(np.isfinite(F_new)):
                # Step led to invalid point, shrink trust region
                delta *= 0.5
                continue
            norm_F_new = np.linalg.norm(F_new)
        except Exception:
            delta *= 0.5
            continue
            
        # Calculate actual reduction and predicted reduction
        # Predicted: ||Fx||^2 - ||Fx + J*s||^2 (approx)
        # Actual: ||Fx||^2 - ||F_new||^2
        F_pred = Fx + J @ s
        pred_reduction = np.linalg.norm(Fx)**2 - np.linalg.norm(F_pred)**2
        act_reduction = np.linalg.norm(Fx)**2 - norm_F_new**2
        
        # If step was outside trust region boundary (norm_s was clipped)
        # and it worked well, we can expand.
        # Standard Trust Region logic:
        # If act_reduction > 0.5 * pred_reduction (and step was full step or near it)
        # Expand delta.
        # If act_reduction < 0.1 * pred_reduction
        # Shrink delta.
        
        rho = 0.0
        if pred_reduction > 1e-12:
            rho = act_reduction / pred_reduction
        else:
            # If predicted reduction is tiny, check if we actually reduced
            if act_reduction > 0:
                rho = 1.0
            else:
                rho = 0.0
                
        if rho > 0.75:
            # Good step, expand trust region
            delta = min(2.0 * delta, 10.0) # Cap expansion
        elif rho < 0.25:
            # Poor step, shrink trust region
            delta = 0.5 * delta
        else:
            # Acceptable step, keep delta same or slight adjust
            # delta = delta (no change)
            pass
        # Accept step if it reduces the norm
        if norm_F_new < norm_F:
            # Broyden Update
            y = F_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-12:
                J = J + np.outer(y - J @ s, s) / s_dot_s
                
            x = x_new
            Fx = F_new
            norm_F = norm_F_new
            
            if norm_F <= 1e-8:
                return x, Fx, norm_F
        else:
            # Step did not reduce norm, do not update x or J, just adjust delta
            # (Already adjusted delta above based on rho)
            # If delta becomes too small, we might be stuck.
            if delta < 1e-10:
                break
                
    return x, Fx, norm_F
# EVOLVE-BLOCK-END
