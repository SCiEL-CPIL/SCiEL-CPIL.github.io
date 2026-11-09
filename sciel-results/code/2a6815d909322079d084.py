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

def _broyden(F, x0, max_iter=50, tol=1e-10, call_budget=200):
    """
    Broyden's Method (rank-1 update) for 2D systems.
    Returns (x, success, calls_used)
    """
    x = np.array(x0, dtype=float)
    calls = 0
    
    Fx = _safe_call(F, x)
    calls += 1
    if Fx is None:
        return x, False, calls
    
    if _norm(Fx) <= 1e-8:
        return x, True, calls
        
    # Initial Jacobian estimate using finite differences
    h = 1e-5
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    
    Fx1 = _safe_call(F, x + e1)
    Fx2 = _safe_call(F, x + e2)
    calls += 2
    
    if Fx1 is None or Fx2 is None:
        # If FD fails, try smaller h
        h = 1e-6
        e1[0] = h
        e2[1] = h
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        calls += 2
        if Fx1 is None or Fx2 is None:
            return x, False, calls
            
    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h
    
    # Check if J is singular
    try:
        det = np.linalg.det(J)
        if abs(det) < 1e-10:
            # Add small regularization
            J += 1e-6 * np.eye(2)
    except:
        pass

    for i in range(max_iter):
        if calls >= call_budget:
            break
            
        # Solve J * d = -Fx
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
            
        if not np.all(np.isfinite(d)):
            break
            
        # Damping if step is too large
        step_norm = _norm(d)
        if step_norm > 1.0:
            d = d * (1.0 / step_norm)
            
        x_new = x + d
        Fx_new = _safe_call(F, x_new)
        calls += 1
        
        if Fx_new is None:
            # Halve the step
            x_new = x + 0.5 * d
            Fx_new = _safe_call(F, x_new)
            calls += 1
            if Fx_new is None:
                break
                
        norm_new = _norm(Fx_new)
        
        if norm_new <= 1e-8:
            return x_new, True, calls
            
        # Broyden update
        s = x_new - x
        y = Fx_new - Fx
        
        # Rank-1 update: J_new = J + (y - J*s) * s^T / (s^T * s)
        try:
            sTs = s @ s
            if sTs > 1e-15:
                J = J + np.outer(y - J @ s, s) / sTs
        except:
            break
            
        x = x_new
        Fx = Fx_new
        
    return x, False, calls

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0  # Safety margin from 120s limit
    
    x0 = np.asarray(x0, dtype=float)
    
    # Strategy: Multi-start Broyden
    # 1. Try x0
    # 2. If fail, try a grid of points around x0
    
    # Scale of exploration
    scale = max(1.0, np.linalg.norm(x0))
    
    # Grid offsets
    offsets = [
        [0.0, 0.0],
        [0.1 * scale, 0.0], [-0.1 * scale, 0.0],
        [0.0, 0.1 * scale], [0.0, -0.1 * scale],
        [0.1 * scale, 0.1 * scale], [-0.1 * scale, -0.1 * scale],
        [0.1 * scale, -0.1 * scale], [-0.1 * scale, 0.1 * scale],
        
        [0.5 * scale, 0.0], [-0.5 * scale, 0.0],
        [0.0, 0.5 * scale], [0.0, -0.5 * scale],
        
        [1.0 * scale, 0.0], [-1.0 * scale, 0.0],
        [0.0, 1.0 * scale], [0.0, -1.0 * scale],
        
        [1.0 * scale, 1.0 * scale], [-1.0 * scale, -1.0 * scale],
        [1.0 * scale, -1.0 * scale], [-1.0 * scale, 1.0 * scale],
        
        # Random-ish large jumps
        [2.0 * scale, 0.0], [-2.0 * scale, 0.0],
        [0.0, 2.0 * scale], [0.0, -2.0 * scale],
        
        [5.0 * scale, 0.0], [-5.0 * scale, 0.0],
        [0.0, 5.0 * scale], [0.0, -5.0 * scale],
        
        [10.0 * scale, 0.0], [-10.0 * scale, 0.0],
        [0.0, 10.0 * scale], [0.0, -10.0 * scale],
    ]
    
    best_x = x0
    best_norm = float('inf')
    found_root = False
    
    # Budget per start point: ~50 calls
    # Total budget ~200-300 calls to be safe, but we have 500 limit.
    # Let's allow up to 50 calls per start point.
    # If we have 20 start points, that's 1000 calls worst case. 
    # We should stop if we exceed a global call budget, say 300.
    
    global_call_budget = 300
    calls_used = 0
    
    for offset in offsets:
        if _elapsed() > deadline:
            break
            
        if calls_used >= global_call_budget:
            break
            
        x_start = x0 + np.array(offset)
        
        # Run Broyden from x_start
        x_res, success, calls_in_run = _broyden(F, x_start, max_iter=50, call_budget=50)
        
        calls_used += calls_in_run
        
        if success:
            return x_res
            
        # Update best if not found
        # We need to evaluate F(x_res) to know its norm? 
        # _broyden returns the last x. 
        # If it failed, it might be stuck at a local min or boundary.
        # Let's evaluate the final x to see if it's better than current best.
        F_final = _safe_call(F, x_res)
        if F_final is not None:
            calls_used += 1
            norm_final = _norm(F_final)
            if norm_final < best_norm:
                best_norm = norm_final
                best_x = x_res
                
    return best_x
