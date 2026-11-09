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

def _levenberg_marquardt(F, x0, max_iter=30, tol=1e-10, call_budget=100):
    """
    Levenberg-Marquardt method for 2D systems.
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
        # If FD fails, we can't start LM properly. 
        # Fallback to simple gradient descent? Or just fail.
        # For robustness, let's try a smaller h or just return fail.
        # Given the "hidden systems" constraint, robustness is key.
        # If FD fails, it might be a boundary. Let's just return current best.
        return x, False, calls
        
    J = np.zeros((2, 2))
    J[:, 0] = (Fx1 - Fx) / h
    J[:, 1] = (Fx2 - Fx) / h
    
    lambda_lm = 1e-3
    
    for i in range(max_iter):
        if calls >= call_budget:
            break
            
        # Update Jacobian every 2 steps to save calls, or use previous J
        # For 2D, recomputing J costs 2 calls. LM step costs 1 call.
        # Total 3 calls/step if recomputing.
        # Let's recompute J every step for accuracy if budget allows, 
        # or every 2 steps. Let's try every step for robustness.
        
        Fx1 = _safe_call(F, x + e1)
        Fx2 = _safe_call(F, x + e2)
        calls += 2
        
        if Fx1 is None or Fx2 is None:
             # Try smaller h?
             h *= 0.1
             e1[0] = h
             e2[1] = h
             Fx1 = _safe_call(F, x + e1)
             Fx2 = _safe_call(F, x + e2)
             calls += 2
             if Fx1 is None or Fx2 is None:
                 break
             
        J[:, 0] = (Fx1 - Fx) / h
        J[:, 1] = (Fx2 - Fx) / h
        
        try:
            JTJ = J.T @ J
            JTg = J.T @ Fx
            
            # LM step: (J^T J + lambda * I) * d = -J^T F
            # We want to solve for d
            # (J^T J + lambda * diag(J^T J)) * d = -J^T F  (Marquardt variant)
            # Or simpler: (J^T J + lambda * I) * d = -J^T F
            
            I = np.eye(2)
            A = JTJ + lambda_lm * I
            b = -JTg
            
            d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
            
        if not np.all(np.isfinite(d)):
            break
            
        # Try the step
        x_new = x + d
        Fx_new = _safe_call(F, x_new)
        calls += 1
        
        if Fx_new is None:
            lambda_lm *= 10.0
            continue
            
        norm_new = _norm(Fx_new)
        norm_curr = _norm(Fx)
        
        if norm_new < norm_curr:
            # Accept
            x = x_new
            Fx = Fx_new
            lambda_lm = max(lambda_lm * 0.1, 1e-12)
            
            if norm_new <= 1e-8:
                return x, True, calls
        else:
            # Reject
            lambda_lm *= 10.0
            if lambda_lm > 1e12:
                break
                
    return x, False, calls

def find_root_system(F, x0):
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 110.0  # Safety margin from 120s limit
    
    # Strategy: Multi-start LM
    # 1. Try x0
    # 2. If fail, try a grid of points around x0
    
    x0 = np.asarray(x0, dtype=float)
    
    # Define a set of starting points
    # x0, x0 +/- small, x0 +/- large
    # We want to cover a reasonable area without wasting too many calls.
    
    # Scale of exploration: if x0 is near 0, explore [-10, 10]. If x0 is large, relative scale.
    scale = max(1.0, np.linalg.norm(x0))
    
    # Grid offsets
    # Coarse grid first
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
    ]
    
    # Shuffle to avoid bias towards specific directions if x0 is bad
    # But keep x0 first as it's the most likely intended start
    # Actually, let's just iterate through them.
    
    best_x = x0
    best_norm = float('inf')
    found_root = False
    
    # Budget per start point: ~20-30 calls
    # Total budget ~100-150 calls to be safe, but we have 500 limit.
    # Let's allow up to 40 calls per start point.
    # If we have 20 start points, that's 800 calls worst case. 
    # We should stop if we exceed a global call budget, say 300.
    
    global_call_budget = 300
    calls_used = 0
    
    for offset in offsets:
        if _elapsed() > deadline:
            break
            
        if calls_used >= global_call_budget:
            break
            
        x_start = x0 + np.array(offset)
        
        # Run LM from x_start
        x_res, success, calls_in_run = _levenberg_marquardt(F, x_start, max_iter=20, call_budget=40)
        
        calls_used += calls_in_run
        
        if success:
            return x_res
            
        # Update best if not found
        # We need to evaluate F(x_res) to know its norm? 
        # _levenberg_marquardt returns the last x. 
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
