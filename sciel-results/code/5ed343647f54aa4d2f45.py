import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 115.0  # Safety margin for 120s limit

def _remaining():
    return _DEADLINE - (time.time() - _T0)

# ... (keep _build_hamiltonians and _ExpHelper) ...

def coefficients(stages):
    global _T0
    stages = int(stages)
    if stages < 2:
        stages = 2
    
    tvals = [0.25, 0.5, 1.0]

    A, B = _build_hamiltonians()
    AB = A + B
    
    helper_A = _ExpHelper(A)
    helper_B = _ExpHelper(B)
    helper_AB = _ExpHelper(AB)
    
    exact_prop = {t: helper_AB.exp(t) for t in tvals}

    # Initial guess: uniform
    a0 = np.array([1.0 / stages] * stages)
    b0 = np.array([1.0 / stages] * stages)
    x0 = np.concatenate([a0, b0])

    best_x = x0.copy()
    best_val = float('inf')
    
    class TimeOutError(Exception):
        pass

    def objective(x):
        nonlocal best_x, best_val
        if _remaining() < 5.0:
            raise TimeOutError()
        
        a = x[:stages]
        b = x[stages:]
        
        # Project to sum=1 to ensure consistency during optimization
        # SLSQP handles constraints, but projecting helps stability
        a_sum = np.sum(a)
        b_sum = np.sum(b)
        if a_sum != 0:
            a = a / a_sum
        if b_sum != 0:
            b = b / b_sum
            
        # Clip to bounds to prevent huge values
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        
        err_sum = 0.0
        for t in tvals:
            U = np.eye(A.shape[0], dtype=complex)
            for k in range(stages):
                bk = b[k]
                if abs(bk) > 1e-15:
                    U = U @ helper_B.exp(bk * t)
                ak = a[k]
                if abs(ak) > 1e-15:
                    U = U @ helper_A.exp(ak * t)
            err_sum += np.linalg.norm(U - exact_prop[t])
        
        val = err_sum / len(tvals)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
        return val

    # Constraints: sum(a) = 1, sum(b) = 1
    cons = (
        {'type': 'eq', 'fun': lambda x: np.sum(x[:stages]) - 1.0},
        {'type': 'eq', 'fun': lambda x: np.sum(x[stages:]) - 1.0}
    )
    
    # Bounds: [-50, 50] for all
    bounds = [(-50.0, 50.0)] * (2 * stages)

    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
        if res.success and res.fun < best_val:
            best_x = res.x
    except TimeOutError:
        pass
    except Exception as e:
        # Fallback to initial guess if something goes wrong
        pass

    # Extract final coefficients
    a_final = best_x[:stages]
    b_final = best_x[stages:]
    
    # Ensure exact sum = 1 and bounds
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    # Normalize to sum 1
    sum_a = np.sum(a_final)
    sum_b = np.sum(b_final)
    if sum_a != 0:
        a_final = a_final / sum_a
    if sum_b != 0:
        b_final = b_final / sum_b
        
    # Final clamp after normalization (might push slightly out of bounds, re-normalize if needed)
    a_final = np.clip(a_final, -50, 50)
    b_final = np.clip(b_final, -50, 50)
    
    # One last sum fix
    diff_a = 1.0 - np.sum(a_final)
    diff_b = 1.0 - np.sum(b_final)
    a_final += diff_a / stages
    b_final += diff_b / stages
    
    return list(a_final), list(b_final)
