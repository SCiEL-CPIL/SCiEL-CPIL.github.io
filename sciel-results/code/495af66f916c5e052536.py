import numpy as np
from scipy.optimize import minimize
import time

def compute_P(support, weights):
    n = len(support)
    if n == 0:
        return 0.0
    
    w_sum = np.sum(weights)
    if w_sum <= 1e-9:
        return 0.0
    w = weights / w_sum
    
    sums_ij = support[:, None] + support[None, :]
    
    total = 0.0
    for k in range(n):
        for l in range(n):
            T = 2 * support[l] - support[k]
            mask = sums_ij < T
            inner = (w * mask).sum(axis=1)
            val = (w * inner).sum()
            total += w[k] * w[l] * val
            
    return total

def _unpack_vars(vars, n):
    s = np.zeros(n)
    s[0:n-1] = vars[0:n-1]
    s[n-1] = 1.0
    s = np.maximum(s, 0.0)
    
    w_raw = vars[n-1:2*n-1]
    w = np.exp(w_raw - np.max(w_raw))
    w = w / np.sum(w)
    
    return s, w

def _pack_vars(s, w):
    vars = np.empty(2*n - 1)
    vars[0:n-1] = s[0:n-1]
    # Use log of weights (with small epsilon to avoid -inf)
    w_safe = np.maximum(w, 1e-15)
    vars[n-1:2*n-1] = np.log(w_safe)
    return vars

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    deadline = time.time() + 600
    best_score = 0.0
    best_support = None
    best_weights = None
    
    for n in range(2, 11):
        if time.time() > deadline - 120:
            break
        
        s_init = np.linspace(0, 1, n)
        w_init = np.ones(n) / n
        
        def objective(vars, n=n):
            if time.time() > deadline - 15:
                return 1.0
            
            s, w = _unpack_vars(vars, n)
            
            if np.any(np.isnan(s)) or np.any(np.isnan(w)):
                return 1.0
            
            P = compute_P(s, w)
            return -P
        
        def objective_grad(vars, n=n):
            if time.time() > deadline - 15:
                return 1.0, np.zeros_like(vars)
            
            s, w = _unpack_vars(vars, n)
            
            if np.any(np.isnan(s)) or np.any(np.isnan(w)):
                return 1.0, np.zeros_like(vars)
            
            P = compute_P(s, w)
            
            # Numerical gradient via central differences
            grad = np.zeros_like(vars)
            h = 1e-7
            for i in range(len(vars)):
                vars_plus = vars.copy()
                vars_minus = vars.copy()
                vars_plus[i] += h
                vars_minus[i] -= h
                
                s_plus, w_plus = _unpack_vars(vars_plus, n)
                s_minus, w_minus = _unpack_vars(vars_minus, n)
                
                P_plus = compute_P(s_plus, w_plus)
                P_minus = compute_P(s_minus, w_minus)
                
                grad[i] = -(P_plus - P_minus) / (2 * h)
            
            return -P, grad
        
        x0 = np.concatenate([s_init[0:n-1], np.log(w_init)])
        
        # SLSQP with numerical gradient
        try:
            res = minimize(objective, x0, jac=objective_grad, method='SLSQP',
                           options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
        except Exception:
            res = None
        
        if res is not None and hasattr(res, 'fun') and res.fun < 0:
            P_val = -res.fun
            if P_val > best_score:
                best_score = P_val
                s_final, w_final = _unpack_vars(res.x, n)
                best_support = s_final.copy()
                best_weights = w_final.copy()
        
        # Polish with tighter tolerance if time allows
        if time.time() < deadline - 60 and best_support is not None and len(best_support) == n:
            try:
                res2 = minimize(objective, res.x, jac=objective_grad, method='SLSQP',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
                if hasattr(res2, 'fun') and res2.fun < 0:
                    P_val2 = -res2.fun
                    if P_val2 > best_score:
                        best_score = P_val2
                        s_final, w_final = _unpack_vars(res2.x, n)
                        best_support = s_final.copy()
                        best_weights = w_final.copy()
            except Exception:
                pass

    if best_support is None:
        n = 32
        support = np.arange(n, dtype=np.float64)
        weights = np.ones(n, dtype=np.float64) / n
        return support, weights
        
    return best_support, best_weights
