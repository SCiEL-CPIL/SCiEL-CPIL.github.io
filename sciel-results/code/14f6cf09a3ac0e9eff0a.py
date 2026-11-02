import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_corrs(h):
    """Compute all linear cross-correlations R_k = (2/N) * sum_{i=0}^{N-k-1} h[i]*h[i+k]"""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k - 1] = np.dot(h[:n - k], h[k:]) * (2.0 / n)
    return corrs

def _objective_and_grad(x):
    """
    x = [h, t]
    Objective: t
    Constraints: t - R_k(h) >= 0
    """
    n = N
    h = x[:n]
    t = x[n]
    
    # Objective is t
    f = t
    
    # Gradient of objective
    grad_f = np.zeros(n + 1)
    grad_f[n] = 1.0
    
    # Compute correlations
    corrs = _compute_corrs(h)
    
    # Constraints: g_k = t - corrs_k >= 0
    # We define the constraint function as returning g_k
    # scipy expects constraints to be >= 0 for 'ineq' type
    
    # Gradient of constraints
    # d(t - R_k)/dt = 1
    # d(t - R_k)/dh_j = - d(R_k)/dh_j
    # d(R_k)/dh_j = (2/N) * (h[j+k] if j+k < N else 0) + (2/N) * (h[j-k] if j-k >= 0 else 0)
    
    # We will compute the Jacobian of the constraints matrix
    # J[i, j] = d(g_i)/d(x_j)
    # g_i corresponds to constraint i (shift k=i+1)
    
    # Instead of returning a dense Jacobian which might be slow to construct in Python loop,
    # we can let scipy compute it numerically if it's faster, but analytic is better.
    # However, SLSQP with analytic Jacobian for constraints is tricky to set up correctly 
    # without a specific structure. 
    # Let's stick to providing the constraint function and let SLSQP use numerical differentiation 
    # for the constraints, but provide analytic gradient for the objective? 
    # No, SLSQP uses numerical gradients for both by default if not provided.
    # Providing analytic gradients for constraints is complex. 
    # Given N=100, numerical differentiation is likely fast enough.
    
    return f, grad_f

def _constraints(x):
    """
    Returns a list of constraint dicts.
    """
    n = N
    h = x[:n]
    t = x[n]
    
    corrs = _compute_corrs(h)
    
    # Equality constraint: sum(h) - TARGET_SUM = 0
    eq = {
        'type': 'eq',
        'fun': lambda x: np.sum(x[:n]) - TARGET_SUM,
        'jac': lambda x: np.concatenate([np.ones(n), [0.0]])
    }
    
    # Inequality constraints: t - R_k(h) >= 0
    # We can vectorize this if we use a single constraint function that returns an array?
    # scipy.optimize.minimize with SLSQP supports multiple constraints.
    # It's often more efficient to provide a single function that returns an array of constraints
    # if the solver supports it, but standard interface is list of dicts.
    # Let's define one constraint dict per shift? That's N-1 dicts.
    # Or define a function that returns the vector of violations?
    # SLSQP accepts 'fun' returning an array for 'ineq' if 'jac' returns a matrix.
    
    # Let's try a single inequality constraint function returning an array
    def ineq_fun(x):
        h = x[:n]
        t = x[n]
        corrs = _compute_corrs(h)
        return t - corrs
    
    def ineq_jac(x):
        h = x[:n]
        t = x[n]
        n = len(h)
        # Jacobian shape: (n-1, n+1)
        # Row k corresponds to constraint t - R_{k+1}
        # Col 0..n-1 correspond to h, Col n corresponds to t
        
        J = np.zeros((n - 1, n + 1))
        
        # d/dt = 1
        J[:, n] = 1.0
        
        # d/dh_j for R_k
        # R_k = (2/N) * sum_{i=0}^{N-k-1} h[i] h[i+k]
        # d(R_k)/d(h_j) = (2/N) * (h[j+k] if j+k < N else 0) + (2/N) * (h[j-k] if j-k >= 0 else 0)
        
        # We need -d(R_k)/d(h_j)
        
        # Vectorized approach for Jacobian construction
        # For each k (1 to N-1):
        #   J[k-1, j] = - (2/N) * (h[j+k] if j+k < N else 0) - (2/N) * (h[j-k] if j-k >= 0 else 0)
        
        # This loop is O(N^2). N=100, so 10k ops. Fast.
        for k in range(1, n):
            row_idx = k - 1
            # Contribution from h[j+k]
            # j ranges from 0 to N-k-1
            # h[j+k] is at index j+k
            # So for j in 0..N-k-1, J[row_idx, j] -= (2/N) * h[j+k]
            J[row_idx, :n-k] -= (2.0 / n) * h[k:]
            
            # Contribution from h[j-k]
            # j ranges from k to N-1
            # h[j-k] is at index j-k
            # So for j in k..N-1, J[row_idx, j] -= (2/N) * h[j-k]
            J[row_idx, k:] -= (2.0 / n) * h[:n-k]
            
        return J

    ineq = {
        'type': 'ineq',
        'fun': ineq_fun,
        'jac': ineq_jac
    }
    
    return [eq, ineq]

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    v = np.asarray(v, dtype=float)
    
    # First project onto {x >= 0, sum = target_sum}
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    # Clip to [0, 1]
    w = np.clip(w, 0, 1)
    
    # Adjust to ensure sum = target_sum
    for _ in range(20):
        diff = target_sum - np.sum(w)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = w < 1.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        else:
            mask = w > 0.0
            cnt = np.sum(mask)
            if cnt == 0:
                break
            w[mask] += diff / cnt
        w = np.clip(w, 0, 1)
    
    return w

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = TARGET_SUM
    
    best_x = np.full(n, 0.5)
    best_t = float('inf')
    
    # Generate diverse initializations
    restarts = []
    
    # 1. Uniform
    restarts.append(np.full(n, 0.5))
    
    # 2. Alternating 0/1
    alt1 = np.zeros(n)
    alt1[1::2] = 1.0
    alt2 = np.zeros(n)
    alt2[0::2] = 1.0
    restarts.append(alt1)
    restarts.append(alt2)
    
    # 3. Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        if k > 0 and k < n:
            block1 = np.zeros(n)
            block1[:k] = target_sum / k
            restarts.append(block1)
            
            block2 = np.zeros(n)
            block2[k:] = target_sum / (n - k)
            restarts.append(block2)
            
            block3 = np.zeros(n)
            block3[:k] = 1.0
            # Adjust to meet sum
            remaining_sum = target_sum - k
            if n - k > 0:
                block3[k:] = remaining_sum / (n - k)
            restarts.append(block3)
            
            block4 = np.zeros(n)
            block4[k:] = 1.0
            remaining_sum = target_sum - (n - k)
            if k > 0:
                block4[:k] = remaining_sum / k
            restarts.append(block4)
            
    # 4. Sinusoidal
    x_grid = np.linspace(0, 2 * np.pi, n)
    restarts.append(0.5 + 0.4 * np.sin(x_grid))
    restarts.append(0.5 + 0.4 * np.cos(x_grid))
    restarts.append(0.5 + 0.3 * np.sin(2 * x_grid))
    
    # 5. Random
    np.random.seed(42)
    for _ in range(5):
        r = np.random.uniform(0, 1, n)
        r = _project_to_simplex(r, target_sum)
        restarts.append(r)
    
    bounds = [(0, 1)] * n + [(0, 1)]
    
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
            
        x0_proj = _project_to_simplex(x0, target_sum)
        # Append t initial guess. Use max correlation as upper bound
        t0 = np.max(_compute_corrs(x0_proj)) + 0.1
        x_init = np.concatenate([x0_proj, [t0]])
        
        try:
            res = minimize(
                _objective_and_grad,
                x_init,
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=_constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                res_h = res.x[:n]
                res_t = res.x[n]
                
                # Project back to ensure feasibility
                res_h = _project_to_simplex(res_h, target_sum)
                
                # Recalculate actual max correlation
                actual_t = np.max(_compute_corrs(res_h))
                
                if actual_t < best_t:
                    best_t = actual_t
                    best_x = res_h
                    
        except Exception:
            pass
            
    # Polish phase: Run SLSQP again on the best solution with tighter tolerance
    if _time_remaining() > 10.0 and best_t < float('inf'):
        x_init = np.concatenate([best_x, [best_t]])
        try:
            res = minimize(
                _objective_and_grad,
                x_init,
                method='SLSQP',
                jac=True,
                bounds=bounds,
                constraints=_constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            if res.success:
                res_h = res.x[:n]
                res_h = _project_to_simplex(res_h, target_sum)
                actual_t = np.max(_compute_corrs(res_h))
                if actual_t < best_t:
                    best_t = actual_t
                    best_x = res_h
        except Exception:
            pass
            
    # Final check and cleanup
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
            
    return best_x
