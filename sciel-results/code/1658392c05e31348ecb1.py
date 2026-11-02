"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of the cross-correlation R(s) = ∫ h(x)h(x+s)dx.
    
    We evaluate at s = 2k/n for k = 1, ..., n-1 (shifts within (0,2)).
    R(2k/n) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    
    We want to maximize min_k R(2k/n).
    """
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _compute_all_correlations(h):
    """Compute all correlations R(k) for k=1,...,n-1."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp."""
    corrs = _compute_all_correlations(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation."""
    h = x
    n = len(h)
    
    corrs = _compute_all_correlations(h)
    
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_neg_beta = np.exp(-beta * shifted)
    sum_exp_neg = np.sum(exp_neg_beta)
    soft_min = min_val - np.log(sum_exp_neg) / beta
    obj = 1.0 - soft_min
    
    weights = exp_neg_beta / sum_exp_neg
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w > 1e-15:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.nonzero(u * np.arange(1, n + 1) > (cssv - target_sum))[0]
    if len(rho_candidates) == 0:
        rho = n - 1
    else:
        rho = rho_candidates[-1]
    theta = (cssv[rho] - target_sum) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    
    w = np.clip(w, 0, 1)
    
    for _ in range(10):
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

def _ising_glauber_optimization(h_init, target_sum, n, time_budget):
    """Optimize using Ising spin chain model with Glauber dynamics.
    
    Model: h[i] = 0.5 + 0.5 * s[i] where s[i] in {-1, 1}
    Energy: E = -J * sum_k w_k * (autocorr_k) + lambda * (sum(s) - target)^2
    The antiferromagnetic coupling J > 0 penalizes high autocorrelation.
    """
    best_h = h_init.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(best_h)
    
    # Initialize spins from h
    s = np.sign(h_init - 0.5)
    s[s == 0] = 1.0
    
    # Current h from spins
    h = 0.5 + 0.5 * s
    
    # Project to satisfy constraints
    h = _project_to_simplex(h, target_sum)
    # Re-derive spins from projected h
    s = np.sign(h - 0.5)
    s[s == 0] = 1.0
    
    J = 1.0  # Antiferromagnetic coupling strength
    lambda_sum = 10.0  # Penalty for sum constraint
    
    step_size = 0.1
    no_improve = 0
    
    for iteration in range(5000):
        if _time_remaining() < 6.0:
            break
        
        # Compute current objective
        min_corr = _compute_cross_correlation_min(h)
        obj = 1.0 - min_corr
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                step_size *= 0.9
                if step_size < 1e-4:
                    break
        
        # Compute local fields for each spin
        # The energy of spin i depends on its neighbors through the autocorrelation
        # dE/ds_i = -J * sum_k (2/n) * (h[i+k] + h[i-k] for valid indices) * 0.5
        #          + lambda_sum * 2 * (sum(s) - target_sum_s) * 0.5
        
        corrs = _compute_all_correlations(h)
        # Find which shifts are most problematic (lowest correlation)
        # Weight shifts by their contribution to the min
        
        # Use soft-min weights to determine which correlations matter
        beta = 50.0
        min_val = np.min(corrs)
        shifted = corrs - min_val
        exp_neg_beta = np.exp(-beta * shifted)
        weights = exp_neg_beta / np.sum(exp_neg_beta)
        
        # Compute gradient of soft-min w.r.t. h
        grad_h = np.zeros(n)
        for k in range(1, n):
            w = weights[k-1] * (2.0 / n)
            if w > 1e-15:
                grad_h[:n-k] += w * h[k:]
                grad_h[k:] += w * h[:n-k]
        
        # Gradient w.r.t. s: grad_s = 0.5 * grad_h (since h = 0.5 + 0.5*s)
        grad_s = 0.5 * grad_h
        
        # Sum constraint penalty
        sum_s = np.sum(s)
        target_sum_s = target_sum - n * 0.5  # sum(s) should be 0 if sum(h) = n/2
        grad_s += lambda_sum * (sum_s - target_sum_s) / n
        
        # Glauber dynamics: flip spins that reduce energy
        # For each spin, compute the energy change if flipped
        # Delta E for flipping s[i]: -2 * s[i] * (local_field[i])
        # where local_field[i] = -grad_s[i] (since E decreases in direction of -grad)
        
        # Stochastic approach: pick a random spin and decide to flip based on Metropolis
        i = np.random.randint(n)
        if s[i] == 1.0:
            delta_e = -2.0 * 1.0 * (-grad_s[i])  # = 2 * grad_s[i]
        else:
            delta_e = -2.0 * (-1.0) * (-grad_s[i])  # = -2 * grad_s[i]
        
        # Metropolis acceptance
        if delta_e < 0:
            accept = True
        else:
            T = max(0.01, 1.0 / (1.0 + iteration * 0.01))
            accept = np.random.random() < np.exp(-delta_e / T)
        
        if accept:
            s[i] = -s[i]
            h = 0.5 + 0.5 * s
            h = _project_to_simplex(h, target_sum)
            s = np.sign(h - 0.5)
            s[s == 0] = 1.0
        
        # Also do a gradient step on the continuous h
        h_cont = h - step_size * grad_h
        h_cont = _project_to_simplex(h_cont, target_sum)
        new_obj = 1.0 - _compute_cross_correlation_min(h_cont)
        if new_obj < obj:
            h = h_cont
            s = np.sign(h - 0.5)
            s[s == 0] = 1.0
        else:
            step_size *= 0.95
    
    return best_h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    np.random.seed(42)
    
    restarts = []
    restarts.append(np.full(n, 0.5))
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    x = np.linspace(0, 2*np.pi, n)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.4*np.cos(2*x))
    
    # Phase 1: Soft-min optimization
    beta_schedule = [10.0, 30.0, 100.0, 300.0]
    
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        for beta in beta_schedule:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
                x = res_x
            except Exception:
                pass
    
    # Phase 2: Ising Glauber optimization on best solution
    if _time_remaining() > 15.0:
        try:
            h_ising, obj_ising = _ising_glauber_optimization(best_x, target_sum, n, 15.0)
            if obj_ising < best_obj:
                best_obj = obj_ising
                best_x = h_ising
        except Exception:
            pass
    
    # Phase 3: High-beta polish
    if _time_remaining() > 10.0:
        for beta in [500.0, 1000.0, 2000.0]:
            if _time_remaining() < 4.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Phase 4: Local subgradient search
    if _time_remaining() > 6.0:
        step_size = 0.05
        no_improve_count = 0
        for _ in range(150):
            if _time_remaining() < 3.0:
                break
            h = best_x
            n = len(h)
            
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if corr < min_corr:
                    min_corr = corr
                    k_star = k
            
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            h_new = h + step_size * grad
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj - 1e-14:
                best_obj = new_obj
                best_x = h_new
                no_improve_count = 0
            else:
                no_improve_count += 1
                step_size *= 0.7
                if no_improve_count > 15:
                    break
    
    # Ensure exact sum
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        if np.sum(mask) > 0:
            best_x[mask] += diff / np.sum(mask)
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
