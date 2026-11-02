"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_overlaps(h):
    """Compute R_k for k=1..N-1. R_k = (2/N) sum h[i]h[i+k]."""
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    return corrs

def _project_to_simplex(h, target_sum):
    """Project h to satisfy sum(h) = target_sum and 0<=h<=1."""
    h = np.clip(h, 0, 1)
    diff = target_sum - np.sum(h)
    for _ in range(20):
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-8) & (h < 1 - 1e-8)
        if np.sum(mask) == 0:
            break
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        diff = target_sum - np.sum(h)
    # Final exact adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-10:
        # Distribute to all elements proportionally
        h += diff / len(h)
        h = np.clip(h, 0, 1)
    return h

def _epigraph_obj_and_grad(params, beta):
    """
    Epigraph formulation: minimize t subject to R_k >= t for all k.
    Using soft constraint: t + (1/beta)*log(sum(exp(beta*(R_k - t))))
    params: [h_0, ..., h_{n-1}, t]
    """
    n = N
    h = params[:n]
    t = params[n]
    
    corrs = _compute_overlaps(h)
    
    # Soft constraint: sum_k exp(beta*(R_k - t)) should be small
    # Objective: t + (1/beta)*log(sum(exp(beta*(corrs - t))))
    # This is smooth and encourages t to be close to min(corrs)
    
    shifted = beta * (corrs - t)
    max_shifted = np.max(shifted)
    if max_shifted > 500:  # Numerical stability
        log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    else:
        log_sum = np.log(np.sum(np.exp(shifted)))
    
    obj = t + log_sum / beta
    
    # Gradient
    # d(obj)/d(t) = 1 - (1/beta)*sum(beta*exp(beta*(R_k-t))/sum(exp(beta*(R_k-t))))
    #             = 1 - sum(exp(beta*(R_k-t))/sum(exp(beta*(R_k-t))))
    #             = 1 - 1 = 0 (when using exact min)
    # But with soft constraint, it's more complex
    
    # d(obj)/d(h_i) = (1/beta) * sum_k [beta*exp(beta*(R_k-t))/sum(exp(beta*(R_k-t)))] * dR_k/dh_i
    #               = sum_k [exp(beta*(R_k-t))/sum(exp(beta*(R_k-t)))] * dR_k/dh_i
    
    exp_shifted = np.exp(shifted - max_shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp
    
    # dR_k/dh_i where R_k = (2/n)*sum_j h[j]*h[j+k]
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        if w == 0: continue
        grad_h[:n-k] += w * h[k:]
        grad_h[k:] += w * h[:n-k]
    
    # d(obj)/d(t): 
    # obj = t + (1/beta)*log(sum(exp(beta*(corrs-t))))
    # d/dt = 1 + (1/beta)*sum(-beta*exp(beta*(corrs-t)))/sum(exp(beta*(corrs-t)))
    #       = 1 - sum(exp(beta*(corrs-t)))/sum(exp(beta*(corrs-t)))
    #       = 1 - 1 = 0
    # Wait, that's not right. Let me recalculate.
    # d/dt [t + (1/beta)*log(sum_j exp(beta*(R_j - t)))]
    # = 1 + (1/beta) * [sum_j (-beta)*exp(beta*(R_j-t))] / sum_j exp(beta*(R_j-t))
    # = 1 - 1 = 0
    # So the gradient w.r.t. t is actually 0? That can't be right for optimization.
    
    # Actually, the epigraph formulation should be:
    # minimize t
    # subject to R_k >= t for all k
    # Using penalty: t + (1/beta)*sum_k max(0, t - R_k)^2
    # Or using log-sum-exp: t + (1/beta)*log(sum_k exp(beta*(t - R_k)))
    
    # Let me use: obj = t + (1/beta)*log(sum_k exp(beta*(t - R_k)))
    # This is convex in t and encourages t to be below min(R_k)
    
    shifted2 = beta * (t - corrs)
    max_shifted2 = np.max(shifted2)
    if max_shifted2 > 500:
        log_sum2 = max_shifted2 + np.log(np.sum(np.exp(shifted2 - max_shifted2)))
    else:
        log_sum2 = np.log(np.sum(np.exp(shifted2)))
    
    obj = t + log_sum2 / beta
    
    # Gradient w.r.t. t:
    # d/dt [t + (1/beta)*log(sum_k exp(beta*(t-R_k)))]
    # = 1 + (1/beta)*sum_k beta*exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k))
    # = 1 + 1 = 2
    # That's also not right.
    
    # Let me think differently. The standard epigraph for min_k f_k is:
    # minimize t
    # s.t. f_k <= t for all k
    # Using soft: minimize t + (1/beta)*sum_k log(1 + exp(beta*(f_k - t)))
    # Or: minimize t + (1/beta)*log(sum_k exp(beta*(f_k - t)))
    
    # For maximize min_k R_k, we want to maximize t s.t. R_k >= t
    # Equivalent to minimize -t s.t. t - R_k <= 0
    # Soft: minimize -t + (1/beta)*log(sum_k exp(beta*(t - R_k)))
    
    obj = -t + log_sum2 / beta
    
    # d(obj)/d(t) = -1 + (1/beta)*sum_k beta*exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k))
    #             = -1 + 1 = 0
    # Still 0? That's because at the optimum, the constraint is active.
    
    # Actually, for the soft formulation, the gradient is:
    # d/dt [log(sum_k exp(beta*(t-R_k)))] = sum_k beta*exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k))
    # So d(obj)/d(t) = -1 + sum_k exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k)) = -1 + 1 = 0
    
    # This means the gradient w.r.t. t is always 0, which is problematic.
    
    # Let me use a different approach: minimize max_k (t - R_k)
    # Using soft-max: (1/beta)*log(sum_k exp(beta*(t - R_k)))
    # This is smooth and we want to minimize it.
    
    obj = log_sum2 / beta
    
    # d(obj)/d(t) = sum_k exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k)) = 1
    # d(obj)/d(h_i) = -sum_k exp(beta*(t-R_k))/sum_k exp(beta*(t-R_k)) * dR_k/dh_i
    #               = -sum_k weights_k * dR_k/dh_i
    
    exp_shifted2 = np.exp(shifted2 - max_shifted2)
    sum_exp2 = np.sum(exp_shifted2)
    weights2 = exp_shifted2 / sum_exp2
    
    grad_h = np.zeros(n)
    for k in range(1, n):
        w = weights2[k-1] * (2.0 / n)
        if w == 0: continue
        grad_h[:n-k] -= w * h[k:]
        grad_h[k:] -= w * h[:n-k]
    
    grad_t = 1.0
    
    return obj, np.concatenate([grad_h, [grad_t]])

def construct():
    n = N
    target_sum = n / 2.0
    
    best_h = np.full(n, 0.5)
    best_score = float('inf')
    
    def _score(h):
        corrs = _compute_overlaps(h)
        return 1.0 - np.min(corrs)
    
    best_score = _score(best_h)
    
    np.random.seed(42)
    
    # Generate diverse restarts
    restarts = []
    
    # Uniform
    restarts.append(np.full(n, 0.5))
    
    # Linear gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    
    # Sinusoidal
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4, 5]:
        for amp in [0.3, 0.4, 0.5]:
            restarts.append(np.clip(0.5 + amp*np.sin(freq*x), 0, 1))
            restarts.append(np.clip(0.5 + amp*np.cos(freq*x), 0, 1))
    
    # Random
    for _ in range(30):
        restarts.append(np.random.uniform(0, 1, n))
    
    # Block patterns
    for block_size in [2, 4, 5, 10, 20, 25, 50]:
        if block_size < n:
            pattern = np.tile([0.0, 1.0], n // block_size + 1)[:n]
            restarts.append(pattern * 0.5 + 0.25)
    
    # Bang-bang patterns (0 or 1)
    for _ in range(20):
        pattern = np.random.choice([0, 1], size=n)
        restarts.append(pattern.astype(float))
    
    # Project all restarts
    projected_restarts = []
    for r in restarts:
        pr = _project_to_simplex(r, target_sum)
        projected_restarts.append(pr)
    
    # Optimize each restart
    for x0 in projected_restarts:
        if _time_remaining() < 3.0:
            break
        
        # Multi-stage optimization with increasing beta
        current = x0.copy()
        
        for beta in [10.0, 30.0, 100.0, 300.0, 1000.0]:
            if _time_remaining() < 1.5:
                break
            
            try:
                # Use L-BFGS-B on the epigraph formulation
                # params = [h_0, ..., h_{n-1}, t]
                # Initial t estimate
                corrs_init = _compute_overlaps(current)
                t_init = np.min(corrs_init)
                
                init_params = np.concatenate([current, [t_init]])
                
                res = minimize(
                    _epigraph_obj_and_grad,
                    init_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n + [(-1, 2)],
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
                )
                
                if res.success:
                    h_new = _project_to_simplex(res.x[:n], target_sum)
                    score_new = _score(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new
                        current = h_new
            except Exception:
                pass
    
    # Final high-precision polish
    if _time_remaining() > 5.0:
        try:
            corrs_init = _compute_overlaps(best_h)
            t_init = np.min(corrs_init)
            init_params = np.concatenate([best_h, [t_init]])
            
            for beta in [500.0, 2000.0, 5000.0]:
                if _time_remaining() < 2.0:
                    break
                res = minimize(
                    _epigraph_obj_and_grad,
                    init_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n + [(-1, 2)],
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                if res.success:
                    h_new = _project_to_simplex(res.x[:n], target_sum)
                    score_new = _score(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new
                        init_params = np.concatenate([best_h, [np.min(_compute_overlaps(best_h))]])
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
