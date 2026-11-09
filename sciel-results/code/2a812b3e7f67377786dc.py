import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(f):
    """Computes the full linear autoconvolution of f."""
    return np.convolve(f, f, mode='full')

def _soft_max_abs_ac(f, beta=50.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient)
    """
    ac = _autoconv(f)
    abs_ac = np.abs(ac)
    # Log-sum-exp for numerical stability
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    # log(sum(exp(beta * abs_ac))) / beta approximates max(abs_ac)
    # For numerical stability: log(sum(exp(beta*(abs_ac - max_ac))))/beta + max_ac
    shifted = beta * (abs_ac - max_ac)
    log_sum = np.log(np.sum(np.exp(shifted))) / beta + max_ac
    # Gradient: d/d f_i of log_sum
    # = (1/beta) * sum_k (beta * sign(ac[k]) * exp(beta*(|ac[k]|-max)) / sum(exp(...)) * d|ac[k]|/df_i)
    # = sum_k (sign(ac[k]) * exp(beta*(|ac[k]|-max)) / sum(exp(...)) * d|ac[k]|/df_i)
    
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(f)
    for k in range(len(ac)):
        if abs_ac[k] < 1e-30:
            continue
        sgn = 1.0 if ac[k] >= 0 else -1.0
        w = weights[k]
        # d|ac[k]|/df[i] = sgn * 2 * f[k-i] if 0 <= k-i < N
        i_vals = k - np.arange(N)
        valid = (i_vals >= 0) & (i_vals < N)
        if np.any(valid):
            grad[i_vals[valid]] += 2.0 * sgn * f[np.arange(N)[valid]] * w
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if s == 0 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    # d/d f log(smax) = grad_smax / smax
    # d/d f (-2 log|s|) = -2 * ones / s
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = np.inf
    
    ac = _autoconv(best)
    s = np.sum(best)
    if abs(s) > 1e-12:
        best_score = np.max(np.abs(ac)) / (s * s)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips
        n_flips = np.random.randint(1, 4)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        ac = _autoconv(candidate)
        s = np.sum(candidate)
        if abs(s) > 1e-12:
            score = np.max(np.abs(ac)) / (s * s)
            if score < best_score:
                best = candidate
                best_score = score
        else:
            continue
        
        # Try small random perturbations
        if it % 3 == 0:
            candidate = best + np.random.randn(N) * 0.01
            ac = _autoconv(candidate)
            s = np.sum(candidate)
            if abs(s) > 1e-12:
                score = np.max(np.abs(ac)) / (s * s)
                if score < best_score:
                    best = candidate
                    best_score = score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_f = None
    best_score = np.inf
    
    # Initial guesses
    n_restarts = 8
    np.random.seed(42)
    
    for r in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if r == 0:
            f0 = np.ones(N)
        elif r == 1:
            f0 = np.random.randn(N)
        elif r == 2:
            f0 = np.random.choice([-1.0, 1.0], size=N)
            if np.sum(f0) < 0:
                f0 = -f0
        elif r == 3:
            # Chirp
            t = np.linspace(0, 10, N)
            f0 = np.cos(t**2)
        elif r == 4:
            # Alternating blocks
            f0 = np.ones(N)
            f0[::2] = -1.0
        elif r == 5:
            # Random with structure
            f0 = np.random.randn(N // 2)
            f0 = np.repeat(f0, 2)
        elif r == 6:
            # Sine wave
            t = np.linspace(0, 4 * np.pi, N)
            f0 = np.sin(t)
        else:
            f0 = np.random.randn(N)
        
        # Optimization
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_score:
                candidate = res.x
                # Calculate actual score
                ac = _autoconv(candidate)
                s = np.sum(candidate)
                if abs(s) > 1e-12:
                    score = np.max(np.abs(ac)) / (s * s)
                    if score < best_score:
                        best_f = candidate
                        best_score = score
                        
        except Exception as e:
            pass
        
        # Local search polish
        if best_f is not None and time.time() < deadline - 5.0:
            polished, polished_score = _local_search(best_f, deadline - 5.0, max_iters=10000)
            if polished_score < best_score:
                best_f = polished
                best_score = polished_score
    
    # If no valid result, return ones
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
