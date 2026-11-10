import numpy as np
import time
from scipy.optimize import minimize

def evaluate_measure(s, w):
    """
    Fast evaluation of P[X1 + X2 + X3 < 2*X4].
    P = sum_{k,l} p_k * p_l * sum_{i,j: s_i+s_j < 2*s_l - s_k} p_i * p_j
    """
    n = len(s)
    if n == 0:
        return 0.0

    w_sum = np.sum(w)
    if w_sum <= 1e-9:
        return 0.0
    p = w / w_sum
    p = np.maximum(p, 0.0)
    s = np.maximum(s, 0.0)

    # Re-normalize after clipping
    p_sum = np.sum(p)
    if p_sum <= 1e-9:
        return 0.0
    p = p / p_sum

    # Build pair sums and probs
    ss = s[:, None] + s[None, :]
    pp = p[:, None] * p[None, :]

    ss_flat = ss.flatten()
    pp_flat = pp.flatten()

    # Sort by sum
    sort_idx = np.argsort(ss_flat)
    ss_sorted = ss_flat[sort_idx]
    pp_sorted = pp_flat[sort_idx]

    # Cumulative sum of probs
    C = np.cumsum(pp_sorted)

    P = 0.0
    for k in range(n):
        for l in range(n):
            threshold = 2.0 * s[l] - s[k]
            idx = np.searchsorted(ss_sorted, threshold, side='left')
            if idx > 0:
                prob_inner = C[idx - 1]
            else:
                prob_inner = 0.0
            P += p[k] * p[l] * prob_inner

    return P

def get_measure_from_params(params):
    """
    params: [n, log_base, weight_alpha]
    n: number of support points (integer)
    log_base: base of the geometric sequence for support
    weight_alpha: exponent for the weight distribution (higher = more concentrated at low values)
    
    Support: s_i = log_base^i, normalized to [0, 1]
    Weights: w_i proportional to (i+1)^(-weight_alpha), normalized
    """
    n = int(np.round(params[0]))
    n = max(2, min(50, n))  # Clamp n to [2, 50]
    log_base = params[1]
    weight_alpha = params[2]
    
    # Ensure log_base > 1 for distinct points, but allow flexibility
    # If log_base is close to 1, points are clustered. If large, spread out.
    # Let's force log_base to be at least 1.01 to avoid division by zero or tiny differences
    if log_base <= 1.001:
        log_base = 1.001
        
    # Generate support
    # s_i = log_base^i
    # To keep numbers stable, we can work with log(s_i) = i * log(log_base)
    # But the metric uses sums s_i + s_j.
    # Let's generate s directly.
    # s = log_base ** np.arange(n)
    # This can overflow if log_base is large and n is large.
    # Let's normalize s to [0, 1] by dividing by max(s).
    
    # To avoid overflow, we can use:
    # s_i = exp(i * log(log_base))
    # max_s = exp((n-1) * log(log_base))
    # s_norm = s / max_s = exp((i - (n-1)) * log(log_base))
    
    log_lb = np.log(log_base)
    exponents = np.arange(n) - (n - 1)
    s = np.exp(exponents * log_lb)
    
    # s is now in [1/log_base^(n-1), 1]. All positive.
    # Note: s[0] is the smallest, s[-1] is 1.
    
    # Generate weights
    # w_i proportional to (i+1)^(-weight_alpha)
    # This puts more mass on small i (small support values).
    # Wait, the strategy says "heavy-tailed X4".
    # If X4 is large, it helps. So we want mass on LARGE support values?
    # Let's check the inequality: X1+X2+X3 < 2*X4.
    # If X4 is large, 2*X4 is large, making the inequality easier to satisfy.
    # So we want X4 to be drawn from the tail (large values).
    # Therefore, weights should be concentrated on LARGE support values.
    # My weight formula w_i ~ (i+1)^(-alpha) puts mass on small i (small s).
    # I need to reverse this or use (n-i) or similar.
    # Let's use w_i proportional to (n - i)^(-weight_alpha) ? No, that's still heavy at small i if alpha>0.
    # Let's use w_i proportional to (i+1)^(weight_alpha) ?
    # If alpha > 0, mass is on large i (large s).
    # Let's use w_i = (i+1)^weight_alpha.
    
    idx = np.arange(1, n + 1)
    w = idx ** weight_alpha
    
    # Normalize
    s_sum = np.sum(s)
    if s_sum > 0:
        s = s / s_sum # Not strictly necessary for support, but keeps scale consistent
    # Actually, support values don't need to be in [0,1], just non-negative.
    # But the problem says "support value < -1e-9" is invalid.
    # My s is positive.
    
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.ones(n) / n
        
    return s, w

def objective(params):
    try:
        s, w = get_measure_from_params(params)
        # Validate
        if np.any(s < -1e-9) or np.any(w < -1e-9):
            return -1.0
        if np.sum(w) <= 1e-9:
            return -1.0
            
        score = evaluate_measure(s, w)
        return -score  # Minimize negative score
    except Exception:
        return 1e6

def beat_the_average_measure() -> tuple:
    deadline = time.time() + 600.0
    safety_margin = 10.0
    
    best_score = -1.0
    best_s = None
    best_w = None
    
    # Initial guesses for params: [n, log_base, weight_alpha]
    # n: 10-30
    # log_base: 1.1 - 2.0
    # weight_alpha: 0 - 5 (higher = more mass on large values)
    
    # We will run Nelder-Mead on a continuous relaxation of n?
    # No, Nelder-Mead works on continuous vectors.
    # We can treat n as continuous in the objective by rounding.
    # Or we can loop over a few discrete n values and optimize the other two.
    
    # Strategy:
    # 1. Try a few discrete n values (e.g., 10, 15, 20, 25, 30)
    # 2. For each n, optimize [log_base, weight_alpha] using Nelder-Mead
    # 3. Keep the best result
    
    n_values = [10, 15, 20, 25, 30]
    
    for n in n_values:
        if time.time() > deadline - safety_margin:
            break
            
        def obj_2d(params_2d):
            full_params = [n, params_2d[0], params_2d[1]]
            return objective(full_params)
            
        # Initial guess: log_base = 1.5, weight_alpha = 2.0
        x0 = np.array([1.5, 2.0])
        
        # Bounds are not directly supported by Nelder-Mead, but we can clip in get_measure_from_params
        # Or use a penalty.
        
        try:
            res = minimize(obj_2d, x0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 100, 'maxfev': 200})
            
            if res.success or res.fun < 1e5:
                s, w = get_measure_from_params([n, res.x[0], res.x[1]])
                score = evaluate_measure(s, w)
                
                if score > best_score:
                    best_score = score
                    best_s = s.copy()
                    best_w = w.copy()
                    
        except Exception:
            pass

    # If no good result found, use a fallback
    if best_s is None:
        # Fallback: simple geometric sequence
        n = 15
        s = 1.5 ** np.arange(n)
        s = s / s[-1]
        w = np.ones(n) / n
        best_s = s
        best_w = w
        best_score = evaluate_measure(s, w)

    # Final validation
    best_s = np.asarray(best_s, dtype=np.float64).flatten()
    best_w = np.asarray(best_w, dtype=np.float64).flatten()
    
    best_s = np.maximum(best_s, 0.0)
    best_w = np.maximum(best_w, 0.0)
    
    w_sum = np.sum(best_w)
    if w_sum > 1e-9:
        best_w = best_w / w_sum
    else:
        best_w = np.ones(len(best_s)) / len(best_s)
        
    return best_s, best_w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(f"Score: {evaluate_measure(s, w)}")
    print(f"n = {len(s)}")
