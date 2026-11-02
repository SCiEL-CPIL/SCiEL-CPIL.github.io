"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

_START_TIME = time.time()
_TIME_LIMIT = 135.0  # safety margin before 150s kill

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

def _soft_min_cross_correlation(h, beta=100.0):
    """Compute a smooth approximation of min_k R(2k/n) using log-sum-exp.
    
    soft_min = -log(sum(exp(-beta * R_k))) / beta
    
    This is differentiable and approximates the hard min as beta -> inf.
    """
    n = len(h)
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Use log-sum-exp for numerical stability
    min_val = np.min(corrs)
    shifted = corrs - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    return soft_min, corrs

def _objective_and_grad_soft(x, beta=100.0):
    """Objective and gradient using soft-min approximation."""
    h = x
    n = len(h)
    
    # Compute all correlations
    corrs = np.zeros(n - 1)
    for k in range(1, n):
        corrs[k-1] = np.dot(h[:n-k], h[k:]) * (2.0 / n)
    
    # Soft min
    min_val = np.min(corrs)
    shifted = corrs - min_val
    exp_terms = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min = min_val - log_sum / beta
    
    obj = 1.0 - soft_min
    
    # Gradient of soft_min w.r.t. corrs:
    # d(soft_min)/d(corrs_k) = (1/beta) * exp(-beta*(corrs_k - min_val)) / sum_exp
    # But we also need d(corrs_k)/d(h)
    
    # Actually, let's compute gradient numerically or analytically
    # d(soft_min)/d(h) = sum_k [d(soft_min)/d(corrs_k)] * d(corrs_k)/d(h)
    
    # d(soft_min)/d(corrs_k) = -exp(-beta*(corrs_k - min_val)) / (beta * sum_exp)
    # Wait, let me re-derive:
    # soft_min = min_val - (1/beta) * log(sum_j exp(-beta*(corrs_j - min_val)))
    # This is tricky because min_val depends on h too.
    
    # Let's use a cleaner formulation:
    # soft_min = -(1/beta) * log(sum_k exp(-beta * corrs_k))
    # d(soft_min)/d(corrs_k) = exp(-beta*corrs_k) / sum_j exp(-beta*corrs_j)
    
    exp_neg_beta = np.exp(-beta * corrs)
    sum_exp_neg = np.sum(exp_neg_beta)
    weights = exp_neg_beta / sum_exp_neg  # These are the softmax weights (inverted)
    
    # d(soft_min)/d(h) = sum_k weights_k * d(corrs_k)/d(h)
    # corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
    # d(corrs_k)/d(h[j]) = (2/n) * [h[j+k] if j < n-k] + (2/n) * [h[j-k] if j >= k]
    
    grad = np.zeros(n)
    for k in range(1, n):
        w = weights[k-1] * (2.0 / n)
        # For corrs_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
        # d/dh[j] = (2/n) * h[j+k] for j in [0, n-k-1]
        #           + (2/n) * h[j-k] for j in [k, n-1]
        if w != 0:
            grad[:n-k] += w * h[k:]
            grad[k:] += w * h[:n-k]
    
    # d(obj)/d(h) = -d(soft_min)/d(h) = -grad
    grad_obj = -grad
    
    return obj, grad_obj

def _project_to_simplex(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum}."""
    n = len(v)
    
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
    for _ in range(10):  # Iterate a few times
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

def _solve_lp_for_n(n):
    """Solve the LP formulation for a given N.
    
    Returns (h, t_val) where h is the optimal step heights and t_val is the max overlap.
    """
    start_time = time.time()
    deadline = start_time + 20  # 20s per LP solve
    
    # Build LP:
    # Variables: h_0, ..., h_{n-1}, t
    # Minimize: t
    # Constraints:
    #   0 <= h_i <= 1
    #   sum h_i = n/2
    #   For each shift s in [0,2], overlap(s) <= t
    
    # Overlap for shift s:
    # If s in [0,1]: overlap = ∫₀^{1-s} h(x)dx + ∫_{1+s}² (1-h(x))dx
    # If s in [1,2]: overlap = ∫₀^{s-1} (1-h(x))dx + ∫_{2-s}² h(x)dx
    
    # For equal-width steps, step i covers [2i/n, 2(i+1)/n)
    # The overlap function is piecewise linear in s.
    # Candidate s values: s = 0, s = 1, s = 2, and s = 2k/n for k=0..n, s = 1 - 2k/n for k=0..n
    
    s_candidates = set()
    s_candidates.add(0.0)
    s_candidates.add(1.0)
    s_candidates.add(2.0)
    for k in range(n + 1):
        s1 = 2.0 * k / n
        s2 = 1.0 - 2.0 * k / n
        if 0.0 <= s1 <= 2.0:
            s_candidates.add(s1)
        if 0.0 <= s2 <= 2.0:
            s_candidates.add(s2)
    
    s_vals = sorted(s_candidates)
    
    # Build constraint matrix A_ub * x <= b_ub
    # x = [h_0, ..., h_{n-1}, t]
    # For each s: overlap(s) - t <= 0
    
    # Compute overlap(s) as linear function of h:
    # overlap(s) = sum_i c_i(s) * h_i + d(s)
    # where c_i(s) = measure of overlap of step i with [0,1-s] - measure of overlap of step i with [1+s,2] (for s in [0,1])
    # and d(s) = measure of [1+s, 2] (since 1-h contributes -h + 1)
    
    num_constraints = len(s_vals)
    A_ub = np.zeros((num_constraints, n + 1))
    b_ub = np.zeros(num_constraints)
    
    step_width = 2.0 / n
    
    for idx, s in enumerate(s_vals):
        d = 0.0
        for i in range(n):
            step_start = 2.0 * i / n
            step_end = 2.0 * (i + 1) / n
            
            if s <= 1.0:
                # overlap(s) = ∫₀^{1-s} h dx + ∫_{1+s}² (1-h) dx
                # w_i = measure(step ∩ [0,1-s]) - measure(step ∩ [1+s,2])
                lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = m1 - m2
                A_ub[idx, i] = w_i
                d += m2  # from (1-h) integral: contributes +1 per unit measure
            else:
                # overlap(s) = ∫₀^{s-1} (1-h) dx + ∫_{2-s}² h dx
                # w_i = -measure(step ∩ [0,s-1]) + measure(step ∩ [2-s,2])
                lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                w_i = -m1 + m2
                A_ub[idx, i] = w_i
                d += m1  # from (1-h) integral: contributes +1 per unit measure
        
        # Constraint: sum_i w_i * h_i + d - t <= 0
        # i.e., sum_i w_i * h_i - t <= -d
        A_ub[idx, n] = -1.0
        b_ub[idx] = -d
    
    # Equality constraint: sum h_i = n/2
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = np.array([n / 2.0])
    
    # Bounds: 0 <= h_i <= 1, t >= 0
    bounds = [(0, 1)] * n + [(0, None)]
    
    # Objective: minimize t
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        if result.success:
            t_val = result.x[n]
            h_val = result.x[:n]
            # Clamp to [0,1] and ensure sum is exactly n/2
            h_val = np.clip(h_val, 0.0, 1.0)
            # Adjust to ensure exact sum
            diff = n / 2.0 - np.sum(h_val)
            if abs(diff) > 1e-12:
                # Distribute adjustment
                h_val += diff / n
                h_val = np.clip(h_val, 0.0, 1.0)
            
            return h_val, t_val
    except Exception:
        pass
    
    return None, float('inf')

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    # Generate diverse initializations
    np.random.seed(42)
    
    restarts = []
    # Uniform
    restarts.append(np.full(n, 0.5))
    # Block structures
    for frac in [0.3, 0.4, 0.5, 0.6, 0.7]:
        k = int(n * frac)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 2.0*target_sum/(n-k))]))
        restarts.append(np.concatenate([np.full(k, 2.0*target_sum/k), np.full(n-k, 0.0)]))
    # Alternating patterns
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    # More alternating patterns
    for period in [3, 4, 5]:
        pattern = [0.0 if (i % period) < period//2 else 1.0 for i in range(n)]
        # Scale to have mean 0.5
        mean_val = np.mean(pattern)
        if mean_val > 0:
            pattern = [p / (2*mean_val) for p in pattern]
        restarts.append(np.array(pattern))
    # Random
    for _ in range(15):
        restarts.append(np.random.uniform(0, 1, n))
    # Smooth gradients
    restarts.append(np.linspace(0, 1, n))
    restarts.append(np.linspace(1, 0, n))
    # Sinusoidal with various frequencies
    x = np.linspace(0, 2*np.pi, n)
    for freq in [1, 2, 3, 4]:
        restarts.append(0.5 + 0.3*np.sin(freq*x))
        restarts.append(0.5 + 0.3*np.cos(freq*x))
    # Two-tone patterns
    for ratio in [0.2, 0.3, 0.4]:
        k = int(n * ratio)
        restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, 1.0/(1-ratio))]))
    
    # Try LP for current N first - this often gives a strong starting point
    if _time_remaining() > 30.0:
        lp_h, lp_t = _solve_lp_for_n(n)
        if lp_h is not None:
            lp_obj = 1.0 - _compute_cross_correlation_min(lp_h)
            if lp_obj < best_obj:
                best_obj = lp_obj
                best_x = lp_h
            # Also add LP solution as a restart point
            restarts.append(lp_h.copy())
    
    # Optimize each restart with soft-min objective
    for x0 in restarts:
        if _time_remaining() < 5.0:
            break
        
        x = _project_to_simplex(x0, target_sum)
        
        # Use soft-min with moderate beta for smoothness
        try:
            res = minimize(
                _objective_and_grad_soft,
                x,
                args=(50.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_obj or (res.success and 1.0 - _compute_cross_correlation_min(res.x) < best_obj):
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
        except Exception:
            pass
    
    # Polish with higher beta for sharper min approximation
    if _time_remaining() > 10.0:
        for beta in [100.0, 200.0, 500.0, 1000.0]:
            if _time_remaining() < 5.0:
                break
            try:
                res = minimize(
                    _objective_and_grad_soft,
                    best_x,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                res_x = _project_to_simplex(res.x, target_sum)
                res_obj = 1.0 - _compute_cross_correlation_min(res_x)
                if res_obj < best_obj:
                    best_obj = res_obj
                    best_x = res_x
            except Exception:
                pass
    
    # Final polish with hard min using subgradient
    if _time_remaining() > 10.0:
        step_size = 0.01
        for _ in range(100):
            if _time_remaining() < 5.0:
                break
            h = best_x
            n = len(h)
            
            # Find which k achieves the minimum
            min_corr = float('inf')
            k_star = 1
            for k in range(1, n):
                corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
                if corr < min_corr:
                    min_corr = corr
                    k_star = k
            
            # Subgradient step
            grad = np.zeros(n)
            for j in range(n - k_star):
                grad[j] += (2.0 / n) * h[j + k_star]
                grad[j + k_star] += (2.0 / n) * h[j]
            
            # Take a small step in the negative gradient direction
            h_new = h - step_size * (-grad)  # We want to increase min_corr, so move in +grad direction
            h_new = _project_to_simplex(h_new, target_sum)
            
            new_obj = 1.0 - _compute_cross_correlation_min(h_new)
            if new_obj < best_obj:
                best_obj = new_obj
                best_x = h_new
                step_size = min(step_size * 1.1, 0.1)
            else:
                step_size *= 0.5
    
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
