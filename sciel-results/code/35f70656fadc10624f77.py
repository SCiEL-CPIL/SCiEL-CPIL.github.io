"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h.
    
    For step function h on [0,2] with N steps of width 2/N:
    h_i is the height on step i (i=0..N-1), sum(h_i) = N/2 (integral 1 over [0,2]).
    
    The shifted 1-h: (1-h)(x) = 1 - h(x).
    Shifted by s: (1-h)(x-s) = 1 - h(x-s).
    
    Overlap at shift s: integral_0^2 h(x) * (1-h(x-s)) dx
    
    For step functions, this is a piecewise function of s.
    The maximum over all shifts s is what we want to minimize.
    
    For discrete steps of width w = 2/N, the overlap as a function of shift
    can be computed. The maximum overlap occurs at some shift.
    
    Actually, let me think about this more carefully.
    
    h is a step function: h(x) = h_i for x in [2i/N, 2(i+1)/N)
    
    (1-h)(x-s) = 1 - h(x-s)
    
    The overlap integral: integral_0^2 h(x) * (1-h(x-s)) dx
    
    For a given shift s, this is:
    sum over i of h_i * integral over step i of (1-h(x-s)) dx
    
    This is complex to compute exactly for all s. But the maximum over s
    of this overlap is bounded by... 
    
    Actually, a simpler characterization: the "maximum overlap" C5 is
    max_s integral h(x) * (1-h(x-s)) dx.
    
    Note that integral h(x)*(1-h(x-s)) dx = integral h(x) dx - integral h(x)*h(x-s) dx
    = 1 - cross_correlation(h, h)(s)
    
    Wait, that's not right either. Let me reconsider.
    
    Actually: integral_0^2 h(x) * (1-h(x-s)) dx
    = integral_0^2 h(x) dx - integral_0^2 h(x) h(x-s) dx
    = 1 - (h * h)(s)  [where * denotes cross-correlation]
    
    So C5 = max_s [1 - (h*h)(s)] = 1 - min_s (h*h)(s)
    
    To minimize C5, we need to maximize min_s (h*h)(s), i.e., make the
    autocorrelation as flat as possible (minimize its minimum).
    
    For the discrete case with N steps, the autocorrelation at lag k
    (shift by k steps) is:
    (h*h)(k) = sum_i h_i * h_{i-k}  (with appropriate boundary handling)
    
    But the shift s can be any real number, not just integer multiples of step width.
    However, for step functions, the cross-correlation is piecewise linear in s,
    so its extrema occur at the breakpoints, which are at multiples of the step width.
    
    So we only need to check shifts that are integer multiples of w = 2/N.
    
    For shift by k steps (k = 0, 1, ..., N-1):
    (h*h)(k) = sum_{i=k}^{N-1} h_i * h_{i-k}
    
    And C5 = 1 - min_{k=0,...,N-1} (h*h)(k)
    
    We want to minimize C5 = 1 - min_k (h*h)(k), i.e., maximize min_k (h*h)(k).
    """
    # Autocorrelation at each lag
    # (h*h)(k) = sum_{i=k}^{N-1} h_i * h_{i-k}
    n = len(h)
    min_corr = float('inf')
    for k in range(n):
        corr = np.dot(h[k:], h[:n-k])
        if corr < min_corr:
            min_corr = corr
    return 1.0 - min_corr


def _overlap_max_grad(h):
    """Gradient of the max overlap with respect to h.
    
    C5 = 1 - min_k (h*h)(k)
    
    Let m = argmin_k (h*h)(k). Then dC5/dh = -d(h*h)(m)/dh.
    
    (h*h)(k) = sum_{i=k}^{N-1} h_i * h_{i-k}
    d(h*h)(k)/dh_j = h_{j-k} (if j >= k) + h_{j+k} (if j+k < N)
    (with appropriate terms)
    
    More precisely:
    d/dh_j [sum_{i=k}^{N-1} h_i h_{i-k}]
    = h_{j-k} * [j-k >= 0 and j >= k]  -- wait, let me be more careful.
    
    (h*h)(k) = sum_{i=k}^{N-1} h_i * h_{i-k}
    
    For a fixed j, h_j appears in this sum when:
    - i = j (requires j >= k): contributes h_{j-k}
    - i-k = j, i.e., i = j+k (requires j+k < N): contributes h_{j+k}
    
    So d(h*h)(k)/dh_j = h_{j-k} * [j >= k] + h_{j+k} * [j+k < N]
    
    For the soft-max version, we use log-sum-exp.
    """
    n = len(h)
    # Compute all autocorrelations
    corrs = np.zeros(n)
    for k in range(n):
        corrs[k] = np.dot(h[k:], h[:n-k])
    
    # C5 = 1 - min(corrs)
    # To handle non-differentiability, use soft-min via log-sum-exp
    # soft_min(corrs, beta) = -logsumexp(-beta * corrs) / beta
    # For the gradient, we use the active set or soft approximation
    
    # Use soft-min with high beta for smoothness
    beta = 100.0
    neg_beta_corrs = -beta * corrs
    max_val = np.max(neg_beta_corrs)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_corrs - max_val)))
    soft_min = -log_sum_exp / beta
    
    # Gradient of soft_min with respect to corrs
    weights = np.exp(neg_beta_corrs - max_val) / np.sum(np.exp(neg_beta_corrs - max_val))
    d_soft_min_d_corrs = -weights / beta
    
    # Now dC5/dh = -sum_k d_soft_min_d_corrs[k] * d(h*h)(k)/dh
    grad = np.zeros(n)
    for k in range(n):
        w = d_soft_min_d_corrs[k]
        if w == 0:
            continue
        # d(h*h)(k)/dh_j for each j
        # h_j appears as h_i when i=j (if j>=k) with coeff h_{j-k}
        # h_j appears as h_{i-k} when i=j+k (if j+k<N) with coeff h_{j+k}
        for j in range(k, n):
            grad[j] += w * h[j-k]  # from i=j term
        for j in range(0, n-k):
            grad[j] += w * h[j+k]  # from i=j+k term
    
    return 1.0 - soft_min, -grad  # C5 value and dC5/dh


def _soft_overlap(h, beta=100.0):
    """Compute soft-max overlap for SLSQP."""
    n = len(h)
    corrs = np.zeros(n)
    for k in range(n):
        corrs[k] = np.dot(h[k:], h[:n-k])
    
    # C5 = 1 - min(corrs), use soft-min
    neg_beta_corrs = -beta * corrs
    max_val = np.max(neg_beta_corrs)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_corrs - max_val)))
    soft_min = -log_sum_exp / beta
    
    return 1.0 - soft_min


def _soft_overlap_grad(h, beta=100.0):
    """Gradient of soft overlap."""
    n = len(h)
    corrs = np.zeros(n)
    for k in range(n):
        corrs[k] = np.dot(h[k:], h[:n-k])
    
    neg_beta_corrs = -beta * corrs
    max_val = np.max(neg_beta_corrs)
    exp_vals = np.exp(neg_beta_corrs - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    soft_min = -(max_val + np.log(sum_exp)) / beta
    C5 = 1.0 - soft_min
    
    # dC5/dh = -sum_k weights[k]/beta * d(h*h)(k)/dh
    grad = np.zeros(n)
    for k in range(n):
        w = weights[k] / beta
        if w == 0:
            continue
        for j in range(k, n):
            grad[j] += w * h[j-k]
        for j in range(0, n-k):
            grad[j] += w * h[j+k]
    
    return C5, -grad


def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Initial guess: try a few structured patterns
    # The optimal should have autocorrelation as flat as possible
    # For N=100, a good starting point might be alternating or periodic patterns
    
    # Try to use SLSQP with the soft-max objective
    def objective_and_grad(h):
        if time.time() > deadline - 5:
            return _overlap_max(h), np.zeros(N)
        return _soft_overlap_grad(h, beta=50.0)
    
    # Constraints: sum(h) = N/2, 0 <= h_i <= 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2, 
                    'jac': lambda h: np.ones(N)}]
    bounds = [(0, 1)] * N
    
    # Start from uniform
    h0 = np.full(N, 0.5)
    
    try:
        result = minimize(
            lambda h: objective_and_grad(h)[0],
            h0,
            jac=lambda h: objective_and_grad(h)[1],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.fun < best_score:
            h_opt = result.x
            # Project to satisfy constraints exactly
            h_opt = np.clip(h_opt, 0, 1)
            # Adjust to satisfy sum constraint
            diff = N/2 - np.sum(h_opt)
            h_opt += diff / N
            h_opt = np.clip(h_opt, 0, 1)
            # Final adjustment
            diff = N/2 - np.sum(h_opt)
            if abs(diff) > 1e-12:
                # Distribute difference
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
            
            score = _overlap_max(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        
        # Polish with higher beta
        if time.time() < deadline - 10:
            for beta in [100.0, 500.0, 2000.0]:
                h_curr = best_h.copy()
                for it in range(100):
                    if time.time() > deadline - 5:
                        break
                    C5, grad = _soft_overlap_grad(h_curr, beta=beta)
                    # Projected gradient step
                    step_size = 0.01
                    h_new = h_curr - step_size * grad
                    # Project to box constraints
                    h_new = np.clip(h_new, 0, 1)
                    # Project to sum constraint
                    diff = N/2 - np.sum(h_new)
                    h_new += diff / N
                    h_new = np.clip(h_new, 0, 1)
                    diff = N/2 - np.sum(h_new)
                    h_new += diff / N
                    
                    if np.linalg.norm(h_new - h_curr) < 1e-10:
                        break
                    h_curr = h_new
                
                # Check improvement
                score_curr = _overlap_max(h_curr)
                if score_curr < best_score:
                    best_score = score_curr
                    best_h = h_curr.copy()
    
    except Exception as e:
        pass
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
