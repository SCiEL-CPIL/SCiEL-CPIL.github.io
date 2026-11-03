import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Find a large Sidon set in {0, ..., N-1}
    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_set(N):
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        return S

    sidon_set = find_sidon_set(N)
    K = len(sidon_set)
    sorted_sidon = sorted(sidon_set)

    # 2. Initialize best_h with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sorted_sidon] = 1.0 / K
    best_score = _score_raw(best_h)

    # 3. Optimization Strategy:
    # The problem is to maximize J(h) = sum(h^2) / max_conv subject to sum(h)=1, h>=0.
    # This is equivalent to minimizing max_conv / sum(h^2).
    # Note that max_conv is a convex function of h (max of bilinear terms, which are not convex, 
    # but the maximum of them... wait. 
    # Actually, the set {h : max_conv(h) <= t} is not convex because max_conv is not convex.
    # However, we can use a smoothed approximation of the max function to create a smooth objective.
    # Or, we can use SLSQP on the original non-smooth objective, relying on its ability to handle
    # kinks if we provide good gradients or if the step size is small enough.
    
    # A better approach: Use the log-sum-exp trick to smooth the max operator.
    # soft_max(hc) = (1/beta) * log(sum(exp(beta * hc)))
    # As beta -> infinity, soft_max -> max.
    # The function f(h) = sum(h^2) / soft_max(conv(h,h)) is smooth for finite beta.
    # We can maximize this with SLSQP.
    # We can also try to minimize the reciprocal: soft_max(conv(h,h)) / sum(h^2).
    # Let's maximize the smoothed ratio.

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N

    # Phase 1: Smoothed optimization with increasing beta
    # Start with low beta for broad search, increase for precision.
    betas = [10.0, 50.0, 200.0, 1000.0]
    
    current_h = best_h.copy()
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        def smooth_obj(h):
            l1 = np.sum(h)
            if l1 < 1e-12: return 0
            l2_sq = np.sum(h**2)
            if l2_sq < 1e-12: return 0
            hc = np.convolve(h, h, mode='full')
            # Soft max
            scaled = beta * hc
            max_scaled = np.max(scaled)
            soft_max_val = (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))) / beta
            if soft_max_val < 1e-12: return 0
            # We want to maximize l2_sq / soft_max_val
            # So we minimize - (l2_sq / soft_max_val)
            return -(l2_sq / soft_max_val)

        def smooth_grad(h):
            # Numerical gradient is safer here given the complexity, or we can derive it.
            # SLSQP can use numerical gradients if not provided, but it's slow.
            # Let's rely on SLSQP's numerical gradient for now, or provide a sparse one if needed.
            # Actually, for N=50, numerical gradient is 51 evaluations per step.
            # With 1000 steps, that's 51000 convolutions.
            # Convolution is O(N^2) = 2500 ops. Total ~1.25e8 ops. Should be fast in numpy.
            return None # Let SLSQP compute it numerically

        try:
            res = minimize(smooth_obj, current_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
                    current_h = h_candidate
                else:
                    # If it didn't improve, keep the previous best as current for next beta
                    current_h = best_h
        except Exception as e:
            # If optimization fails, continue with next beta
            pass

    # Phase 2: Polish with exact max objective
    # The exact objective is non-smooth, but SLSQP can often handle it if we start close to the optimum.
    # We use the result from the highest beta as the starting point.
    
    def exact_obj(h):
        l1 = np.sum(h)
        if l1 < 1e-12: return 0
        l2_sq = np.sum(h**2)
        if l2_sq < 1e-12: return 0
        hc = np.convolve(h, h, mode='full')
        mc = np.max(hc)
        if mc < 1e-12: return 0
        return -(l2_sq / mc)

    if time.time() < deadline - 2:
        try:
            res = minimize(exact_obj, best_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-14})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
