import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin

    def _corr_value(h):
        # Compute S_k - C_k = corr(h, 1-h)
        return np.correlate(h, 1.0 - h, mode="full")

    def _smooth_obj(h_t, beta):
        h = h_t[:N]
        t = h_t[N]
        corr = _corr_value(h)
        # Soft max: t + (1/beta) log sum exp(beta (corr - t))
        # To avoid overflow, shift by max
        max_corr = np.max(corr)
        if max_corr > t:
            shifted = corr - max_corr
            log_sum = max_corr + np.log(np.sum(np.exp(beta * shifted))) / beta
            return t + log_sum - t # Wait, objective is t + soft_max(corr - t)?
            # Actually we want to minimize t subject to corr <= t.
            # Penalty method: t + (1/beta) log sum exp(beta (corr - t))
            # If corr <= t, exp(beta(corr-t)) is small.
            # Let's just minimize t + (1/beta) log sum exp(beta (corr_k - t))
            return log_sum
        else:
            # If max_corr <= t, then all corr_k <= t.
            # The penalty is small.
            shifted = corr - t
            return t + np.log(np.sum(np.exp(beta * shifted))) / beta

    # Better smooth objective:
    # Minimize t such that corr_k <= t.
    # Equivalent to minimizing max(corr_k).
    # Smooth approx: (1/beta) log sum exp(beta corr_k).
    # We don't need t in the objective if we just minimize the smooth max.
    # But we need to ensure the result is a valid t?
    # The value is max(corr)/N * 2.
    # So we just need to minimize max(corr).
    
    def _smooth_max_corr(h, beta):
        corr = _corr_value(h)
        max_corr = np.max(corr)
        shifted = corr - max_corr
        return max_corr + np.log(np.sum(np.exp(beta * shifted))) / beta

    def _objective(h):
        corr = _corr_value(h)
        return np.max(corr)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)

    np.random.seed(42)
    
    # Generate inits
    inits = []
    # Random
    for _ in range(30):
        h = np.random.uniform(0, 1, N)
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        inits.append(h)
        
    # Structured
    for k in [1, 2, 3, 5, 10]:
        for amp in [0.2, 0.4, 0.6]:
            h = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)

    for h_init in inits:
        if time.time() > deadline:
            break
        
        # Optimize smooth surrogate
        current_beta = 20.0
        h_curr = h_init.copy()
        for _ in range(3):
            res = minimize(
                lambda x: _smooth_max_corr(x, current_beta),
                h_curr,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.success:
                h_curr = res.x
            current_beta *= 2.0
            
        # Polish with exact objective
        res = minimize(
            _objective,
            h_curr,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if res.success:
            h_cand = res.x
            h_cand = np.clip(h_cand, 0, 1)
            diff = N/2.0 - np.sum(h_cand)
            if abs(diff) > 1e-9:
                h_cand += diff / N
                h_cand = np.clip(h_cand, 0, 1)
            val = _objective(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()
                
    # Final polish
    if time.time() < deadline - 5:
        res = minimize(
            _objective,
            best_h,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-14}
        )
        if res.success:
            h_cand = res.x
            h_cand = np.clip(h_cand, 0, 1)
            diff = N/2.0 - np.sum(h_cand)
            if abs(diff) > 1e-9:
                h_cand += diff / N
                h_cand = np.clip(h_cand, 0, 1)
            val = _objective(h_cand)
            if val < best_val:
                best_val = val
                best_h = h_cand.copy()

    return best_h
