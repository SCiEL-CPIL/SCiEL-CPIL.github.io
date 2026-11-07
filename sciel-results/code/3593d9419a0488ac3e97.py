import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    # Hard deadline: 110 seconds to leave 10s buffer for evaluator overhead
    deadline = start_time + 110.0

    def _objective(h):
        # The value to minimize is max(corr) / N * 2
        # Minimizing max(corr) is equivalent
        corr = np.correlate(h, 1.0 - h, mode="full")
        return np.max(corr)

    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N / 2.0}]

    # Seed result
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)

    np.random.seed(42)

    # Generate a small set of diverse initializations
    inits = []
    
    # 1. Random inits (fewer than before)
    for _ in range(5):
        h = np.random.uniform(0, 1, N)
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        inits.append(h)

    # 2. Structured inits (sinusoids)
    for k in [1, 2, 3]:
        for amp in [0.2, 0.4, 0.6]:
            h = 0.5 + amp * np.cos(2 * np.pi * k * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            inits.append(h)

    # 3. Block inits
    for block_size in [10, 20, 25]:
        h = np.zeros(N)
        h[::block_size] = 1.0
        h = np.clip(h, 0, 1)
        h += (N/2.0 - np.sum(h)) / N
        h = np.clip(h, 0, 1)
        inits.append(h)

    # Optimize each initialization with a strict time check
    for h_init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use SLSQP with limited iterations to ensure speed
            # maxiter=100 is usually enough for a good local optimum in this dimension
            res = minimize(
                _objective,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if res.success:
                h_cand = res.x
                # Enforce constraints strictly
                h_cand = np.clip(h_cand, 0, 1)
                diff = N/2.0 - np.sum(h_cand)
                if abs(diff) > 1e-9:
                    h_cand += diff / N
                    h_cand = np.clip(h_cand, 0, 1)
                
                val = _objective(h_cand)
                if val < best_val:
                    best_val = val
                    best_h = h_cand.copy()
                    
        except Exception:
            continue

    # Final polish on the best found
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
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
        except Exception:
            pass

    return best_h
