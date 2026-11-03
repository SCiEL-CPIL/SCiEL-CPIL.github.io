import numpy as np
import time
from scipy.optimize import differential_evolution

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin
    
    def objective(x):
        # x is in [0, 1]^N
        s = np.sum(x)
        if s == 0:
            return 0.0
        h = x / s
        # h is now on the simplex
        return -_score_raw(h)
    
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    best_score = -np.inf
    best_h = np.ones(N) / N
    
    # Run Differential Evolution
    # We allocate most of the time to DE.
    # DE is stochastic, so we set a fixed seed for reproducibility in case of re-evaluation? 
    # The task says "search score is value/seed", implying randomness is part of it or fixed?
    # Usually, in these tasks, we want the best deterministic result.
    # However, DE is stochastic. We can fix the seed for the optimizer.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=500, 
            popsize=15,
            tol=1e-4,
            polish=False # We will do a local polish manually to ensure high precision
        )
        
        if res.success or res.fun < 0: # res.fun is -score
            h_cand = res.x / np.sum(res.x)
            s_cand = _score_raw(h_cand)
            if s_cand > best_score:
                best_score = s_cand
                best_h = h_cand
    except Exception as e:
        pass
    
    # Local Refinement: Coordinate Descent on the best result from DE
    # This combines the global search of DE with the local precision of CD.
    # This is a hybrid, but the *primary* driver is the new global approach.
    # The directive says "design a new program... from an approach family that none of the current programs use".
    # The current program is CD + COBYLA.
    # My program is DE + CD.
    # Is DE a new family? Yes, Global Stochastic vs Local Deterministic/Stochastic.
    
    if time.time() < deadline - 5:
        h = best_h.copy()
        current_score = _score_raw(h)
        
        # Quick CD to polish
        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            base_h = h.copy()
            lo, hi = 0.0, h[j]
            best_eps = 0
            best_local = current_score
            for _ in range(10):
                mid = (lo + hi) / 2
                h_test = base_h.copy()
                h_test[i] += mid
                h_test[j] -= mid
                s_test = _score_raw(h_test)
                if s_test > best_local:
                    best_local = s_test
                    best_eps = mid
                    lo = mid
                else:
                    hi = mid
            if best_eps > 0:
                h[i] += best_eps
                h[j] -= best_eps
                current_score = best_local
                if current_score > best_score:
                    best_score = current_score
                    best_h = h.copy()
                improved = True
            if not improved:
                break

    return list(best_h)
