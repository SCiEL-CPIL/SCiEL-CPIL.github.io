import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _proxy_score(h):
    s = np.sum(h)
    if s < 1e-9:
        return 0.0
    h = h / s
    c = np.convolve(h, h, mode='full')
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    max_c = np.max(c)
    if max_c < 1e-9:
        return 0.0
    return (l2_sq / (l1 * max_c))

def construct():
    t0 = time.time()
    deadline = t0 + 120  # Safety margin before 150s kill
    
    bounds = [(0.0, None)] * N
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
    
    def objective(h):
        return -_proxy_score(h)
    
    rng = np.random.default_rng(42)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Try single spike at each position
    for i in range(N):
        h_spike = np.zeros(N)
        h_spike[i] = 1.0
        score = _proxy_score(h_spike)
        if score > best_score:
            best_score = score
            best_h = h_spike.copy()
    
    # Try pairs of spikes
    for i in range(N):
        if time.time() > deadline - 30:
            break
        for j in range(i+1, N):
            for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
                h_pair = np.zeros(N)
                h_pair[i] = a
                h_pair[j] = 1.0 - a
                score = _proxy_score(h_pair)
                if score > best_score:
                    best_score = score
                    best_h = h_pair.copy()
    
    # Multi-start local optimization
    initializations = []
    
    # Best found so far
    initializations.append(best_h.copy())
    
    # Single spikes
    for i in [0, 1, 2, N//2, N-1]:
        h_spike = np.zeros(N)
        h_spike[i] = 1.0
        initializations.append(h_spike)
    
    # Pairs of spikes at various positions
    for i in range(0, N, 5):
        for j in range(i+1, min(i+10, N)):
            h_pair = np.zeros(N)
            h_pair[i] = 0.5
            h_pair[j] = 0.5
            initializations.append(h_pair)
    
    # Random sparse supports
    for k in range(2, 8):
        idx = rng.choice(N, k, replace=False)
        w = rng.random(k)
        w = w / np.sum(w)
        h_init = np.zeros(N)
        h_init[idx] = w
        initializations.append(h_init)
    
    # Optimize each initialization
    for h_init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            res = minimize(
                objective,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if res.success:
                h_opt = res.x
                h_opt[h_opt < 0] = 0
                s = np.sum(h_opt)
                if s > 1e-9:
                    h_opt = h_opt / s
                    score_opt = _proxy_score(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final refinement
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.success:
                h_opt = res.x
                h_opt[h_opt < 0] = 0
                s = np.sum(h_opt)
                if s > 1e-9:
                    h_opt = h_opt / s
                    score_opt = _proxy_score(h_opt)
                    if score_opt > best_score:
                        best_score = score_opt
                        best_h = h_opt.copy()
        except Exception:
            pass
    
    return [float(x) for x in best_h]
