import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """
    Computes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    Note: The problem description implies maximizing this.
    We normalize h to sum to 1 inside the evaluation to handle scale invariance,
    effectively maximizing the shape's contribution.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm**2)
    # Autoconvolution
    # h_norm is length N. h_norm * h_norm is length 2N-1.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / conv_inf

def construct():
    t0 = time.time()
    # Safety margin: return at 140s
    deadline = t0 + 140.0
    
    best_h = np.zeros(N, dtype=np.float64)
    best_score = 0.0

    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()

    # 1. Generate diverse initial candidates
    candidates = []
    
    # A. The known 2-point asymmetric optimum (ratio ~1.25)
    # h[i] = 2/3, h[i+1] = 1/3
    for i in range(N-1):
        h = np.zeros(N)
        h[i] = 2.0/3.0
        h[i+1] = 1.0/3.0
        candidates.append(h)
        
    # Also reversed
    for i in range(N-1):
        h = np.zeros(N)
        h[i] = 1.0/3.0
        h[i+1] = 2.0/3.0
        candidates.append(h)

    # B. 3-point variations
    # [a, b, c] with a+b+c=1
    # Try grid search for small support
    for i in range(N-2):
        for a in np.linspace(0.1, 0.8, 10):
            for b in np.linspace(0.1, 0.8, 10):
                c = 1.0 - a - b
                if c < 0.01: continue
                h = np.zeros(N)
                h[i] = a
                h[i+1] = b
                h[i+2] = c
                candidates.append(h)
                
    # C. Gaussian shapes (symmetric)
    x = np.arange(N) - N/2.0
    for sigma in np.linspace(1.0, 10.0, 20):
        h = np.exp(-0.5 * (x/sigma)**2)
        candidates.append(h)
        
    # D. Uniform
    h = np.ones(N)
    candidates.append(h)
    
    # E. Random sparse
    np.random.seed(42)
    for _ in range(50):
        h = np.random.exponential(1.0, N)
        # Make it sparse
        mask = np.random.rand(N) > 0.8
        h[mask] = 0
        if np.sum(h) > 0:
            candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        sc = compute_ratio(h)
        update_best(h, sc)

    # 2. Local Optimization
    # We optimize over the full vector h >= 0.
    # Objective: -compute_ratio(h)
    # We use L-BFGS-B with bounds (0, None)
    
    def objective(params):
        h = np.maximum(params, 0.0)
        if np.sum(h) < 1e-10:
            return 1e10
        return -compute_ratio(h)

    bounds = [(0.0, None) for _ in range(N)]
    
    # Select a few best starting points from candidates
    # Sort candidates by score
    scored_candidates = []
    for h in candidates:
        sc = compute_ratio(h)
        scored_candidates.append((sc, h))
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    starts = [h for _, h in scored_candidates[:10]]
    
    # Add best_h as a start
    if best_score > 0:
        starts.append(best_h)
        
    # Optimization Loop
    # We have ~130s.
    # L-BFGS-B on 50 vars is fast.
    # Let's run 5 restarts from the best starts, and 10 random restarts.
    
    num_restarts = 15
    for i in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if i < len(starts):
            start_h = starts[i]
        else:
            # Random restart
            start_h = np.random.exponential(1.0, N)
            # Add noise to best_h
            if best_score > 0:
                start_h = best_h + np.random.normal(0, 0.1, N)
                start_h = np.maximum(start_h, 0)
                
        try:
            res = minimize(objective, start_h, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success or res.fun < -best_score:
                h_opt = np.maximum(res.x, 0.0)
                sc = compute_ratio(h_opt)
                update_best(h_opt, sc)
        except Exception:
            pass

    # 3. Final Polish: Coordinate Descent on the best result
    # Sometimes L-BFGS-B gets stuck on the boundary.
    # Try small perturbations.
    if best_score > 0 and time.time() < deadline - 2.0:
        h_curr = best_h.copy()
        sc_curr = best_score
        step = 0.01
        for _ in range(200): # 200 steps of coord descent
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                # Try increasing
                h_trial = h_curr.copy()
                h_trial[i] += step
                sc_trial = compute_ratio(h_trial)
                if sc_trial > sc_curr:
                    h_curr = h_trial
                    sc_curr = sc_trial
                    improved = True
                    continue
                # Try decreasing
                if h_curr[i] > step:
                    h_trial = h_curr.copy()
                    h_trial[i] -= step
                    sc_trial = compute_ratio(h_trial)
                    if sc_trial > sc_curr:
                        h_curr = h_trial
                        sc_curr = sc_trial
                        improved = True
                        continue
            if not improved:
                break
        update_best(h_curr, sc_curr)

    return list(best_h)
