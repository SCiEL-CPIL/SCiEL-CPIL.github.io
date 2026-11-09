import numpy as np
import time

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Handle shape: expect (5, 100)
    if prizes.ndim == 1:
        P = prizes
        w = weights.reshape(5, -1) if weights.ndim == 1 else weights
    else:
        P = prizes.sum(axis=0)
        w = weights
    
    n = P.shape[0]
    C = 5
    cap = 25.0
    
    # Time budget
    t0 = time.time()
    deadline = t0 + 310.0  # Safety margin
    
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute weight statistics
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # Pareto dominance filter
    # An item i is dominated by j if P[j] >= P[i] and w[j,k] <= w[i,k] for all k,
    # with at least one strict inequality.
    dominance_count = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    
    def fast_colony_sim(h, seed, n_iter=30, n_ants=20):
        """Vectorized colony simulation for a given heuristic."""
        rng = np.random.RandomState(seed)
        pheromone = np.ones(n)
        best_total = 0.0
        
        for it in range(n_iter):
            ant_totals = np.zeros(n_ants)
            
            for a in range(n_ants):
                remaining = np.full(C, cap)
                used = np.zeros(n, dtype=bool)
                total = 0.0
                
                while True:
                    # Find items that fit
                    # w is (C, n), remaining is (C,)
                    # w <= remaining[:, None] is (C, n)
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not fits_mask.any():
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    probs = probs / probs.sum()
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = ant_totals.max()
            best_total = max(best_total, iter_best)
            pheromone = pheromone * 0.9 + iter_best / 10.0
        
        return best_total

    def robust_score(h):
        """Evaluate heuristic using multiple seeds for robustness."""
        scores = []
        for s in [42, 123, 999]:
            scores.append(fast_colony_sim(h, s))
        return np.mean(scores)

    best_h = None
    best_score = -1.0
    
    # Base weights to try
    bases = [
        ("wmax", wmax),
        ("wsum", wsum),
        ("wmean", wmean)
    ]
    
    # Grid search parameters
    alphas = np.linspace(0.5, 3.0, 11)
    taus = np.linspace(0.5, 10.0, 11)
    
    # Coarse Grid Search
    for base_name, base_w in bases:
        for alpha in alphas:
            for tau in taus:
                if time.time() > deadline - 30.0:
                    break
                
                # Pareto weight
                pw = np.exp(-dominance_count / tau)
                
                # Heuristic
                h = sanitize((P / (base_w + eps)) ** alpha * pw)
                
                score = robust_score(h)
                
                if score > best_score:
                    best_score = score
                    best_h = h
                    best_params = (base_name, alpha, tau)
                
            if time.time() > deadline - 30.0:
                break
        if time.time() > deadline - 30.0:
            break

    # Refine around the best found parameters
    if best_h is not None and time.time() < deadline - 60.0:
        base_name, alpha, tau = best_params
        
        # Find the base array again
        base_w = None
        for name, w_arr in bases:
            if name == base_name:
                base_w = w_arr
                break
        
        # Fine grid
        alpha_step = 0.1
        tau_step = 0.5
        
        for da in np.linspace(-0.5, 0.5, 11):
            for dt in np.linspace(-2.0, 2.0, 11):
                if time.time() > deadline - 30.0:
                    break
                
                new_alpha = max(0.1, alpha + da)
                new_tau = max(0.1, tau + dt)
                
                pw = np.exp(-dominance_count / new_tau)
                h = sanitize((P / (base_w + eps)) ** new_alpha * pw)
                
                score = robust_score(h)
                
                if score > best_score:
                    best_score = score
                    best_h = h
                    best_params = (base_name, new_alpha, new_tau)
                
            if time.time() > deadline - 30.0:
                break

    # Fallback if something went wrong
    if best_h is None:
        best_h = sanitize(P / (wmax + eps))
        
    return best_h
