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
    deadline = t0 + 310.0
    
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute weight statistics
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # Pareto dominance filter: identify items not strictly dominated
    # An item i is dominated by j if P[j] >= P[i] and w[j,k] <= w[i,k] for all k,
    # with at least one strict inequality. We use a soft version: score based on
    # how "close" an item is to being Pareto-optimal.
    
    # Compute dominance scores: for each item, count how many other items dominate it
    # (P[j] > P[i] and w[j] <= w[i] element-wise)
    # This is O(n^2 * C) which is fine for n=100
    dominance_count = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if P[j] >= P[i] and w[j,:] <= w[i,:] with at least one strict
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    
    # Items with dominance_count=0 are Pareto-optimal; assign them high scores
    # Use a smooth penalty: exp(-dominance_count / tau)
    tau = 2.0
    pareto_weight = np.exp(-dominance_count / tau)
    
    # Base candidates
    candidates = [
        sanitize((P / (wmax + eps)) ** 2),
        sanitize(P / (wmax + eps)),
        sanitize((P / (wsum + eps)) ** 2),
        sanitize(P / (wsum + eps)),
        sanitize(P / (wmean + eps)),
        sanitize((P / (wmax + eps)) * (1.0 / (1.0 + wsum / C))),
        sanitize(P ** 2 / (wmax + eps)),
        sanitize(P / (wmax + eps) * (1.0 - wsum / (C * 2.0 + eps))),
        # Pareto-filtered candidates
        sanitize((P / (wmax + eps)) ** 2 * pareto_weight),
        sanitize((P / (wmax + eps)) * pareto_weight),
        sanitize((P / (wsum + eps)) ** 2 * pareto_weight),
        sanitize((P / (wsum + eps)) * pareto_weight),
        sanitize(P / (wmean + eps) * pareto_weight),
        # Stronger Pareto emphasis
        sanitize((P / (wmax + eps)) ** 3 * pareto_weight),
        sanitize((P / (wmax + eps)) ** 1.5 * pareto_weight),
    ]
    
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
    
    # Evaluate candidates
    best_h = candidates[0]
    best_score = -1.0
    
    for i, h in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        score = fast_colony_sim(h, 42)
        if score > best_score:
            best_score = score
            best_h = h
    
    # Try to refine: mix top candidates or adjust exponents
    if time.time() < deadline - 60.0:
        # Try power variations of the best base with Pareto weighting
        base_ratio = P / (wmax + eps)
        for alpha in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize((base_ratio ** alpha) * pareto_weight)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try different tau values for Pareto weighting
        for tau_val in [0.5, 1.0, 1.5, 3.0, 5.0]:
            if time.time() > deadline - 30.0:
                break
            pw = np.exp(-dominance_count / tau_val)
            h = sanitize((P / (wmax + eps)) ** 2 * pw)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
        
        # Try mixing with sum-based and Pareto
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            if time.time() > deadline - 30.0:
                break
            h = sanitize((P / (wmax + eps)) ** alpha * (1.0 / (1.0 + wsum / C)) * pareto_weight)
            score = fast_colony_sim(h, 42)
            if score > best_score:
                best_score = score
                best_h = h
    
    return best_h
