import numpy as np
import time
from scipy.optimize import linprog


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
    dominance_count = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if P[j] >= P[i] and np.all(w[:, j] <= w[:, i]):
                if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                    dominance_count[i] += 1
    
    tau = 2.0
    pareto_weight = np.exp(-dominance_count / tau)
    
    # LP relaxation to get dual prices (shadow prices)
    # Maximize sum(P_i * x_i) subject to w[:,i]*x_i <= cap for each constraint
    # This is equivalent to minimizing -sum(P_i * x_i)
    c_obj = -P
    A_ub = w.T  # shape (C, n), each row is a constraint
    b_ub = np.full(C, cap)
    
    # Bounds for x_i: 0 <= x_i <= 1
    bounds = [(0, 1)] * n
    
    try:
        lp_result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if lp_result.success and hasattr(lp_result, 'ineqlin'):
            # Dual prices are in lp_result.ineqlin.marginals (negative for maximization)
            # Actually, for linprog minimizing, marginals are the dual variables
            # For maximization (we negated), the duals are -lp_result.ineqlin.marginals
            dual_prices = -lp_result.ineqlin.marginals if hasattr(lp_result.ineqlin, 'marginals') else np.zeros(C)
        else:
            dual_prices = np.zeros(C)
    except Exception:
        dual_prices = np.zeros(C)
    
    # Ensure dual prices are non-negative
    dual_prices = np.maximum(dual_prices, 0)
    
    # Reduced cost: prize minus weighted sum of weights by dual prices
    # For item i: reduced_cost_i = P_i - sum_k(dual_price_k * w_k_i)
    # Higher reduced cost means more "profitable" relative to resource cost
    reduced_cost = P - w.T @ dual_prices
    
    # The heuristic should favor items with high reduced cost
    # Also incorporate efficiency metrics
    candidates = [
        # Original candidates
        sanitize((P / (wmax + eps)) ** 2),
        sanitize(P / (wmax + eps)),
        sanitize((P / (wsum + eps)) ** 2),
        sanitize(P / (wsum + eps)),
        sanitize(P / (wmean + eps)),
        sanitize((P / (wmax + eps)) * (1.0 / (1.0 + wsum / C))),
        sanitize(P ** 2 / (wmax + eps)),
        sanitize(P / (wmax + eps) * (1.0 - wsum / (C * 2.0 + eps))),
        sanitize((P / (wmax + eps)) ** 2 * pareto_weight),
        sanitize((P / (wmax + eps)) * pareto_weight),
        sanitize((P / (wsum + eps)) ** 2 * pareto_weight),
        sanitize((P / (wsum + eps)) * pareto_weight),
        sanitize(P / (wmean + eps) * pareto_weight),
        sanitize((P / (wmax + eps)) ** 3 * pareto_weight),
        sanitize((P / (wmax + eps)) ** 1.5 * pareto_weight),
        # LP-based candidates
        sanitize(np.maximum(reduced_cost, 0) ** 2),
        sanitize(np.maximum(reduced_cost, 0)),
        sanitize(np.maximum(reduced_cost, 0) * pareto_weight),
        sanitize((np.maximum(reduced_cost, 0) / (wmax + eps)) ** 2),
        sanitize(np.maximum(reduced_cost, 0) / (wmax + eps)),
        sanitize((P / (wmax + eps)) ** 2 * np.maximum(reduced_cost / (P + eps), 0.1)),
        sanitize((P / (wmax + eps)) * np.maximum(reduced_cost / (P + eps), 0.1)),
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
        
        # Try LP-based refinements
        if time.time() < deadline - 30.0:
            rc_pos = np.maximum(reduced_cost, 0)
            for alpha in [0.5, 1.0, 1.5, 2.0]:
                if time.time() > deadline - 30.0:
                    break
                h = sanitize((rc_pos ** alpha) * pareto_weight)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
            
            # Mix LP and ratio-based
            for mix_w in [0.25, 0.5, 0.75]:
                if time.time() > deadline - 30.0:
                    break
                h_raw = mix_w * (P / (wmax + eps)) ** 2 + (1 - mix_w) * (rc_pos / (rc_pos.max() + eps))
                h = sanitize(h_raw * pareto_weight)
                score = fast_colony_sim(h, 42)
                if score > best_score:
                    best_score = score
                    best_h = h
    
    return best_h
