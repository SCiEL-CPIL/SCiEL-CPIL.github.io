import numpy as np
import time
from scipy.optimize import linprog

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Ensure shape is (5, 100)
    if prizes.ndim == 1:
        # If 1D, assume prizes is the sum or we need to reconstruct? 
        # The task says shapes are (5, 100). 
        # If 1D is passed, it might be flattened or just the sum?
        # The current code handles 1D by summing axis=0 if ndim>1 else using directly.
        # Let's assume standard case: ndim == 2.
        pass
        
    if prizes.ndim != 2:
        prizes = prizes.reshape(5, -1)
    if weights.ndim != 2:
        weights = weights.reshape(5, -1)
        
    P = prizes.sum(axis=0)
    w = weights
    C = w.shape[0]
    n = P.shape[0]
    cap = 25.0
    
    t0 = time.time()
    # Keep a safety margin. 330s limit, return by 310s.
    deadline = t0 + 305.0
    
    eps = 1e-9
    
    def sanitize(h):
        h = np.asarray(h, dtype=float)
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        return np.clip(h, 1e-9, 1e6)
    
    # Precompute simple stats
    wmax = w.max(axis=0)
    wsum = w.sum(axis=0)
    wmean = w.mean(axis=0)
    
    # --- LP Relaxation to get Dual Prices ---
    # Maximize P^T x subject to W x <= cap, 0 <= x <= 1
    # Minimize -P^T x
    dual_prices = np.ones(C)  # Fallback to uniform prices if LP fails
    try:
        c_obj = -P
        A_ub = w
        b_ub = np.full(C, cap)
        bounds = [(0, 1)] * n
        # Use HiGHS which is fast and provides marginals
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            # In newer SciPy/HiGHS, duals are in res.ineqlin.marginals
            # The marginals are negative of the shadow prices for minimization?
            # Actually, for min c^T x s.t. Ax <= b, the dual is max b^T y s.t. A^T y >= c.
            # y_i is the shadow price of constraint i.
            # HiGHS returns marginals which are -y_i for <= constraints in minimization?
            # Let's verify sign. If capacity increases, objective (max) increases, so shadow price > 0.
            # In min formulation, increasing RHS (relaxing constraint) decreases objective (more negative), 
            # so dObj/dRHS should be positive if it's a "cost" reduction? 
            # Standard LP duality: shadow prices are non-negative for max constraints <=.
            # SciPy linprog docs: "marginals: The marginal value of the right-hand side of the inequalities."
            # For a max problem converted to min, if we relax constraint, min value goes down (more negative).
            # So marginal is negative? 
            # Let's just take abs or check. 
            # Usually, we want lambda > 0.
            if hasattr(res, 'ineqlin') and res.ineqlin is not None and hasattr(res.ineqlin, 'marginals'):
                dual_prices = np.abs(res.ineqlin.marginals)
                if np.allclose(dual_prices, 0):
                    dual_prices = np.ones(C)
            else:
                dual_prices = np.ones(C)
    except Exception:
        dual_prices = np.ones(C)
    
    # Effective weight based on dual prices: how much "capacity cost" an item has
    # w_eff[i] = sum_j w[j, i] * lambda[j]
    w_eff_dual = w.T.dot(dual_prices)
    
    # Candidates based on LP insights and classic ratios
    candidates = [
        # 1. Prize per dual-weight (Most promising from inspiration)
        sanitize(P / (w_eff_dual + eps)),
        # 2. Classic ratio (Prize per max weight)
        sanitize(P / (wmax + eps)),
        # 3. Prize per sum weight
        sanitize(P / (wsum + eps)),
        # 4. Mixed: Prize / (dual-weight + max-weight)
        sanitize(P / (w_eff_dual + wmax + eps)),
        # 5. Prize / (dual-weight + sum-weight)
        sanitize(P / (w_eff_dual + wsum + eps)),
        # 6. Square of dual ratio
        sanitize((P / (w_eff_dual + eps)) ** 1.5),
        # 7. Square of classic ratio
        sanitize((P / (wmax + eps)) ** 1.5),
        # 8. Combination: 0.5 * dual_ratio + 0.5 * classic_ratio
        sanitize(0.5 * (P / (w_eff_dual + eps)) + 0.5 * (P / (wmax + eps))),
    ]
    
    def fast_colony_sim(h, seed, n_iter=30, n_ants=20):
        """
        Vectorized colony simulation for a given heuristic.
        Optimized for speed where possible.
        """
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
                    # w <= remaining[:, None] -> (C, n) bool
                    fits_mask = np.all(w <= remaining[:, None], axis=0) & ~used
                    if not np.any(fits_mask):
                        break
                    
                    fit_indices = np.where(fits_mask)[0]
                    probs = pheromone[fit_indices] * h[fit_indices]
                    probs = probs / probs.sum()
                    
                    idx = rng.choice(fit_indices, p=probs)
                    used[idx] = True
                    remaining -= w[:, idx]
                    total += P[idx]
                
                ant_totals[a] = total
            
            iter_best = np.max(ant_totals)
            best_total = max(best_total, iter_best)
            pheromone = pheromone * 0.9 + iter_best / 10.0
            
            # Early termination if time is tight
            if time.time() > deadline - 5.0:
                return best_total
                
        return best_total
    
    # Evaluate candidates
    best_h = candidates[0]
    best_score = -1.0
    
    # Prioritize LP-based candidates
    order = [0, 3, 4, 5, 1, 2, 7, 6]
    
    for i in order:
        if i >= len(candidates):
            continue
        if time.time() > deadline - 10.0:
            break
            
        h = candidates[i]
        score = fast_colony_sim(h, 42)
        
        if score > best_score:
            best_score = score
            best_h = h
            
    # If we have time, try a few more variations of the best base
    if time.time() < deadline - 20.0:
        # Refine around the best found (assume it's index 0 or similar)
        base_idx = 0 # The dual one is usually best
        base_h = candidates[base_idx]
        
        # Try power adjustments
        for alpha in [0.8, 1.2, 2.0]:
            if time.time() > deadline - 10.0:
                break
            P_safe = P + eps
            w_eff_safe = w_eff_dual + eps
            h_new = sanitize((P_safe / w_eff_safe) ** alpha)
            score = fast_colony_sim(h_new, 42)
            if score > best_score:
                best_score = score
                best_h = h_new
        
        # Try mixing with Pareto dominance (simple version: penalize high weights)
        # Pareto filter: if item i is worse than item j in all weights and lower prize, penalize i.
        # This is O(n^2) but n=100, so 10k ops, very fast.
        pareto_mult = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j dominates i: P[j] >= P[i] and w[j,k] <= w[i,k] for all k, with strict in at least one
                    if np.all(P[j] >= P[i]) and np.all(w[:, j] <= w[:, i]):
                        if P[j] > P[i] or np.any(w[:, j] < w[:, i]):
                            pareto_mult[i] *= 0.5 # Penalize dominated items
                            
        h_pareto = sanitize(base_h * pareto_mult)
        if time.time() < deadline - 5.0:
            score = fast_colony_sim(h_pareto, 42)
            if score > best_score:
                best_score = score
                best_h = h_pareto
                
    return best_h
