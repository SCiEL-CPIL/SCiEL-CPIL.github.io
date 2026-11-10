import time
import numpy as np

T_START = time.perf_counter()
LIMIT = 300.0
MARGIN = 10.0

def _deadline_left():
    return (T_START + LIMIT) - time.perf_counter()

def heuristics(prizes, weights):
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if prizes.ndim == 2:
        p = prizes.sum(axis=0)
    else:
        p = prizes.ravel()
    
    if weights.ndim == 1:
        w = weights.reshape(1, -1)
    elif weights.ndim == 2:
        w = weights
    else:
        w = weights.reshape(5, -1)

    n = p.shape[0]
    m = w.shape[0]
    
    if n == 0:
        return np.ones(0)

    cap = 25.0
    C = np.full(m, cap, dtype=float)

    # 1. Compute Lagrangian Multipliers via Subgradient Method
    # We want to find lambda such that the greedy solution under reduced costs (p - w.T @ lambda)
    # approximates the optimal packing density.
    lam = np.zeros(m)
    best_lam = lam.copy()
    best_val = -1.0
    
    # Hyperparameters for subgradient
    n_sub = 100
    step_init = 5.0
    # We use a decaying step size. 
    # The dual function is concave, we are maximizing it (minimizing -dual).
    # Actually, for 0-1 knapsack, the Lagrangian relaxation upper bound is:
    # max sum (p_i - sum_j lambda_j w_ij) x_i + sum_j lambda_j C_j
    # subject to 0 <= x_i <= 1 (relaxing integrality)
    # The optimal x_i is 1 if reduced cost > 0, else 0.
    # The dual function value is sum_{i: rc_i > 0} rc_i + sum lambda_j C_j.
    # We want to minimize this upper bound.
    # Subgradient of the dual function (upper bound) w.r.t lambda_j is:
    # C_j - sum_{i: rc_i > 0} w_ij
    # If the constraint is tight (used > C), the subgradient is negative, so we should increase lambda.
    # Wait, standard subgradient for minimization: lambda_new = lambda_old - step * subgrad
    # If subgrad = C - used. If used > C, subgrad < 0, so lambda increases. Correct.
    
    for it in range(n_sub):
        if _deadline_left() < MARGIN + 5.0:
            break
            
        # Reduced costs
        shadow_cost = w.T @ lam # shape (n,)
        rc = p - shadow_cost
        
        # Greedy pick based on rc (take all with rc > 0)
        # Note: In Lagrangian relaxation for knapsack, we sort by rc and take positive ones.
        # But here we are just using it to estimate the "value" and "usage" for the subgradient update.
        # The "primal" solution for the heuristic is just taking items with rc > 0 if they fit?
        # No, the Lagrangian relaxation assumes fractional or just "take if profitable".
        # For the subgradient update, we need the usage of the items that were "taken" in the relaxed problem.
        # The relaxed problem takes x_i = 1 if rc_i > 0.
        # Let's compute usage based on that.
        mask = rc > 0
        used = w[:, mask].sum(axis=1) if mask.any() else np.zeros(m)
        
        # Dual value (upper bound)
        # sum(rc_i for rc_i > 0) + sum(lambda_j * C_j)
        dual_val = rc[mask].sum() if mask.any() else 0.0 + lam @ C
        
        # We want to minimize dual_val.
        # However, the "primal" value (sum of p for taken items) is a lower bound.
        # We track the best primal value found by the greedy heuristic using these lambdas
        # to select the best lambda set at the end.
        
        # Primal heuristic: Sort by rc, take if fits.
        # This is the actual solution we will use to evaluate the heuristic quality.
        order = np.argsort(-rc, kind='stable')
        rem = C.copy()
        val = 0.0
        used_primal = np.zeros(m)
        for i in order:
            if rc[i] <= 0:
                continue
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
                used_primal += w[:, i]
                
        if val > best_val:
            best_val = val
            best_lam = lam.copy()
            
        # Update lambda using subgradient of the dual function (upper bound)
        # subgrad_j = C_j - used_j (from relaxed problem)
        slack = C - used
        step = step_init / (1.0 + 0.1 * it)
        # lambda = max(0, lambda - step * subgrad)
        # If slack > 0 (underused), subgrad > 0, lambda decreases.
        # If slack < 0 (overused), subgrad < 0, lambda increases.
        lam = np.maximum(0.0, lam - step * slack)

    # Use the best lambda found (based on primal value)
    lam = best_lam
    shadow_cost = w.T @ lam
    rc = p - shadow_cost
    
    # 2. Construct Heuristic Signals
    
    # Signal A: Reduced Cost (Lagrangian)
    # High rc means high value relative to resource cost.
    rc_pos = np.where(rc > 0, rc, 0.0)
    rc_max = rc_pos.max()
    if rc_max > 0:
        sig_rc = rc_pos / rc_max
    else:
        sig_rc = np.ones(n)
        
    # Signal B: Density (Prize / Total Weight)
    sum_w = w.sum(axis=0)
    safe_sum_w = np.maximum(sum_w, 1e-9)
    density = p / safe_sum_w
    d_max = density.max()
    if d_max > 0:
        sig_den = density / d_max
    else:
        sig_den = np.ones(n)
        
    # Signal C: Bottleneck Density (Prize / Max Weight in any constraint)
    # This helps with items that are light in most constraints but heavy in one.
    max_w = w.max(axis=0)
    safe_max_w = np.maximum(max_w, 1e-9)
    bot_density = p / safe_max_w
    b_max = bot_density.max()
    if b_max > 0:
        sig_bot = bot_density / b_max
    else:
        sig_bot = np.ones(n)

    # Signal D: Feasibility Slack (How much capacity is left if we pick this item?)
    # Not a static property, but we can approximate "ease of fitting".
    # An item is "easier" if its max weight is small relative to capacity.
    # Let's use 1 / (max_w / cap + 1) or similar.
    # Or simply: items with smaller max_w are more likely to fit later.
    # Let's normalize max_w.
    mw_max = max_w.max()
    if mw_max > 0:
        sig_fit = 1.0 - (max_w / mw_max) # Higher is better (smaller weight)
        sig_fit = np.maximum(sig_fit, 0.0)
    else:
        sig_fit = np.ones(n)

    # 3. Combine Signals
    # We want a heuristic that balances high value (rc, density) with good fit (bottleneck, slack).
    # The ACO pheromone will also guide.
    # Let's try a few combinations and pick the one that yields the best greedy solution.
    
    candidates = [
        sig_rc,
        sig_den,
        sig_bot,
        sig_rc * sig_den,
        sig_rc * sig_bot,
        (sig_rc + sig_den) / 2.0,
        (sig_rc + sig_bot) / 2.0,
        sig_rc ** 1.5 * sig_den ** 0.5,
        sig_rc ** 1.2 * sig_bot ** 0.8,
        sig_rc * sig_den * sig_fit,
        sig_rc ** 1.5 * sig_bot,
    ]
    
    best_score = -1.0
    best_heur = candidates[0]
    
    # Evaluate each candidate with a greedy solver
    for h in candidates:
        if _deadline_left() < MARGIN + 2.0:
            break
        order = np.argsort(-h, kind='stable')
        rem = C.copy()
        val = 0.0
        for i in order:
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
        if val > best_score:
            best_score = val
            best_heur = h
            
    # 4. Final Polish: Local Search on the best greedy solution?
    # The heuristic is static. The ACO will do the search.
    # However, we can refine the heuristic values to be more "peaked" for the best items.
    # The current best_heur is in [0, 1].
    # Let's ensure it's strictly positive.
    
    final_heur = np.asarray(best_heur, dtype=float)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    final_heur = np.maximum(final_heur, 1e-9)
    
    # Optional: Scale up to avoid underflow if pheromone is also small?
    # Pheromone starts at 1. Heuristic in [0,1] is fine.
    # But if we want to emphasize differences, we can exponentiate.
    # Let's try a mild exponentiation to sharpen the preference.
    # If h is in [0,1], h^2 makes low values lower.
    # Let's check if the greedy score improves with sharpening.
    
    # Actually, the ACO probability is proportional to (tau * eta)^alpha.
    # If eta is small, it might not matter much if tau is dominant initially.
    # But early on, tau is uniform (1.0). So eta dominates.
    # A sharper eta might help find the good items faster.
    
    # Let's test a few power transformations of the best_heur
    best_sharp_score = best_score
    best_sharp_heur = final_heur
    
    for power in [1.5, 2.0, 2.5, 3.0]:
        if _deadline_left() < MARGIN + 1.0:
            break
        h_sharp = np.power(final_heur, power)
        h_sharp = np.maximum(h_sharp, 1e-9)
        
        order = np.argsort(-h_sharp, kind='stable')
        rem = C.copy()
        val = 0.0
        for i in order:
            if np.all(w[:, i] <= rem):
                rem -= w[:, i]
                val += p[i]
        
        if val > best_sharp_score:
            best_sharp_score = val
            best_sharp_heur = h_sharp

    return best_sharp_heur
