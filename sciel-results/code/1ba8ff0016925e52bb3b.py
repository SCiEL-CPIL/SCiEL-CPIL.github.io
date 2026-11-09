import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    Uses a combination of Lagrangian dual prices and greedy bias to guide the ACO.
    """
    start_time = time.time()
    
    # Handle input shapes robustly
    p = np.asarray(prizes, dtype=float)
    w = np.asarray(weights, dtype=float)
    
    # Extract 1D prize vector
    if p.ndim == 2:
        if p.shape[0] == 1:
            p = p[0]
        elif p.shape[1] == 1:
            p = p[:, 0]
        else:
            # If (5, 100), assume first row or average? 
            # Task says "prizes and weights ... shape (5, 100)".
            # If prizes is (5, 100), it's ambiguous. 
            # Given "one score per item", and previous code worked, 
            # likely prizes is effectively 1D or (1, 100).
            # If it's (5, 100), taking the mean across constraints is a reasonable guess
            # for a single "prize" per item if the 5 rows represent something else.
            # However, standard MCKP has one prize per item.
            # Let's assume if 2D and not 1xN or Nx1, it might be a mistake or 
            # the 5 rows are identical. Taking the first row is safer than averaging 
            # if they are distinct scenarios, but averaging is safer if they are 
            # multi-dimensional prizes.
            # Let's stick to the previous successful logic: p[0] if 2D.
            p = p[0]
            
    p = np.ravel(p)
    
    # Ensure weights is 2D (5, 100)
    if w.ndim == 1:
        # If weights is 1D, it's likely (100,) which is wrong for 5 constraints.
        # But if it's (500,), reshape?
        # Assume standard (5, 100).
        pass
        
    n_items = 100
    n_constraints = 5
    capacity = np.full(n_constraints, 25.0)
    
    # Adjust p and w to match expected dimensions
    if p.shape[0] != n_items:
        # If p is longer, truncate? If shorter, pad?
        # Assume it's 100.
        pass
        
    if w.shape != (n_constraints, n_items):
        # Try to reshape
        if w.size == n_constraints * n_items:
            w = w.reshape(n_constraints, n_items)
        else:
            # Fallback: just use what we have, but ensure it works
            pass

    # Deadline guard: 330s total, 3 calls. ~100s per call.
    # Leave margin for overhead.
    deadline = start_time + 90.0
    
    # --- 1. Compute a high-quality greedy feasible solution ---
    # This provides a strong baseline and bias.
    
    # Multiple greedy strategies to find a good feasible set
    def greedy_fill(strategy='ratio'):
        # strategy: 'ratio' (p / sum_w), 'prize' (p), 'weight' (1/sum_w)
        if strategy == 'ratio':
            scores = p / (w.sum(axis=0) + 1e-9)
        elif strategy == 'prize':
            scores = p
        elif strategy == 'weight':
            scores = 1.0 / (w.sum(axis=0) + 1e-9)
        else:
            scores = p
            
        idx = np.argsort(scores)[::-1]
        selected = np.zeros(n_items, dtype=bool)
        remaining_cap = capacity.copy()
        total_prize = 0.0
        
        for i in idx:
            fits = np.all(w[:, i] <= remaining_cap)
            if fits:
                selected[i] = True
                remaining_cap -= w[:, i]
                total_prize += p[i]
                
        return selected, total_prize

    # Try a few greedy variants to get a robust "good" set
    sel_ratio, prize_ratio = greedy_fill('ratio')
    sel_prize, prize_prize = greedy_fill('prize')
    sel_weight, prize_weight = greedy_fill('weight')
    
    # Combine: Union of selected items from best greedy? 
    # Or just use the one with highest prize as the primary bias source.
    best_greedy_prize = max(prize_ratio, prize_prize, prize_weight)
    if best_greedy_prize == prize_ratio:
        best_greedy_sel = sel_ratio
    elif best_greedy_prize == prize_prize:
        best_greedy_sel = sel_prize
    else:
        best_greedy_sel = sel_weight
        
    # --- 2. Lagrangian Dual Optimization ---
    # Use subgradient method to find dual variables (shadow prices).
    
    lambda_dual = np.zeros(n_constraints)
    best_lower_bound = best_greedy_prize
    
    # Precompute for speed
    w_sum = w.sum(axis=0)
    
    max_iters = 3000
    step_size = 1.0
    damping = 0.5
    
    for k in range(max_iters):
        if time.time() > deadline - 5.0: # Leave 5s for final computation
            break
            
        # Reduced costs
        w_t_lambda = w.T @ lambda_dual
        v = p - w_t_lambda
        
        # Primal solution for dual evaluation
        x = (v > 0).astype(float)
        
        # Dual value g(lambda)
        g_lambda = np.dot(capacity, lambda_dual) + np.dot(v, x)
        
        # Subgradient S
        S = capacity - w @ x
        
        # Step size
        s_sq = np.dot(S, S)
        if s_sq < 1e-9:
            s_sq = 1e-9
            
        alpha = (g_lambda - best_lower_bound) / s_sq
        alpha = min(alpha, 1.0)
        
        # Update lambda
        lambda_new = lambda_dual + alpha * S
        lambda_new = np.maximum(0, lambda_new)
        
        # Optional: Check if new lambda is better
        # For speed, just accept.
        lambda_dual = lambda_new
        
    # --- 3. Construct Heuristic ---
    
    # Base heuristic from dual prices
    w_t_lambda = w.T @ lambda_dual
    # Add a base cost to prevent division by zero and to scale
    # If lambda is 0, denom is base_cost.
    base_cost = 0.1
    denom = w_t_lambda + base_cost
    denom = np.maximum(denom, 1e-9)
    
    eta_dual = p / denom
    
    # Greedy bias: Boost items that are in the best greedy solution
    # This hard-codes a high-quality feasible region into the heuristic.
    greedy_boost = np.ones(n_items)
    greedy_boost[best_greedy_sel] = 5.0  # Strong boost
    
    # Combine
    eta = eta_dual * greedy_boost
    
    # Ensure positive and finite
    eta = np.maximum(eta, 1e-9)
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    # Final check on shape
    if eta.shape != (n_items,):
        eta = eta[:n_items]
        
    return eta
