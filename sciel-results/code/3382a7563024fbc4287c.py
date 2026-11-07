"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Lagrangian dual-based scoring. We estimate shadow prices (dual
    variables) for each constraint via subgradient ascent on the Lagrangian
    dual of the MCKP, then score each item by its reduced cost:
        reduced_cost[i] = sum_c prize[c,i] - sum_c lambda[c] * weight[c,i]
    
    Items with high positive reduced cost are "profitable" after accounting for
    the opportunity cost of consuming scarce resources. This directly models the
    knapsack's economic structure rather than blending ad-hoc features.
    
    We also blend in a rank-based efficiency signal to ensure robustness.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        n_constraints, n_items = 5, 100
        capacity = 25.0
        eps = 1e-9
        
        # --- Step 1: Estimate shadow prices via subgradient method ---
        # Lagrangian dual: maximize g(lambda) = sum_i max(0, prize_i - lambda @ w_i) + lambda @ b
        # where b = capacity vector, prize_i = sum_c prize[c,i], w_i = weight column i
        # Subgradient of g w.r.t. lambda[c] = sum_{i in S(lambda)} w_i[c] - capacity
        # where S(lambda) = {i : prize_i - lambda @ w_i > 0}
        
        total_prize = np.sum(prizes, axis=0)  # (100,)
        # Initialize shadow prices: proportional to constraint tightness
        total_weight = np.sum(weights, axis=1)  # (5,)
        tightness = total_weight / capacity  # >1 means over capacity if all items taken
        
        # Initialize lambda based on tightness: tighter constraints get higher initial prices
        lambda_dual = np.maximum(tightness - 1.0, 0.0) * 0.5 + 0.1  # (5,)
        
        # Subgradient ascent iterations
        n_iter = 200
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute reduced costs for all items
            rc = total_prize - weights.T @ lambda_dual  # (100,)
            
            # Items with positive reduced cost are "selected" in the Lagrangian solution
            selected = rc > 0
            
            # Subgradient: sum of weights of selected items per constraint minus capacity
            if np.any(selected):
                sg = weights[:, selected].sum(axis=1) - capacity  # (5,)
            else:
                sg = -capacity  # (5,)
            
            # Polyak step size: (g_max - g(lambda)) / ||sg||^2
            # g(lambda) = sum of positive reduced costs + lambda @ capacity
            g_val = np.sum(np.maximum(rc, 0)) + lambda_dual @ capacity
            
            # Upper bound: sum of all positive prizes (relax all constraints)
            g_upper = np.sum(np.maximum(total_prize, 0))
            
            gap = g_upper - g_val
            sg_norm_sq = np.sum(sg ** 2)
            
            if sg_norm_sq < 1e-12:
                step = 0.0
            else:
                step = max(gap, 0.0) / sg_norm_sq
                # Dampen step to avoid oscillation
                step *= 0.5
            
            lambda_dual = np.maximum(lambda_dual + step * sg, 0.0)
        
        # --- Step 2: Compute final reduced costs ---
        rc_final = total_prize - weights.T @ lambda_dual  # (100,)
        
        # --- Step 3: Rank-based efficiency signal ---
        # Per-constraint efficiency: prize / weight, then geometric mean
        safe_w = weights + eps
        ratios = prizes / safe_w  # (5, 100)
        log_ratios = np.log(ratios + eps)
        geo_eff = np.exp(np.mean(log_ratios, axis=0))  # (100,)
        
        # --- Step 4: Combine reduced cost and geometric efficiency ---
        # Normalize both to [0, 1]
        def norm01(arr):
            arr = np.where(np.isfinite(arr), arr, 0.0)
            mn = np.min(arr)
            mx = np.max(arr)
            if mx > mn + 1e-12:
                return (arr - mn) / (mx - mn)
            return np.ones_like(arr) * 0.5
        
        h_rc = norm01(rc_final)
        h_geo = norm01(geo_eff)
        
        # Blend: reduced cost is primary, geometric efficiency is secondary
        # The reduced cost already captures the economic trade-off
        h = 0.7 * h_rc + 0.3 * h_geo
        
        # --- Step 5: Sharpen via power transformation ---
        # Emphasize high-scoring items more strongly
        h = np.power(h + eps, 1.5)
        
        # --- Cleanup ---
        h = np.where(np.isfinite(h) & (h > 0), h, eps)
        h = h.reshape(n_items)
        
        return h
        
    except Exception:
        return np.full((100,), 1e-9)
# EVOLVE-BLOCK-END
