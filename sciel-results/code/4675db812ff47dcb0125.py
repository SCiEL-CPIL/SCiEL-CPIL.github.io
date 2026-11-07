"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Shadow-price marginal utility blended with geometric slack and
    competitive exclusion, sharpened via soft-min to emphasize worst-constraint
    efficiency.
    
    The shadow price of each constraint is estimated as the inverse of its total
    weight sum (scarce constraints get higher shadow prices). Items are scored by
    their prize minus the weighted sum of their resource costs (penalizing items
    that consume scarce resources disproportionately). This is combined with:
      - Geometric slack: rewards items that consume less capacity
      - Competitive exclusion: rewards items that dominate their constraint's prize mass
      - Soft-min sharpening: emphasizes items efficient across ALL constraints
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        n_constraints, n_items = 5, 100
        capacity = 25.0
        eps = 1e-9
        
        # --- Component 1: Shadow-price marginal utility ---
        # Shadow price for each constraint: inverse of total weight in that constraint
        # Scarce constraints (high total weight relative to capacity) get higher prices
        total_weight_per_constraint = np.sum(weights, axis=1)  # (5,)
        # Shadow price: higher when constraint is more loaded
        shadow_prices = 1.0 / (total_weight_per_constraint + eps)  # (5,)
        # Normalize shadow prices so they're comparable
        sp_sum = np.sum(shadow_prices)
        if sp_sum > 0:
            shadow_prices = shadow_prices / sp_sum * n_constraints
        
        # Marginal utility: prize - sum of (shadow_price[c] * weight[c,i])
        # This directly penalizes items that consume scarce resources
        resource_cost = np.sum(shadow_prices[:, np.newaxis] * weights, axis=0)  # (100,)
        total_prize = np.sum(prizes, axis=0)  # (100,)
        marginal_utility = total_prize - resource_cost  # (100,)
        
        # --- Component 2: Harmonic mean of per-constraint ratios ---
        safe_w = weights + eps
        ratios = prizes / safe_w  # (5, 100)
        inv_ratios = 1.0 / (ratios + eps)  # (5, 100)
        h_harmonic = n_constraints / (np.sum(inv_ratios, axis=0) + eps)  # (100,)
        
        # --- Component 3: Geometric slack ---
        slack = 1.0 - weights / capacity
        slack = np.clip(slack, eps, 1.0)
        log_slack = np.log(slack)
        geo_slack = np.exp(np.mean(log_slack, axis=0))  # (100,)
        
        # --- Component 4: Competitive exclusion ---
        total_prize_per_constraint = np.sum(prizes, axis=1)  # (5,)
        prize_share = prizes / (total_prize_per_constraint[:, np.newaxis] + eps)  # (5, 100)
        weighted_share = ratios * prize_share  # (5, 100)
        log_ws = np.log(weighted_share + eps)
        h_competitive = np.exp(np.mean(log_ws, axis=0))  # (100,)
        
        # --- Soft-min across ratios to emphasize worst-constraint efficiency ---
        # Use a soft-min with temperature T to approximate min while remaining differentiable
        T = 0.05
        log_ratios = np.log(ratios + eps)  # (5, 100)
        # Soft-min: -T * log(sum(exp(-log_r/T)))
        neg_log_ratios = -log_ratios / T
        max_neg = np.max(neg_log_ratios, axis=0, keepdims=True)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_log_ratios - max_neg), axis=0, keepdims=True))
        soft_min = -T * log_sum_exp.squeeze(axis=0)  # (100,)
        soft_min = np.exp(soft_min)  # Convert back from log-space
        
        # --- Normalize all components to [0, 1] ---
        def normalize_positive(arr):
            arr = np.where(arr > 0, arr, eps)
            mn = np.min(arr)
            mx = np.max(arr)
            if mx > mn:
                return (arr - mn) / (mx - mn)
            return np.ones_like(arr)
        
        h_mu = normalize_positive(marginal_utility)
        h_harm = normalize_positive(h_harmonic)
        h_slack = normalize_positive(geo_slack)
        h_comp = normalize_positive(h_competitive)
        h_softmin = normalize_positive(soft_min)
        
        # --- Blend components with tuned weights ---
        # Shadow-price marginal utility: captures opportunity cost directly
        # Harmonic mean: smooth min-ratio surrogate
        # Geometric slack: rewards low consumption
        # Competitive exclusion: rewards dominance in constraint prize mass
        # Soft-min: sharp emphasis on worst-constraint efficiency
        alpha_mu = 0.30
        alpha_harm = 0.20
        alpha_slack = 0.15
        alpha_comp = 0.15
        alpha_softmin = 0.20
        
        h = (alpha_mu * h_mu +
             alpha_harm * h_harm +
             alpha_slack * h_slack +
             alpha_comp * h_comp +
             alpha_softmin * h_softmin)
        
        # --- Sharpening: power to emphasize high-efficiency items ---
        h = np.power(h + eps, 1.8)
        
        # --- Cleanup ---
        h = np.where(np.isfinite(h) & (h > 0), h, eps)
        h = h.reshape(n_items)
        
        # Time check
        if time.time() > deadline:
            return h
        
        return h
        
    except Exception:
        return np.full((100,), 1e-9)
# EVOLVE-BLOCK-END
