"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Inventory-aware quoting: depth adjusts to penalize inventory buildup."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    
    q = inventory.astype(np.float64)
    
    # Marginal penalty coefficients
    # Bid fill: q -> q+1, penalty increase = gamma*sigma^2*(q+0.5)
    # Ask fill: q -> q-1, penalty increase = -gamma*sigma^2*(q-0.5)
    bid_cost = gamma * sigma**2 * (q + 0.5)
    ask_cost = -gamma * sigma**2 * (q - 0.5)
    
    # Optimal depth: d * A * exp(-k*d) = cost, if cost in (0, A)
    # d = (1/k) * ln(A / cost)
    # If cost <= 0: quote at d -> 0 (use small epsilon)
    # If cost >= A: no quote (inf)
    
    eps = 1e-12
    
    bid_depths = np.empty(len(inventory), dtype=np.float64)
    ask_depths = np.empty(len(inventory), dtype=np.float64)
    
    # Bid side
    # No bid fill at q = Q
    bid_mask_no_quote = (q >= Q) | (bid_cost >= A)
    bid_mask_quote = ~bid_mask_no_quote
    
    bid_depths[bid_mask_no_quote] = np.inf
    
    if np.any(bid_mask_quote):
        costs = np.maximum(bid_cost[bid_mask_quote], eps)
        # d = (1/k) * ln(A / cost), clamp to reasonable range
        d = (1.0 / k) * np.log(A / costs)
        # Clamp: min depth ~0, max depth reasonable
        d = np.clip(d, 0.0, 10.0 / k)
        bid_depths[bid_mask_quote] = d
    
    # Ask side
    # No ask fill at q = -Q
    ask_mask_no_quote = (q <= -Q) | (ask_cost >= A)
    ask_mask_quote = ~ask_mask_no_quote
    
    ask_depths[ask_mask_no_quote] = np.inf
    
    if np.any(ask_mask_quote):
        costs = np.maximum(ask_cost[ask_mask_quote], eps)
        d = (1.0 / k) * np.log(A / costs)
        d = np.clip(d, 0.0, 10.0 / k)
        ask_depths[ask_mask_quote] = d
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
