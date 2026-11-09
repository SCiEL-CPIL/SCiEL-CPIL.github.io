"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """Inventory-aware market making: adjust depths based on inventory to manage risk."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    dt = params["T"] / params["steps"]
    Q = params["Q"]
    
    n = len(inventory)
    inv = np.asarray(inventory, dtype=np.float64)
    
    # Base depth: risk-neutral where exponential term is order 1
    base_depth = 1.0 / k
    
    # Inventory adjustment: shift bid up and ask down for positive inventory, vice versa
    # Adjustment magnitude scales with inventory risk cost
    adj_scale = np.e * gamma * sigma**2 * dt / (k * A)
    
    bid_depth = base_depth + adj_scale * inv
    ask_depth = base_depth - adj_scale * inv
    
    # Clip to reasonable bounds to avoid extreme quotes
    min_depth = 0.01
    max_depth = 10.0
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    
    # At inventory limits, don't quote the side that would increase inventory further
    # No bid fill at q = Q (can't buy more), no ask fill at q = -Q (can't sell more)
    bid_depth = np.where(inv >= Q, np.inf, bid_depth)
    ask_depth = np.where(inv <= -Q, np.inf, ask_depth)
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
