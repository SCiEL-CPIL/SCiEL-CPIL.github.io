"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed values
_cache = {}

def _get_cache_key(params, t, inventory):
    """Create a cache key from parameters and time."""
    return (params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q'], t)

def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy.
    
    Adjusts bid/ask depths based on current inventory to manage risk.
    When long (inventory > 0), quote tighter bids and wider asks.
    When short (inventory < 0), quote wider bids and tighter asks.
    """
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Time remaining
    t_rem = T - t
    
    # Base depth: risk-neutral quote depth
    base_depth = 1.0 / k
    
    # Inventory risk penalty coefficient
    # The objective has -gamma/2 * sigma^2 * q^2 * dt per step
    # This creates a quadratic cost for holding inventory
    
    # Compute a "skew" factor based on inventory
    # Higher inventory -> more aggressive to reduce position
    # Use a soft scaling to avoid extreme quotes
    
    # The optimal skew should balance:
    # 1. Earnings from fills (depth)
    # 2. Risk cost of changing inventory
    
    # For small inventory, use near-symmetric quotes
    # For large inventory, skew heavily toward reducing position
    
    # Normalize inventory to [-1, 1]
    inv_norm = inventory.astype(np.float64) / Q
    
    # Compute skew: positive when we want to reduce inventory
    # When long (inv > 0), we want tighter bids (lower bid_depth) and wider asks (higher ask_depth)
    # When short (inv < 0), we want wider bids and tighter asks
    
    # Skew magnitude: scale with inventory and risk parameters
    # Higher gamma or sigma -> more risk aversion -> larger skew for same inventory
    skew_scale = gamma * sigma * sigma * t_rem * 0.1
    
    # Compute bid and ask depth adjustments
    # bid_depth should decrease when long (to encourage bid fills)
    # ask_depth should increase when long (to discourage ask fills)
    # And vice versa when short
    
    # Use a smooth function: depth_adjust = base * (1 + skew * inv_norm)
    # But clip to ensure positive depths and reasonable bounds
    
    bid_adj = base_depth * (1.0 + skew_scale * inv_norm)
    ask_adj = base_depth * (1.0 - skew_scale * inv_norm)
    
    # Clip to valid range: must be > -3/k and finite
    min_depth = -3.0 / k + 0.01  # Small margin above -3/k
    max_depth = 10.0 / k  # Reasonable upper bound
    
    bid_depths = np.clip(bid_adj, min_depth, max_depth)
    ask_depths = np.clip(ask_adj, min_depth, max_depth)
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    # At inventory == Q, we cannot take more bid fills, so set bid to inf
    # At inventory == -Q, we cannot take more ask fills, so set ask to inf
    
    # Create masks for boundary conditions
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    # Set bid to inf where we're at max inventory (can't take more bids)
    bid_depths[at_max_inventory] = np.inf
    
    # Set ask to inf where we're at min inventory (can't take more asks)
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
