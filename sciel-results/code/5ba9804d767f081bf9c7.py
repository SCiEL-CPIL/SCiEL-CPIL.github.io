"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290  # Safety margin before 330s kill

def quote(t, inventory, mid, params):
    """
    Heuristic market making strategy based on inventory risk management.
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    n_paths = len(inventory)
    
    # Base spread determined by volatility and risk aversion
    # Higher gamma or sigma -> wider spread
    # Higher A -> can afford tighter spread
    base_spread = 0.5 / k * (1.0 + gamma * sigma**2 * dt)
    
    # Inventory adjustment: shift quote to reduce inventory
    # When long (q > 0), lower bid (wider bid depth) and raise ask (tighter ask depth)
    # When short (q < 0), raise bid (tighter bid depth) and lower ask (wider ask depth)
    # Adjustment is proportional to inventory, scaled by risk aversion and time remaining
    
    time_remaining = T - t
    time_factor = time_remaining / T if T > 0 else 1.0
    
    # Inventory adjustment coefficient
    # Larger gamma -> stronger response to inventory
    # Larger sigma -> more volatility, need more caution
    adj_coeff = gamma * sigma**2 * time_factor / (Q + 1.0)
    
    # Compute adjustments for each path
    q = inventory.astype(np.float64)
    
    # Shift: positive q (long) -> increase bid depth, decrease ask depth
    # Negative q (short) -> decrease bid depth, increase ask depth
    shift = adj_coeff * q
    
    # Base depths
    bid_base = base_spread
    ask_base = base_spread
    
    # Apply inventory adjustment
    bid_depths = bid_base + shift
    ask_depths = ask_base - shift
    
    # Ensure depths are positive and valid
    min_depth = 0.001
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Boundary conditions: no bid at q = Q, no ask at q = -Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure no NaN or negative values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
