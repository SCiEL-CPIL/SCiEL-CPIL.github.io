"""
Rewritten program using an Analytic Hedging Heuristic.
The policy skews the bid and ask depths based on current inventory to manage risk,
approximating the first-order optimal control solution.
"""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on inventory risk.
    
    Args:
        t: Current time step (float).
        inventory: Array of current inventories (int array).
        mid: Array of mid prices (float array).
        params: Dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'.
        
    Returns:
        Tuple of (bid_depths, ask_depths) as float arrays.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    inv = inventory.astype(np.float64)
    n = len(inv)
    
    # Base depth: risk-neutral depth where fill probability is balanced
    base_depth = 1.0 / k
    
    # Calculate inventory skew
    # The term (gamma * sigma^2) / (A * k) represents the strength of the risk aversion
    # relative to the fill intensity and depth sensitivity.
    # We scale by inventory.
    skew_factor = (gamma * sigma**2) / (A * k)
    skew = skew_factor * inv
    
    # Compute raw depths
    bid_depths = base_depth * (1.0 + skew)
    ask_depths = base_depth * (1.0 - skew)
    
    # Ensure depths are positive (minimum depth epsilon)
    min_depth = 1e-6
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Handle inventory bounds:
    # No bid fill at q = Q (we are max long, don't buy more)
    # No ask fill at q = -Q (we are max short, don't sell more)
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
