"""Optimal market making with inventory risk using Avellaneda-Stoikov framework."""
# EVOLVE-BLOCK-START
import numpy as np


def quote(t, inventory, mid, params):
    """
    State-dependent optimal quoting using second-order Taylor expansion
    of the value function around current inventory.
    
    The optimal depth balances the spread earned against the marginal
    inventory risk cost. Using the AS framework:
    - Reservation price shift: q * gamma * sigma^2 * (T - t)
    - Optimal bid depth: max(0, shift - base_depth) clipped to valid range
    - Optimal ask depth: max(0, shift + base_depth) clipped to valid range
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    n = len(inventory)
    q = inventory.astype(np.float64)
    remaining = T - t
    
    # Risk-neutral base depth (1/k is where fill rate = A*dt per unit depth)
    base_depth = 1.0 / k
    
    # Inventory risk premium: marginal cost of holding one more unit
    # Second-order Taylor: V(q) ≈ V(0) + V'(0)*q + 0.5*V''(0)*q^2
    # V''(0) ≈ gamma * sigma^2 (from the quadratic risk term)
    # Reservation price shift = V'(q) ≈ gamma * sigma^2 * q * remaining
    shift = gamma * sigma * sigma * q * remaining
    
    # Optimal bid depth: we want to buy when q is negative
    # d_bid = base_depth - shift (buy more when shift is negative)
    # Optimal ask depth: we want to sell when q is positive  
    # d_ask = base_depth + shift (sell more when shift is positive)
    
    bid_depth = base_depth - shift
    ask_depth = base_depth + shift
    
    # Clip to valid range: must be >= -3/k (per contract) and >= 0 for practical quoting
    # Values below -3/k are invalid, so we clip to just above that
    min_depth = -3.0 / k + 1e-9  # just above the invalid threshold
    
    # But for practical market making, we want positive depths
    # If depth is too small, we can quote at minimum or not quote
    # The contract says np.inf means no quote
    
    # Ensure bid and ask depths are valid
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Handle boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, we should not quote bid (set to inf)
    # At q=-Q, we should not quote ask (set to inf)
    bid_depth[q >= Q] = np.inf
    ask_depth[q <= -Q] = np.inf
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
