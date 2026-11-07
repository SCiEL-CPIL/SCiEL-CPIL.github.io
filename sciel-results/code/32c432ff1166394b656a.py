"""Analytical optimal market making: closed-form depth from Bellman FOC."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def quote(t, inventory, mid, params):
    """
    Analytical optimal quote depths from the Bellman first-order condition.
    
    The optimal depth balances expected fill profit against inventory risk.
    FOC gives: d = (1/k) * ln(A*dt / (lambda + eta))
    
    where:
      - lambda(q,t) = gamma * sigma^2 * q  (marginal inventory risk)
      - eta(q,t) = gamma * sigma^2 * (Q - q)  (boundary/inventory loss near Q)
      - Time decay: as t -> T, remaining value drops, so we widen quotes
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = len(inventory)
    dt = T / steps
    
    # Time-to-horizon factor: as we approach T, remaining steps decrease
    # The value of inventory at time t is proportional to remaining time
    remaining_frac = max(0.0, 1.0 - t / T)
    
    # Inventory as float array
    q = inventory.astype(np.float64)
    
    # Marginal inventory risk: gamma * sigma^2 * q
    # This is the cost of holding one more unit of inventory
    lambda_q = gamma * sigma * sigma * q
    
    # Boundary correction: when q is near Q, a bid fill would push us to Q
    # where we can't fill anymore, losing future profit opportunities
    # eta represents the "shadow value" of being near the boundary
    # Near q=Q: losing inventory value (can't sell more) -> eta large for bid
    # Near q=-Q: losing inventory value (can't buy more) -> eta large for ask
    # 
    # For bid side: risk is being at high q (near Q)
    # For ask side: risk is being at low q (near -Q)
    #
    # Boundary effect: (Q - q) for bid, (q + Q) for ask
    eta_bid = gamma * sigma * sigma * (Q - q)
    eta_ask = gamma * sigma * sigma * (q + Q)
    
    # Time scaling: the value of future fills decreases as we approach T
    # Scale the "effective" risk by remaining fraction
    # Early in horizon: more time to rebalance, so risk matters more
    # Late in horizon: less time, so we should be more aggressive (wider)
    time_scale = 1.0 + (1.0 - remaining_frac) * 0.5
    
    # Effective denominators for the log
    denom_bid = lambda_q + eta_bid + time_scale * gamma * sigma * sigma * 0.1
    denom_ask = lambda_q + eta_ask + time_scale * gamma * sigma * sigma * 0.1
    
    # Ensure positive denominators
    denom_bid = np.maximum(denom_bid, 1e-10)
    denom_ask = np.maximum(denom_ask, 1e-10)
    
    # Optimal depth: d = (1/k) * ln(A * dt / denom)
    # A * dt is the expected number of fills per unit depth per step
    base = A * dt
    
    # Clip to valid range: depth must be > -3/k and > 0 for meaningful quotes
    # d > 0 means A*dt > denom, i.e., fills are profitable enough
    # If denom > A*dt, the log is negative -> depth negative -> no quote (inf)
    
    log_bid = np.log(base / denom_bid)
    log_ask = np.log(base / denom_ask)
    
    depth_bid = log_bid / k
    depth_ask = log_ask / k
    
    # Validity: depth must be >= -3/k (task constraint) and we only quote if depth > 0
    # Actually, re-reading: "distances below and above mid" - these should be positive
    # np.inf means no quote. Negative depths are invalid.
    # The constraint is values below -3/k are invalid, so depth >= -3/k
    # But for a market maker, depth should be positive (we quote at mid - depth for bid)
    
    # If log is negative, depth is negative -> set to inf (no quote)
    # Actually, let's allow small negative depths down to -3/k as the task allows
    # But practically, we want positive depths. Let's set inf where depth <= 0.
    
    # Minimum valid depth is -3/k, but we prefer positive
    min_depth = -3.0 / k
    
    # Where depth < min_depth or depth <= 0 (no profitable quote), use inf
    bid_depths = np.where(depth_bid > 0, depth_bid, np.inf)
    ask_depths = np.where(depth_ask > 0, depth_ask, np.inf)
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, np.inf)
    
    # Final safety: ensure all values >= -3/k
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    # But inf should stay inf
    bid_depths = np.where(np.isinf(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isinf(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
