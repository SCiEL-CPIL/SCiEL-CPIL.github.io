"""Market-making with inventory risk: fast heuristic policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard (300 s hard limit for the whole run).
_T0 = time.time()
_DEADLINE = 290.0  # seconds of wall clock budget for the whole run.

# Cache policy params per regime
_POLICY_CACHE = {}


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _compute_policy_params(sigma, A, k, gamma, T, steps, Q):
    """Compute heuristic policy parameters: base depth and inventory tilt.
    
    We use a simple model:
    - Base symmetric depth d0 = 1/k
    - Tilt alpha proportional to gamma * sigma^2 / A, clipped
    
    The idea: when inventory is high (positive), we want to be more aggressive
    on the bid side (lower bid depth to attract buys that reduce inventory) 
    and less aggressive on ask (higher ask depth).
    
    So: bid_depth = d0 - alpha * q
         ask_depth = d0 + alpha * q
    
    where alpha > 0 encourages inventory reduction.
    
    We choose alpha to balance the risk penalty against the expected flow.
    """
    dt = T / steps
    d0 = 1.0 / k  # symmetric base depth
    
    # The risk penalty per step is gamma * sigma^2 * q^2 * dt / 2
    # The benefit of tilting: if q > 0, lower bid depth increases bid fill prob
    # which reduces inventory, earning depth and reducing future risk.
    
    # A reasonable heuristic: alpha = gamma * sigma^2 * dt / (A * k)
    # This scales with risk and volatility, and inversely with intensity.
    
    # Let's also consider the time horizon: near the end, we care less about inventory
    # But since we have no terminal penalty, we can still adjust throughout.
    
    alpha = gamma * sigma**2 * dt / (A * k)
    
    # Clip alpha so that at q = Q, the depth doesn't go negative or exceed 3/k
    # bid_depth at q=Q: d0 - alpha*Q >= 0 => alpha <= d0/Q
    # ask_depth at q=Q: d0 + alpha*Q <= 3/k => alpha <= (3/k - d0)/Q = 2/(k*Q)
    max_alpha = min(d0 / Q, 2.0 / (k * Q))
    alpha = min(alpha, max_alpha)
    
    # Also ensure alpha is non-negative
    alpha = max(alpha, 0.0)
    
    # Additional consideration: if A is very large, fills are very likely even at 
    # large depths, so we might want to be more aggressive (larger alpha).
    # If A is small, fills are rare, so we should be more conservative.
    # The formula above already captures this via 1/A.
    
    return d0, alpha


def _get_policy(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _POLICY_CACHE:
        if _remaining_budget() < 1.0:
            # Not enough time, use default
            d0 = 1.0 / k
            _POLICY_CACHE[key] = (d0, 0.0)
        else:
            _POLICY_CACHE[key] = _compute_policy_params(sigma, A, k, gamma, T, steps, Q)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    # Check deadline
    if _remaining_budget() < 0.5:
        # Emergency fallback: return safe default quotes
        max_depth = 3.0 / k
        bid = np.full(len(inventory), 0.5 * max_depth)
        ask = np.full(len(inventory), 0.5 * max_depth)
        return bid, ask
    
    d0, alpha = _get_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Compute depths: bid_depth = d0 - alpha * q, ask_depth = d0 + alpha * q
    max_depth = 3.0 / k
    q = inventory.astype(np.float64)
    
    bid_depths = d0 - alpha * q
    ask_depths = d0 + alpha * q
    
    # Clip to valid range [0, 3/k]
    bid_depths = np.clip(bid_depths, 0.0, max_depth)
    ask_depths = np.clip(ask_depths, 0.0, max_depth)
    
    # Ensure no NaN or inf
    bid_depths = np.nan_to_num(bid_depths, nan=0.5 * max_depth, posinf=max_depth, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.5 * max_depth, posinf=max_depth, neginf=0.0)
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
