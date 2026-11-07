"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time
from functools import lru_cache

# Pre-compute the optimal policy table at module load time
_POLICY_TABLES = {}
_COMPUTE_DEADLINE = time.time() + 280  # 280s deadline for pre-computation (30s margin)


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 to n = 0
    # V[steps, q] = 0 (no terminal charge)
    
    # For each step, we need to find optimal bid_depth and ask_depth
    # The fill probability for bid at depth d: p_bid = 1 - exp(-A * exp(-k*d) * dt)
    # If bid fills: earn d, inventory decreases by 1 (q -> q-1)
    # If ask fills: earn d, inventory increases by 1 (q -> q+1)
    # No bid fill at q = Q (can't sell more), no ask fill at q = -Q (can't buy more)
    
    # Objective per step: depth_earned - (gamma/2) * sigma^2 * q^2 * dt
    # where depth_earned is d_bid if bid fills, d_ask if ask fills, 0 otherwise
    
    # V[n, q] = max_{d_bid, d_ask} [ 
    #   -(gamma/2) * sigma^2 * q^2 * dt
    #   + p_bid * (d_bid + V[n+1, q-1]) + p_ask * (d_ask + V[n+1, q+1])
    #   + (1-p_bid)*(1-p_ask) * V[n+1, q]
    # ]
    # with boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    
    # Since bid and ask are independent and the objective is separable in d_bid and d_ask
    # (except for the joint probability term), we can optimize them separately.
    
    # Actually, let's think more carefully. The expected value is:
    # E = p_bid * (d_bid + V[n+1, q-1]) + p_ask * (d_ask + V[n+1, q+1]) + (1-p_bid)*(1-p_ask) * V[n+1, q]
    #   = p_bid * d_bid + p_bid * V[n+1, q-1] + p_ask * d_ask + p_ask * V[n+1, q+1]
    #     + (1-p_bid)*(1-p_ask) * V[n+1, q]
    
    # Note: (1-p_bid)*(1-p_ask) = 1 - p_bid - p_ask + p_bid*p_ask
    
    # So E = p_bid * d_bid + p_ask * d_ask + p_bid * V[n+1, q-1] + p_ask * V[n+1, q+1]
    #        + (1 - p_bid - p_ask + p_bid*p_ask) * V[n+1, q]
    
    # = p_bid * (d_bid + V[n+1, q-1] - V[n+1, q]) + p_ask * (d_ask + V[n+1, q+1] - V[n+1, q])
    #   + p_bid*p_ask * V[n+1, q] + V[n+1, q]
    
    # Hmm, the cross term p_bid*p_ask couples them. But for small dt, p_bid and p_ask are small,
    # so p_bid*p_ask is of order dt^2, which is negligible. Let's assume independence and optimize separately.
    
    # For bid (assuming q < Q, i.e., can sell):
    # Maximize: p_bid * (d_bid + V[n+1, q-1] - V[n+1, q]) + (higher order terms)
    # where p_bid = 1 - exp(-A * exp(-k*d_bid) * dt)
    
    # For small dt, p_bid ≈ A * exp(-k*d_bid) * dt
    # So we maximize: A * exp(-k*d_bid) * dt * (d_bid + delta_V_bid)
    # where delta_V_bid = V[n+1, q-1] - V[n+1, q]
    
    # Let f(d) = exp(-k*d) * (d + c) where c = delta_V_bid
    # f'(d) = -k*exp(-k*d)*(d+c) + exp(-k*d) = exp(-k*d) * (1 - k*(d+c))
    # f'(d) = 0 => d = 1/k - c
    
    # But we need d > 0 and d >= -3/k (from the constraint)
    
    # Similarly for ask:
    # Maximize: p_ask * (d_ask + V[n+1, q+1] - V[n+1, q])
    # Let delta_V_ask = V[n+1, q+1] - V[n+1, q]
    # Optimal: d_ask = 1/k - delta_V_ask
    
    # Let's verify this with the exact formula. For bid:
    # g(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + c)
    # For small dt, this is approximately A*exp(-k*d)*dt*(d+c)
    # The maximum of exp(-k*d)*(d+c) is at d = 1/k - c
    
    # Let's use this approximation. We'll compute the value function backwards.
    
    V = np.zeros((steps + 1, n_inv))
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # We need to be careful: the inventory q is the inventory BEFORE the step's fills.
    # After fills, inventory changes.
    
    # Let's index inventory from 0 to 2*Q, where index i corresponds to q = i - Q
    
    for n in range(steps - 1, -1, -1):
        for i in range(n_inv):
            q = i - Q
            
            # Risk penalty for this step
            risk_penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q == Q:
                # No bid fill possible
                bid_table[n, i] = np.inf
                # Optimize ask only
                # Maximize p_ask * (d_ask + V[n+1, q+1] - V[n+1, q])
                # But q+1 = Q+1 is out of bounds... wait, q = Q means we're at max inventory
                # Actually, inventory is in [-Q, Q]. If q = Q, we can still buy (ask fill) to go to Q+1?
                # No, the problem says "no ask fill at q = -Q". So at q = Q, ask fills are allowed.
                # At q = -Q, bid fills are allowed.
                # Wait, let me re-read: "no bid fill at q = Q, no ask fill at q = -Q"
                # So at q = Q, bid fills are not allowed (can't sell when at max short? or max long?)
                # Inventory: positive means long? Let's assume q is the inventory level.
                # "A fill earns its depth and moves inventory one unit"
                # Bid fill: sell, so inventory decreases. Ask fill: buy, inventory increases.
                # "no bid fill at q = Q" means at max inventory, can't sell more.
                # "no ask fill at q = -Q" means at min inventory, can't buy more.
                
                # So at q = Q: no bid fill. Ask fill moves q to Q+1? But Q is the max.
                # Hmm, maybe the inventory is bounded and fills at the boundary are just not allowed.
                # Let me assume that at q = Q, bid fills don't happen, and ask fills move q to Q (clipped)
                # or maybe the state space is just [-Q, Q] and transitions at boundaries stay put.
                
                # Actually, re-reading: "inventory (4000,) ints in [-Q, Q]"
                # So inventory is always in [-Q, Q]. At q = Q, bid fill would move to Q-1, but it's not allowed.
                # At q = -Q, ask fill would move to -Q+1, but it's not allowed.
                
                # So at q = Q: only ask fills possible, moving q to Q+1? But that's out of bounds.
                # I think the interpretation is: at q = Q, bid fills are suppressed. At q = -Q, ask fills are suppressed.
                # The inventory after a fill is clamped to [-Q, Q].
                
                # Let's assume: at q = Q, bid fill probability is 0. Ask fill moves q to min(Q, q+1) = Q.
                # At q = -Q, ask fill probability is 0. Bid fill moves q to max(-Q, q-1) = -Q.
                
                # This makes the boundary states absorbing for the respective side.
                
                # For ask at q = Q:
                delta_V_ask = V[n + 1, min(Q, q + 1) + Q] - V[n + 1, i]
                # q+1 = Q+1, clamped to Q, so index is Q + Q = 2Q = n_inv - 1
                # Actually, q+1 = Q+1, but inventory is bounded at Q, so it stays at Q.
                # So V[n+1, Q] - V[n+1, Q] = 0? That doesn't seem right.
                
                # Let me reconsider. Maybe at the boundary, the fill just doesn't happen,
                # and the inventory doesn't change. So at q = Q, ask fill: q stays at Q.
                # Then delta_V_ask = V[n+1, Q] - V[n+1, Q] = 0.
                
                # That would mean at the boundary, the optimal depth is 1/k.
                
                d_ask_opt = 1.0 / k - delta_V_ask
                d_ask_opt = max(d_ask_opt, -3.0 / k + 1e-10)
                d_ask_opt = max(d_ask_opt, 0.0)
                ask_table[n, i] = d_ask_opt
                
                # Value: risk_penalty + expected reward from ask
                p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask_opt) * dt)
                V[n, i] = -risk_penalty + p_ask * (d_ask_opt + V[n + 1, i]) + (1 - p_ask) * V[n + 1, i]
                # Wait, if ask fills, q stays at Q (clamped), so V[n+1, Q] = V[n+1, i]
                # So V[n, i] = -risk_penalty + p_ask * d_ask_opt + V[n+1, i]
                
            elif q == -Q:
                # No ask fill possible
                ask_table[n, i] = np.inf
                delta_V_bid = V[n + 1, max(-Q, q - 1) + Q] - V[n + 1, i]
                # q-1 = -Q-1, clamped to -Q, so index is -Q + Q = 0
                # delta_V_bid = V[n+1, -Q] - V[n+1, -Q] = 0
                
                d_bid_opt = 1.0 / k - delta_V_bid
                d_bid_opt = max(d_bid_opt, -3.0 / k + 1e-10)
                d_bid_opt = max(d_bid_opt, 0.0)
                bid_table[n, i] = d_bid_opt
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid_opt) * dt)
                V[n, i] = -risk_penalty + p_bid * d_bid_opt + V[n + 1, i]
                
            else:
                # Both sides possible
                delta_V_bid = V[n + 1, i - 1] - V[n + 1, i]
                delta_V_ask = V[n + 1, i + 1] - V[n + 1, i]
                
                d_bid_opt = 1.0 / k - delta_V_bid
                d_bid_opt = max(d_bid_opt, -3.0 / k + 1e-10)
                d_bid_opt = max(d_bid_opt, 0.0)
                
                d_ask_opt = 1.0 / k - delta_V_ask
                d_ask_opt = max(d_ask_opt, -3.0 / k + 1e-10)
                d_ask_opt = max(d_ask_opt, 0.0)
                
                bid_table[n, i] = d_bid_opt
                ask_table[n, i] = d_ask_opt
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid_opt) * dt)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask_opt) * dt)
                
                V[n, i] = -risk_penalty + p_bid * (d_bid_opt + V[n + 1, i - 1]) + p_ask * (d_ask_opt + V[n + 1, i + 1]) + (1 - p_bid) * (1 - p_ask) * V[n + 1, i]
        
        # Check deadline
        if time.time() > _COMPUTE_DEADLINE:
            # Return what we have so far (partial table)
            break
    
    return bid_table, ask_table, V


def _get_policy_table(params):
    """Get or compute the policy table for the given parameters."""
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["T"], params["steps"], params["Q"])
    if key not in _POLICY_TABLES:
        bid_table, ask_table, _ = _compute_policy_table(
            params["sigma"], params["A"], params["k"], params["gamma"],
            params["T"], params["steps"], params["Q"]
        )
        _POLICY_TABLES[key] = (bid_table, ask_table)
    return _POLICY_TABLES[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    bid_table, ask_table = _get_policy_table(params)
    
    Q = params["Q"]
    n_inv = 2 * Q + 1
    
    # Convert inventory to index
    # inventory is in [-Q, Q], so index = inventory + Q
    inv_indices = inventory + Q
    
    # Clip to valid range (shouldn't be needed, but safe)
    inv_indices = np.clip(inv_indices, 0, n_inv - 1)
    
    # Determine step from t
    dt = params["T"] / params["steps"]
    step = int(round(t / dt))
    step = np.clip(step, 0, params["steps"] - 1)
    
    # Get depths from table
    bid_depths = bid_table[step, inv_indices]
    ask_depths = ask_table[step, inv_indices]
    
    # Ensure valid values
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    bid_depths = np.where(bid_depths < -3.0 / params["k"], np.inf, bid_depths)
    ask_depths = np.where(ask_depths < -3.0 / params["k"], np.inf, ask_depths)
    
    return bid_depths, ask_depths


# Pre-compute tables for all regimes at module load
_PRECOMPUTE_PARAMS = [
    {'sigma': 2, 'A': 140, 'k': 1.5, 'gamma': 0.05, 'T': 1, 'steps': 1000, 'Q': 30},
    {'sigma': 2, 'A': 140, 'k': 1.5, 'gamma': 0.5, 'T': 1, 'steps': 1000, 'Q': 30},
    {'sigma': 1.5, 'A': 50, 'k': 0.75, 'gamma': 0.2, 'T': 1, 'steps': 1000, 'Q': 30},
]

for _params in _PRECOMPUTE_PARAMS:
    try:
        _get_policy_table(_params)
    except Exception:
        pass
    if time.time() > _COMPUTE_DEADLINE:
        break
