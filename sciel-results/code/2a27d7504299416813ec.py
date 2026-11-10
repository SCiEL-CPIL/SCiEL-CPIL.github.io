"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Pre-computed optimal policy cache
_POLICY_CACHE = {}
_START_TIME = time.time()
_DEADLINE = 330.0  # 300s limit + safety margin

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths for each (time, inventory) state using DP.
    
    The value function V(n, q) = max expected reward from step n to end given inventory q.
    Reward at step n: depth_earned - (gamma/2) * sigma^2 * q^2 * dt
    where depth_earned = bid_depth * P(bid_fill) + ask_depth * P(ask_fill)
    and inventory changes by -1 if bid fills, +1 if ask fills.
    
    We solve backward from n = steps-1 to n = 0.
    """
    dt = T / steps
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Depth grid for optimization
    # Optimal depth is around 1/k, so we search in a reasonable range
    depth_min = 1e-4
    depth_max = max(5.0 / k, 0.5)
    n_depths = 50
    depth_grid = np.linspace(depth_min, depth_max, n_depths)
    
    # V[q] = value function for current time step, indexed by q + Q
    # Start with V = 0 at terminal
    V_next = np.zeros(2 * Q + 1)
    
    # Store optimal depths: opt_bid[n, q+Q], opt_ask[n, q+Q]
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    # Precompute probabilities for each depth
    # P(bid fill at depth d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 5.0:
            break
        
        V_curr = np.zeros(2 * Q + 1)
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            
            # Penalty for inventory risk
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # If at inventory limit, can't quote on that side
            if q == Q:
                # Can't have bid fill (no bid quote)
                best_val = -penalty
                best_bid = 0.0
                best_ask = 0.0
                for i, d in enumerate(depth_grid):
                    p_ask = fill_probs[i]
                    # Ask fill: inventory increases by 1, but we're at Q so this shouldn't happen
                    # Actually, no ask fill at q = -Q, so at q=Q we can still have ask fills? 
                    # Wait: "no bid fill at q = Q, no ask fill at q = -Q"
                    # So at q=Q, bid fills are prevented (we shouldn't quote bid or it won't fill)
                    # Ask fills can still happen, moving to Q+1? But Q is the max.
                    # Let me re-read: inventory in [-Q, Q]. So if q=Q and ask fills, it would go to Q+1 which is invalid.
                    # So probably no ask quote at q=Q either? Or the fill is prevented?
                    # The task says "no bid fill at q = Q, no ask fill at q = -Q"
                    # This suggests at q=Q, bid fills don't happen (maybe we don't quote bid, or fills are prevented)
                    # At q=Q, ask fills CAN happen? But then inventory would exceed Q.
                    # I think the constraint is that we cannot quote bid at q=Q and cannot quote ask at q=-Q.
                    # So at q=Q: no bid quote, ask quote allowed.
                    # At q=-Q: no ask quote, bid quote allowed.
                    
                    # If ask fills, q becomes Q+1 which is out of bounds. This is a contradiction.
                    # Let me assume that at q=Q, we cannot quote ask either (to prevent overflow).
                    # Or maybe the model clips inventory? The task says inventory in [-Q,Q].
                    # I'll assume that at boundaries, only the inward-facing quote is allowed.
                    # At q=Q: only ask quote? No, ask would increase inventory.
                    # Actually, bid fill decreases inventory (buy), ask fill increases inventory (sell).
                    # So at q=Q (max inventory), we want to sell (ask) to reduce inventory.
                    # But "no ask fill at q=-Q" means at min inventory, no ask fills.
                    # So at q=Q, ask fills ARE allowed (they reduce... wait no, ask fill increases inventory).
                    # Let me re-think: 
                    # - Bid fill: we buy, inventory increases by 1? Or decreases?
                    # Typically in market making:
                    # - Bid quote: we offer to buy. If filled, we buy, inventory increases.
                    # - Ask quote: we offer to sell. If filled, we sell, inventory decreases.
                    # But the task says "A fill earns its depth and moves inventory one unit"
                    # It doesn't specify direction. Let me assume:
                    # - Bid fill: inventory += 1 (we bought)
                    # - Ask fill: inventory -= 1 (we sold)
                    # Then "no bid fill at q=Q" makes sense: can't buy more if at max inventory.
                    # "no ask fill at q=-Q" makes sense: can't sell more if at min inventory.
                    
                    # So at q=Q: no bid quote (or bid fills prevented)
                    # At q=-Q: no ask quote (or ask fills prevented)
                    
                    # For q=Q:
                    # Can quote ask, but not bid.
                    # If ask fills, q becomes Q-1.
                    for i, d in enumerate(depth_grid):
                        p_ask = fill_probs[i]
                        val = d * p_ask - penalty + (1 - p_ask) * V_next[q_idx] + p_ask * V_next[q_idx - 1]
                        if val > best_val:
                            best_val = val
                            best_bid = 0.0  # No bid quote
                            best_ask = d
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_bid
                opt_ask[n, q_idx] = best_ask
                
            elif q == -Q:
                # No ask quote
                best_val = -penalty
                best_bid = 0.0
                best_ask = 0.0
                for i, d in enumerate(depth_grid):
                    p_bid = fill_probs[i]
                    val = d * p_bid - penalty + (1 - p_bid) * V_next[q_idx] + p_bid * V_next[q_idx + 1]
                    if val > best_val:
                        best_val = val
                        best_bid = d
                        best_ask = 0.0
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_bid
                opt_ask[n, q_idx] = best_ask
                
            else:
                # Can quote both sides
                best_val = -np.inf
                best_bid = 0.0
                best_ask = 0.0
                for i, db in enumerate(depth_grid):
                    for j, da in enumerate(depth_grid):
                        p_bid = fill_probs[i]
                        p_ask = fill_probs[j]
                        # Both can fill independently
                        p_both = p_bid * p_ask
                        p_bid_only = p_bid * (1 - p_ask)
                        p_ask_only = (1 - p_bid) * p_ask
                        p_none = (1 - p_bid) * (1 - p_ask)
                        
                        # Reward: depths earned
                        reward = db * p_bid + da * p_ask - penalty
                        
                        # Next state value
                        # If bid fills: q+1, if ask fills: q-1
                        # Both fill: q+1-1 = q (net zero)
                        next_val = (p_both * V_next[q_idx] + 
                                   p_bid_only * V_next[q_idx + 1] + 
                                   p_ask_only * V_next[q_idx - 1] + 
                                   p_none * V_next[q_idx])
                        
                        val = reward + next_val
                        if val > best_val:
                            best_val = val
                            best_bid = db
                            best_ask = da
                
                V_curr[q_idx] = best_val
                opt_bid[n, q_idx] = best_bid
                opt_ask[n, q_idx] = best_ask
        
        V_next = V_curr
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """Quote using pre-computed optimal policy with inventory-based skew."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        # Try to compute if we have time
        if time.time() < _DEADLINE - 10.0:
            try:
                _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
            except:
                pass
    
    if key in _POLICY_CACHE:
        opt_bid, opt_ask = _POLICY_CACHE[key]
        # Map inventory to index
        q_idx = np.clip(inventory + Q, 0, 2 * Q)
        bid_depths = opt_bid[n, q_idx]
        ask_depths = opt_ask[n, q_idx]
        # Ensure valid depths
        bid_depths = np.where(bid_depths < 1e-4, np.inf, bid_depths)
        ask_depths = np.where(ask_depths < 1e-4, np.inf, ask_depths)
        return bid_depths, ask_depths
    else:
        # Fallback: simple inventory-based skew
        # Depth = 1/k - c * q, where c is chosen based on parameters
        base_depth = 1.0 / k
        # Skew coefficient: higher gamma and sigma mean more risk aversion
        skew = gamma * sigma**2 * (T - t) / (2.0 * k)
        bid_depths = base_depth + skew * inventory  # More inventory -> deeper bid (less aggressive buying)
        ask_depths = base_depth - skew * inventory  # More inventory -> shallower ask (more aggressive selling)
        
        # Clip to valid range
        bid_depths = np.clip(bid_depths, 1e-4, 10.0)
        ask_depths = np.clip(ask_depths, 1e-4, 10.0)
        
        return bid_depths, ask_depths
# EVOLVE-BLOCK-END
