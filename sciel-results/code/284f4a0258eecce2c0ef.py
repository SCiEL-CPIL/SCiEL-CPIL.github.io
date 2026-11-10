"""Market making with inventory risk – DP-based optimal quoting."""
import numpy as np
import time

# Global cache for DP tables
_dp_cache = {}

def _get_dp_table(params):
    """Compute or retrieve the DP table for given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _dp_cache:
        return _dp_cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # DP over inventory q in [-Q, Q] and time step n
    # V[n, q] = optimal expected value from step n onwards given inventory q
    # We work backwards from n=steps to n=0
    
    # For efficiency, we'll precompute the optimal bid and ask depths for each (n, q)
    # The objective per step: 
    #   If bid filled: earn bid_depth, inventory becomes q-1
    #   If ask filled: earn ask_depth, inventory becomes q+1
    #   Both can't fill simultaneously (independent but we treat as mutually exclusive for simplicity, 
    #   or handle both cases)
    #   Inventory penalty: (gamma/2) * sigma^2 * q^2 * dt
    
    # Actually, let's think about this more carefully.
    # At each step, given inventory q:
    # - Choose bid_depth b and ask_depth a
    # - Bid fill prob: p_b = 1 - exp(-A * exp(-k*b) * dt)
    # - Ask fill prob: p_a = 1 - exp(-A * exp(-k*a) * dt)
    # - These are independent, so:
    #   P(both) = p_b * p_a
    #   P(bid only) = p_b * (1-p_a)
    #   P(ask only) = (1-p_b) * p_a
    #   P(neither) = (1-p_b) * (1-p_a)
    # - If bid only: earn b, new inventory q-1 (if q < Q)
    # - If ask only: earn a, new inventory q+1 (if q > -Q)
    # - If both: earn b + a, new inventory q (bid and ask cancel)
    # - If neither: earn 0, new inventory q
    # - Penalty: (gamma/2) * sigma^2 * q^2 * dt
    
    # The DP recurrence:
    # V[n, q] = max_{b,a} [ - (gamma/2)*sigma^2*q^2*dt 
    #   + p_b*(1-p_a) * (b + V[n+1, q-1])  [if q < Q, else b*0 + V[n+1,q] for bid part... wait]
    #   ... this is getting complex. Let me restructure.
    
    # Actually, the fill events:
    # - Bid fill: probability p_b, but only if q < Q (can't go below -Q... wait, inventory convention)
    # Let me re-read: "no bid fill at q = Q, no ask fill at q = -Q"
    # So if q = Q, bid fill is not allowed (inventory can't increase further? or decrease?)
    # Wait: "A fill earns its depth and moves inventory one unit"
    # Bid fill: you sell, so inventory decreases? Or increases?
    # Typically in market making:
    # - Bid fill: you buy, inventory increases
    # - Ask fill: you sell, inventory decreases
    # But the problem says "no bid fill at q = Q" and "no ask fill at q = -Q"
    # If Q is max inventory, then bid fill (buying) would increase inventory, so at q=Q you can't buy more.
    # So: bid fill -> q increases by 1, ask fill -> q decreases by 1.
    # Constraint: bid fill only if q < Q, ask fill only if q > -Q.
    
    # Let me redefine:
    # - Bid fill: q -> q+1, earn b
    # - Ask fill: q -> q-1, earn a
    # - Bid fill allowed if q < Q
    # - Ask fill allowed if q > -Q
    
    # DP: V[n, q] for n in [0, steps], q in [-Q, Q]
    # V[steps, q] = 0 (no terminal charge)
    # V[n, q] = max_{b,a >= 0} [ - (gamma/2)*sigma^2*q^2*dt 
    #   + E[reward + V[n+1, q']] ]
    
    # For a given (b, a):
    # p_b = 1 - exp(-A*exp(-k*b)*dt) if q < Q, else 0
    # p_a = 1 - exp(-A*exp(-k*a)*dt) if q > -Q, else 0
    # 
    # Cases:
    # 1. Neither: (1-p_b)*(1-p_a) * V[n+1, q]
    # 2. Bid only: p_b*(1-p_a) * (b + V[n+1, q+1])  [if q < Q]
    # 3. Ask only: (1-p_b)*p_a * (a + V[n+1, q-1])  [if q > -Q]
    # 4. Both: p_b*p_a * (b + a + V[n+1, q])
    
    # This is a 2D optimization over (b, a) at each (n, q). That's expensive.
    # But note that the optimal b and a might be separable or have a simple structure.
    
    # Alternative: since the fill probabilities are monotonic in depth, and the reward is linear in depth,
    # there might be an optimal depth that balances the marginal benefit of more fills vs. the cost of deeper quotes.
    
    # Given the time constraint (330s), let's use a simplified approach:
    # Precompute the optimal policy using a grid search over depths for each (n, q).
    
    # Grid for depths: from 0 to some max_depth
    max_depth = 5.0 / k  # reasonable upper bound
    n_depths = 100
    depths = np.linspace(0, max_depth, n_depths)
    
    # V[n, q] table
    # n: 0..steps, q: -Q..Q
    # We'll store V as a 2D array: V[n+1, q+Q] for indexing
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    # V[steps, :] = 0 (terminal)
    
    # Also store optimal bid and ask depths
    opt_bid = np.zeros((steps + 1, 2 * Q + 1))
    opt_ask = np.zeros((steps + 1, 2 * Q + 1))
    
    # Precompute exp(-k * depth) for the grid
    exp_k_depth = np.exp(-k * depths)
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            penalty = 0.5 * gamma * sigma**2 * q**2 * dt
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            for i, b in enumerate(depths):
                p_b = 1.0 - np.exp(-A * exp_k_depth[i] * dt) if can_bid else 0.0
                for j, a in enumerate(depths):
                    p_a = 1.0 - np.exp(-A * exp_k_depth[j] * dt) if can_ask else 0.0
                    
                    # Expected value
                    val = -penalty
                    
                    # Neither
                    val += (1 - p_b) * (1 - p_a) * V[n + 1, q_idx]
                    
                    # Bid only
                    if can_bid:
                        val += p_b * (1 - p_a) * (b + V[n + 1, q_idx + 1])
                    
                    # Ask only
                    if can_ask:
                        val += (1 - p_b) * p_a * (a + V[n + 1, q_idx - 1])
                    
                    # Both
                    val += p_b * p_a * (b + a + V[n + 1, q_idx])
                    
                    if val > best_val:
                        best_val = val
                        best_b = b
                        best_a = a
            
            V[n, q_idx] = best_val
            opt_bid[n, q_idx] = best_b
            opt_ask[n, q_idx] = best_a
    
    _dp_cache[key] = (opt_bid, opt_ask)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int32)
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    try:
        opt_bid, opt_ask = _get_dp_table(params)
        
        # Clip inventory to valid range
        inv_clipped = np.clip(inv, -Q, Q)
        q_indices = inv_clipped + Q
        
        bid_depths = opt_bid[n, q_indices].astype(np.float64)
        ask_depths = opt_ask[n, q_indices].astype(np.float64)
        
        # Ensure valid depths: >= -3/k, no NaN/inf
        min_depth = -3.0 / k
        bid_depths = np.maximum(bid_depths, min_depth)
        ask_depths = np.maximum(ask_depths, min_depth)
        bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=0.0, neginf=0.0)
        ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=0.0, neginf=0.0)
        
        return bid_depths, ask_depths
    
    except Exception:
        # Fallback: risk-neutral depth
        depth = np.full(n_paths, 1.0 / k)
        return depth, depth.copy()
