"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 300.0  # seconds total budget
_DEADLINE = _START_TIME + _TIMEOUT - 5.0  # safety margin


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q, max_depth=5.0, n_depths=200):
    """
    Compute optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time Bellman equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Candidate depths: log-spaced from very small to max_depth
    # We need depths > 0 and finite
    depths = np.linspace(0.01, max_depth, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Value function: V[q] = optimal expected objective from current step with inventory q
    # Start from the last step (no future steps, so value is just the inventory cost for this step)
    # Actually, let's think about this more carefully.
    
    # At each step n, with inventory q:
    # 1. Pay inventory cost: (gamma/2) * sigma^2 * q^2 * dt
    # 2. Choose bid_depth and ask_depth
    # 3. Bid fills with prob p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt) [if q < Q]
    #    Ask fills with prob p_ask = 1 - exp(-A * exp(-k * ask_depth) * dt) [if q > -Q]
    # 4. If bid fills: earn bid_depth, inventory becomes q+1
    #    If ask fills: earn ask_depth, inventory becomes q-1
    #    Both can fill independently
    # 5. Then proceed to next step with updated inventory
    
    # V[n][q] = -cost + max over (bid_depth, ask_depth) of:
    #   E[earnings + V[n+1][q']]
    # where q' depends on which fills occurred
    
    # For the last step (n = steps-1), V[steps][q] = 0 (no terminal charge, no future steps)
    
    # Let's use arrays indexed by inventory offset: q_idx = q + Q, so q_idx in [0, 2Q]
    
    # V_next[q_idx] for the next step
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Process steps in reverse
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() > _DEADLINE:
            # Return what we have so far (remaining steps use default depth)
            default_depth = 1.0 / k
            for nn in range(n, steps):
                bid_table[nn, :] = default_depth
                ask_table[nn, :] = default_depth
            return bid_table, ask_table
        
        V_curr = np.zeros(n_inv)
        
        for q_idx in range(n_inv):
            q = q_idx - Q  # actual inventory
            
            # Inventory cost for this step
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Determine which fills are possible
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_val = -np.inf
            best_bid = 1.0 / k
            best_ask = 1.0 / k
            
            # Try all combinations of bid and ask depths
            # To speed up, we can use the fact that the optimal depth is likely near 1/k
            # But let's first try a reduced set of depths
            
            # Use a smaller set of candidate depths for speed
            # Focus on depths around 1/k
            center = 1.0 / k
            # Create a log-spaced set centered around 1/k
            ratio = 2.0
            n_per_side = 30
            bid_candidates = np.concatenate([
                np.logspace(np.log10(0.01), np.log10(center), n_per_side),
                np.logspace(np.log10(center), np.log10(max_depth), n_per_side + 1)
            ])
            ask_candidates = bid_candidates.copy()
            
            for bid_d in bid_candidates:
                if not can_bid:
                    p_bid = 0.0
                else:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                
                for ask_d in ask_candidates:
                    if not can_ask:
                        p_ask = 0.0
                    else:
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                    
                    # Four cases: no fill, bid only, ask only, both
                    # No fill: prob (1-p_bid)(1-p_ask), inventory stays q
                    # Bid only: prob p_bid(1-p_ask), earn bid_d, inventory q+1
                    # Ask only: prob (1-p_bid)p_ask, earn ask_d, inventory q-1
                    # Both: prob p_bid*p_ask, earn bid_d + ask_d, inventory q (net 0)
                    
                    p_no = (1 - p_bid) * (1 - p_ask)
                    p_bid_only = p_bid * (1 - p_ask)
                    p_ask_only = (1 - p_bid) * p_ask
                    p_both = p_bid * p_ask
                    
                    # Next inventory indices
                    q_idx_no = q_idx
                    q_idx_bid = q_idx + 1 if can_bid else q_idx
                    q_idx_ask = q_idx - 1 if can_ask else q_idx
                    q_idx_both = q_idx  # net zero
                    
                    # Ensure indices are valid
                    q_idx_bid = min(q_idx_bid, n_inv - 1)
                    q_idx_ask = max(q_idx_ask, 0)
                    
                    # Expected value
                    ev = (p_no * V_next[q_idx_no] +
                          p_bid_only * (bid_d + V_next[q_idx_bid]) +
                          p_ask_only * (ask_d + V_next[q_idx_ask]) +
                          p_both * (bid_d + ask_d + V_next[q_idx_both]))
                    
                    total = -cost + ev
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            V_curr[q_idx] = best_val
            bid_table[n, q_idx] = best_bid
            ask_table[n, q_idx] = best_ask
        
        V_next = V_curr
    
    return bid_table, ask_table


# Cache for policy tables
_POLICY_CACHE = {}


def _get_policy_table(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the policy table for given parameters."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _POLICY_CACHE:
        if time.time() > _DEADLINE:
            # Too late, return default
            default_depth = 1.0 / k
            bid_table = np.full((steps, 2 * Q + 1), default_depth)
            ask_table = np.full((steps, 2 * Q + 1), default_depth)
            _POLICY_CACHE[key] = (bid_table, ask_table)
            return bid_table, ask_table
        
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = (bid_table, ask_table)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on pre-computed optimal policy.
    
    Parameters:
        t: current time
        inventory: array of shape (4000,) with integer inventories
        mid: array of shape (4000,) with mid prices
        params: dict with 'sigma', 'A', 'k', 'gamma', 'T', 'steps', 'Q'
    
    Returns:
        (bid_depths, ask_depths): two arrays of shape (4000,)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Check time budget
    if time.time() > _DEADLINE:
        default_depth = 1.0 / k
        return np.full(len(inventory), default_depth), np.full(len(inventory), default_depth)
    
    # Get or compute policy table
    bid_table, ask_table = _get_policy_table(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to table index: q_idx = q + Q
    q_idx = np.array(inventory) + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    # Ensure no invalid values
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    return bid_depths, ask_depths
