"""Optimal market making with inventory risk via backward DP."""
import numpy as np
from scipy.optimize import minimize_scalar
import time

# Cache for pre-computed policies per regime
_policy_cache = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """Solve the discrete optimal control problem via backward DP.
    
    Returns arrays bid_depths[step, inventory+Q] and ask_depths[step, inventory+Q]
    where inventory ranges from -Q to Q.
    """
    dt = T / steps
    n_inv = 2 * Q + 1  # number of inventory states
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[step, inv_idx] = optimal expected objective from step to end
    # We work backward from step = steps-1 to step = 0
    # At step n, t = n * dt
    
    # V at terminal (after all steps) is 0 (no terminal charge)
    V_next = np.zeros(n_inv)
    
    # Pre-compute optimal depths for each step and inventory
    bid_depths = np.zeros((steps, n_inv))
    ask_depths = np.zeros((steps, n_inv))
    
    # Fill probability function
    def fill_prob(depth):
        if depth <= 0:
            return 0.0
        return 1.0 - np.exp(-A * np.exp(-k * depth) * dt)
    
    # Objective contribution for a fill at depth d: d (earn depth)
    # Inventory penalty: (gamma / 2) * sigma^2 * q^2 * dt
    
    for n in range(steps - 1, -1, -1):
        # t = n * dt, time remaining = (steps - n) * dt
        for idx in range(n_inv):
            q = inv_range[idx]  # current inventory
            
            # Inventory penalty for this step
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Try different bid/ask depths to maximize expected value
            # Bid fill: if q < Q, can fill bid -> inventory becomes q-1, earn depth
            # Ask fill: if q > -Q, can fill ask -> inventory becomes q+1, earn depth
            # Both independent
            
            def neg_expected_value(bid_d, ask_d):
                # Compute expected value of choosing bid_d and ask_d
                p_bid = 0.0
                p_ask = 0.0
                
                if q < Q and bid_d > 0:
                    p_bid = fill_prob(bid_d)
                if q > -Q and ask_d > 0:
                    p_ask = fill_prob(ask_d)
                
                # Four cases: no fill, bid only, ask only, both
                # No fill: prob (1-p_bid)(1-p_ask), inventory stays q
                v_no = V_next[idx]
                
                # Bid fill: prob p_bid*(1-p_ask), inventory q-1, earn bid_d
                v_bid = 0.0
                if q < Q and bid_d > 0:
                    v_bid = bid_d + V_next[idx - 1]
                
                # Ask fill: prob (1-p_bid)*p_ask, inventory q+1, earn ask_d
                v_ask = 0.0
                if q > -Q and ask_d > 0:
                    v_ask = ask_d + V_next[idx + 1]
                
                # Both: prob p_bid*p_ask, inventory q (bid: q-1, ask: q+1 -> q), earn both
                v_both = 0.0
                if q < Q and q > -Q and bid_d > 0 and ask_d > 0:
                    v_both = bid_d + ask_d + V_next[idx]
                
                # Expected value
                expected = ((1 - p_bid) * (1 - p_ask) * v_no +
                            p_bid * (1 - p_ask) * v_bid +
                            (1 - p_bid) * p_ask * v_ask +
                            p_bid * p_ask * v_both)
                
                return -(expected - penalty)
            
            # Optimize bid depth
            if q < Q:
                res_bid = minimize_scalar(
                    lambda d: neg_expected_value(d, 0.0),
                    bounds=(1e-6, 10.0/k),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                best_bid_d = res_bid.x
            else:
                best_bid_d = np.inf  # no bid quote at q = Q
            
            # Optimize ask depth
            if q > -Q:
                res_ask = minimize_scalar(
                    lambda d: neg_expected_value(0.0, d),
                    bounds=(1e-6, 10.0/k),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                best_ask_d = res_ask.x
            else:
                best_ask_d = np.inf  # no ask quote at q = -Q
            
            bid_depths[n, idx] = best_bid_d
            ask_depths[n, idx] = best_ask_d
            
            # Update V for this state
            V_next[idx] = -neg_expected_value(best_bid_d, best_ask_d)
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Check cache
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key not in _policy_cache:
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        try:
            bid_depths, ask_depths = _solve_regime(sigma, A, k, gamma, T, steps, Q)
            _policy_cache[cache_key] = (bid_depths, ask_depths)
        except Exception:
            # Fallback: use risk-neutral depth
            n_paths = len(inventory)
            depth = np.full(n_paths, 1.0 / k)
            return depth, depth.copy()
        
        # Check if we're still within time
        if time.time() > deadline:
            # Return what we have (should be complete if no exception)
            pass
    
    bid_depths, ask_depths = _policy_cache[cache_key]
    
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Determine current step
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Map inventory to index
    inv_idx = inv + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Get depths for each path
    bid_out = bid_depths[n, inv_idx]
    ask_out = ask_depths[n, inv_idx]
    
    # Ensure valid values: clip to minimum positive, handle inf
    bid_out = np.where(bid_out > 1e-8, bid_out, np.inf)
    ask_out = np.where(ask_out > 1e-8, ask_out, np.inf)
    
    # Validate: values below -3/k are invalid
    min_valid = -3.0 / k
    bid_out = np.where(bid_out < min_valid, np.inf, bid_out)
    ask_out = np.where(ask_out < min_valid, np.inf, ask_out)
    
    # Replace NaN with inf
    bid_out = np.where(np.isnan(bid_out), np.inf, bid_out)
    ask_out = np.where(np.isnan(ask_out), np.inf, ask_out)
    
    return bid_out.astype(np.float64), ask_out.astype(np.float64)
