"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
import time
from functools import lru_cache

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds

# Cache for policy tables
_policy_cache = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each (step, inventory) pair
    using backward induction on the discrete-time HJB equation.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1) with optimal bid depths
        ask_table: array of shape (steps, 2*Q+1) with optimal ask depths
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]
    
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_range = np.arange(-Q, Q + 1)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # We work backwards from n = steps-1 to n = 0
    
    # For each step, we need to find optimal bid_depth and ask_depth
    # The objective per step: depth_earned - (gamma/2)*sigma^2*q^2*dt
    # Bid hit prob: 1 - exp(-A*exp(-k*bid_depth)*dt)
    # Ask hit prob: 1 - exp(-A*exp(-k*ask_depth)*dt)
    
    # Terminal condition: V[steps, q] = 0 (no terminal charge)
    V_next = np.zeros(n_inv)
    
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Pre-compute fill probabilities as function of depth
    # For a given depth d, fill prob = 1 - exp(-A*exp(-k*d)*dt)
    # We'll search over a grid of depths
    
    # Depth grid: from 0 to some max
    # The optimal depth is bounded. Let's use a reasonable grid.
    # For small k*depth, fill prob ~ A*exp(-k*d)*dt
    # We want to balance depth earned vs inventory risk
    
    # Use a logarithmic or linear grid for depths
    # Max reasonable depth: when fill prob becomes negligible
    # A*exp(-k*d_max)*dt ~ 1e-6 => d_max ~ (1/k)*ln(A*dt/1e-6)
    # But we also need to consider the risk term
    
    # Let's use a grid from 0 to d_max with sufficient resolution
    d_max = max(5.0 / k, 1.0)  # reasonable upper bound
    n_depths = 200
    depth_grid = np.linspace(0.0, d_max, n_depths)
    
    # For each step, compute optimal depths for each inventory level
    for n in range(steps - 1, -1, -1):
        # For each inventory level, find optimal bid and ask depths
        # V[n, q] = max_{bid, ask} [ 
        #   p_bid * (bid_depth - V_next[q+1] if q < Q else 0) 
        #   + p_ask * (ask_depth - V_next[q-1] if q > -Q else 0)
        #   + (1-p_bid)*(1-p_ask) * 0  # no fill, no depth earned
        #   - (gamma/2)*sigma^2*q^2*dt
        # ]
        # Wait, I need to be more careful.
        
        # The objective is: depths earned minus sum of (gamma/2)*sigma^2*q^2*dt
        # "depths earned" means when a bid fill occurs, you earn bid_depth
        # when an ask fill occurs, you earn ask_depth
        
        # So the step reward is:
        # p_bid * bid_depth + p_ask * ask_depth - (gamma/2)*sigma^2*q^2*dt
        # Plus the expected future value:
        # p_bid * (V_next[q+1] if q < Q else 0) 
        # + p_ask * (V_next[q-1] if q > -Q else 0)
        # + (1-p_bid)*(1-p_ask) * V_next[q]
        # + p_bid*p_ask * V_next[q]  # both fill? No, they're independent but only one can happen?
        
        # Actually, re-reading: "bid hit with probability ..., ask likewise, independently"
        # "A fill earns its depth and moves inventory one unit"
        # So if both hit, you get both depths and inventory changes by... ?
        # "no bid fill at q = Q, no ask fill at q = -Q"
        # If both hit at interior q, inventory could go to q+1 (bid) and q-1 (ask)?
        # That's ambiguous. Let me re-read.
        
        # "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt), ask likewise, independently"
        # "A fill earns its depth and moves inventory one unit"
        
        # I think if both hit, you earn both depths, and inventory... 
        # Actually, in market making, if you quote bid and ask, and both are hit,
        # you buy one (bid hit) and sell one (ask hit), so net inventory change is 0.
        # But the problem says "moves inventory one unit" per fill.
        
        # Let me assume: if bid hits, inventory increases by 1 (you buy).
        # If ask hits, inventory decreases by 1 (you sell).
        # If both hit, inventory is unchanged (buy and sell cancel).
        # You earn bid_depth + ask_depth in that case.
        
        # So:
        # p_bid = 1 - exp(-A*exp(-k*bid_depth)*dt)
        # p_ask = 1 - exp(-A*exp(-k*ask_depth)*dt)
        
        # Reward this step: p_bid*bid_depth + p_ask*ask_depth - (gamma/2)*sigma^2*q^2*dt
        # Future value:
        # p_bid*(1-p_ask) * V_next[q+1]  (if q < Q, else 0)
        # (1-p_bid)*p_ask * V_next[q-1]  (if q > -Q, else 0)
        # [p_bid*p_ask + (1-p_bid)*(1-p_ask)] * V_next[q]
        
        # Actually wait, if both hit, inventory doesn't change, so V_next[q].
        # If only bid hits, inventory goes to q+1, so V_next[q+1].
        # If only ask hits, inventory goes to q-1, so V_next[q-1].
        # If neither hits, inventory stays at q, so V_next[q].
        
        # Let me code this up.
        
        for idx, q in enumerate(inv_range):
            best_val = -np.inf
            best_bid = 0.0
            best_ask = 0.0
            
            # Search over bid and ask depths
            # To speed up, we can do a nested loop but it's O(n_depths^2) per (n, q)
            # That's 200^2 = 40000 per (n,q), times 1000 steps * 61 inventory = 244 million
            # That's too slow.
            
            # Alternative: separate the optimization? 
            # The value function couples bid and ask through the future value terms.
            # But maybe we can use the fact that the problem is separable in some way?
            
            # Actually, let me think about this differently.
            # The key insight is that the optimal policy might have a structure.
            
            # For now, let's try a coarser grid and see if it's fast enough.
            # Or use a smarter search.
            
            # Let's try: for each bid_depth, find the best ask_depth, then optimize over bid_depth.
            # This is O(n_depths^2) still.
            
            # Alternative: use a smaller grid.
            n_depths_small = 50
            depth_grid_small = np.linspace(0.0, d_max, n_depths_small)
            
            for i, bid_d in enumerate(depth_grid_small):
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                
                for j, ask_d in enumerate(depth_grid_small):
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                    
                    # Compute expected future value
                    val_future = 0.0
                    if q < Q:
                        val_future += p_bid * (1.0 - p_ask) * V_next[idx + 1]
                    if q > -Q:
                        val_future += (1.0 - p_bid) * p_ask * V_next[idx - 1]
                    val_future += (p_bid * p_ask + (1.0 - p_bid) * (1.0 - p_ask)) * V_next[idx]
                    
                    # Step reward
                    step_reward = p_bid * bid_d + p_ask * ask_d - 0.5 * gamma * sigma**2 * q**2 * dt
                    
                    total = step_reward + val_future
                    
                    if total > best_val:
                        best_val = total
                        best_bid = bid_d
                        best_ask = ask_d
            
            V_next[idx] = best_val
            bid_table[n, idx] = best_bid
            ask_table[n, idx] = best_ask
        
        # Check deadline
        if time.time() > _DEADLINE - 30:
            # Return what we have so far
            break
    
    _policy_cache[cache_key] = (bid_table, ask_table)
    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Quote both sides using the pre-computed optimal policy.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n = int(round(t / (T / steps)))
    n = max(0, min(n, steps - 1))
    
    bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
    
    # inventory is in [-Q, Q], map to index
    inv_idx = inventory + Q
    
    bid_depths = bid_table[n, inv_idx]
    ask_depths = ask_table[n, inv_idx]
    
    # Ensure validity: values must be >= -3/k, not -inf, not NaN
    min_valid = -3.0 / k
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, min_valid)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, min_valid)
    
    # Enforce no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q, don't quote bid (set to inf)
    # At q=-Q, don't quote ask (set to inf)
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    return bid_depths, ask_depths
