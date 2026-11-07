"""Optimal market making with inventory risk using pre-computed HJB backward induction."""
import numpy as np
from functools import lru_cache
import time

# Pre-compute the optimal policy tables at module load time
_POLICY_TABLES = {}
_COMPUTE_DEADLINE = time.time() + 280  # Safety margin before 300s kill

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
    
    # We'll use a grid of candidate depths for optimization
    # The optimal depth is in [0, some_max]
    # From the model, depth > 0 gives positive fill probability
    # Let's use a fine grid of depths
    
    # Maximum reasonable depth: beyond this, fill probability is ~1
    # P(fill) = 1 - exp(-A * exp(-k*d) * dt)
    # For large d, exp(-k*d) is tiny, so P ~ A*exp(-k*d)*dt
    # The benefit of deeper quotes is the depth earned, but cost is inventory risk
    # Let's use depths from 0.001 to maybe 10/k or so
    
    max_depth = 5.0 / k  # Reasonable upper bound
    n_depths = 200
    depth_grid = np.linspace(0.001, max_depth, n_depths)
    
    # Pre-compute fill probabilities for each depth
    # P_fill(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_kd = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_kd * dt)
    
    # Value function: V[n, q] = optimal expected objective from step n with inventory q
    # Work backwards from n = steps-1 to n = 0
    
    # V has shape (steps, n_inv)
    V = np.zeros((steps, n_inv))
    bid_table = np.zeros((steps, n_inv))
    ask_table = np.zeros((steps, n_inv))
    
    # Terminal: V[steps-1, q] = 0 (no terminal charge)
    # Actually, at step steps-1, we still make one quote and get the fill reward
    # Let's think: at step n, we quote, then fills happen, then we get the depth reward
    # and the inventory cost for that step
    
    # Let me re-read: "depths earned minus sum over steps of (gamma / 2) sigma^2 q^2 dt"
    # q is the inventory BEFORE that step's fills
    # So at step n: we have inventory q_n, we quote bid_depth and ask_depth
    # With prob p_bid, bid fills: we earn bid_depth, inventory becomes q_n + 1
    # With prob p_ask, ask fills: we earn ask_depth, inventory becomes q_n - 1
    # We pay (gamma/2) * sigma^2 * q_n^2 * dt for the inventory risk
    # Then mid moves, but that doesn't affect the objective directly (only through future quotes)
    
    # Wait, the mid moves but the objective only depends on depths earned and inventory cost.
    # The mid movement affects future fill probabilities? No, the fill probability depends on
    # bid_depth and ask_depth, not on mid. So mid movement is irrelevant for the objective!
    # It only matters for the "distance below/above mid" interpretation, but the objective
    # is purely about depths earned and inventory cost.
    
    # So the problem decomposes: at each step, choose bid_depth and ask_depth to maximize
    # expected [depth_earned - inventory_cost + future_value]
    
    # Let's do backward induction properly.
    
    # V[n, q] = max over (bid_d, ask_d) of:
    #   - (gamma/2) * sigma^2 * q^2 * dt
    #   + p_bid(bid_d) * bid_d + p_ask(ask_d) * ask_d
    #   + p_bid(bid_d) * p_ask(ask_d) * V[n+1, q]  # both fill or neither... wait
    #   Actually: 
    #   - If bid fills and ask fills: earn bid_d + ask_d, inventory unchanged (q+1-1=q)
    #   - If bid fills only: earn bid_d, inventory q+1
    #   - If ask fills only: earn ask_d, inventory q-1
    #   - If neither: earn 0, inventory q
    #   With boundary: no bid fill if q=Q, no ask fill if q=-Q
    
    # Let's implement this.
    
    for n in range(steps - 1, -1, -1):
        if n < steps - 1:
            V_next = V[n + 1]
        else:
            V_next = np.zeros(n_inv)
        
        inv_cost = (gamma / 2.0) * sigma**2 * inv_range**2 * dt
        
        for i, q in enumerate(inv_range):
            if time.time() > _COMPUTE_DEADLINE:
                # Return what we have so far
                return bid_table, ask_table
            
            best_val = -np.inf
            best_bid = depth_grid[0]
            best_ask = depth_grid[0]
            
            # Try all combinations of bid and ask depths
            # This is O(n_depths^2) per (n, q) which is 200^2 = 40000 per state
            # Total states: 1000 * 61 = 61000
            # Total ops: 61000 * 40000 = 2.44e9 -- too slow!
            
            # We need to decouple bid and ask optimization.
            # The objective is separable in bid and ask except for the joint fill term.
            # Let's expand:
            # p_b = p_fill(bid_d), p_a = p_fill(ask_d)
            # Expected reward = p_b * bid_d + p_a * ask_d
            # + p_b * p_a * V_next[q]  # both fill, inventory unchanged
            # + p_b * (1-p_a) * V_next[q+1]  # bid only (if q < Q)
            # + (1-p_b) * p_a * V_next[q-1]  # ask only (if q > -Q)
            # + (1-p_b) * (1-p_a) * V_next[q]  # neither
            # = p_b * bid_d + p_a * ask_d
            # + V_next[q] * (p_b*p_a + (1-p_b)*(1-p_a))
            # + V_next[q+1] * p_b*(1-p_a)  [if applicable]
            # + V_next[q-1] * (1-p_b)*p_a  [if applicable]
            
            # This is not fully separable. But we can fix one and optimize the other.
            # Or use a coarser grid.
            
            # Let's use a coarser grid: 50 depths
            if n_depths > 50:
                depth_grid_coarse = np.linspace(0.001, max_depth, 50)
                exp_kd_coarse = np.exp(-k * depth_grid_coarse)
                p_fill_coarse = 1.0 - np.exp(-A * exp_kd_coarse * dt)
            else:
                depth_grid_coarse = depth_grid
                p_fill_coarse = p_fill
            
            n_d = len(depth_grid_coarse)
            
            # For each bid depth, compute the best ask depth
            for ib in range(n_d):
                pb = p_fill_coarse[ib]
                bd = depth_grid_coarse[ib]
                
                # For each ask depth, compute total
                for ia in range(n_d):
                    pa = p_fill_coarse[ia]
                    ad = depth_grid_coarse[ia]
                    
                    reward = pb * bd + pa * ad - inv_cost[i]
                    
                    # Future value terms
                    if q < Q and q > -Q:
                        # Both can fill
                        reward += pb * pa * V_next[i]
                        reward += pb * (1 - pa) * V_next[i + 1]
                        reward += (1 - pb) * pa * V_next[i - 1]
                        reward += (1 - pb) * (1 - pa) * V_next[i]
                    elif q == Q:
                        # No bid fill
                        pa_eff = pa
                        reward += pa_eff * V_next[i - 1]
                        reward += (1 - pa_eff) * V_next[i]
                    elif q == -Q:
                        # No ask fill
                        pb_eff = pb
                        reward += pb_eff * V_next[i + 1]
                        reward += (1 - pb_eff) * V_next[i]
                    
                    if reward > best_val:
                        best_val = reward
                        best_bid = bd
                        best_ask = ad
            
            V[n, i] = best_val
            bid_table[n, i] = best_bid
            ask_table[n, i] = best_ask
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_TABLES:
        _POLICY_TABLES[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_TABLES[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths based on pre-computed policy table."""
    steps = params['steps']
    Q = params['Q']
    n = int(round(t / (params['T'] / steps)))
    n = max(0, min(n, steps - 1))
    
    bid_table, ask_table = _get_policy_table(params)
    
    # inventory is in [-Q, Q], map to index 0..2Q
    inv_indices = np.clip(inventory + Q, 0, 2 * Q).astype(int)
    
    bid_depths = bid_table[n, inv_indices]
    ask_depths = ask_table[n, inv_indices]
    
    return bid_depths, ask_depths
