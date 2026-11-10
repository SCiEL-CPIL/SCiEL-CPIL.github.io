"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 315.0  # Safety margin before 330s kill

def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses separable 1D optimization: in the small-dt limit, bid and ask decisions
    are approximately independent. We optimize each side separately using the
    expected value function, which is much faster than the full 2D search.
    """
    dt = T / steps
    depths = depth_grid
    n_depths = len(depths)
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # For each side, compute the contribution to the objective:
    # For bid at depth d: immediate reward = d * p(d), and it affects inventory
    #   with probability p(d) (increases q by 1) or (1-p(d)) (no change)
    # For ask at depth d: immediate reward = d * p(d), and it affects inventory
    #   with probability p(d) (decreases q by 1) or (1-p(d)) (no change)
    
    # We'll do a two-pass approach:
    # Pass 1: Compute V using a reasonable approximation (e.g., assume optimal
    #         depths are constant or use a coarse grid), then refine.
    # For speed, we'll use the separable approximation directly.
    
    # The key insight: in the limit of small dt, the cross-term between bid
    # and ask fills is O(dt^2), so we can optimize each side independently.
    # The value of choosing bid depth d at state (n, q) is approximately:
    #   d * p(d) + (1 - p(d)) * V[n+1, q] + p(d) * V[n+1, q+1] - C_inv[q]
    # (ignoring the ask side's contribution to V[n+1], which is handled in
    #  the DP step itself)
    
    # More precisely, we can write:
    # V[n, q] = max_{d_b, d_a} { d_b*p_b + d_a*p_a - C_inv[q]
    #             + (1-p_b)(1-p_a)*V[n+1,q] + p_b(1-p_a)*V[n+1,q+1]
    #             + (1-p_b)p_a*V[n+1,q-1] + p_b*p_a*V[n+1,q] }
    
    # In the small-dt limit, p_b and p_a are small, so p_b*p_a is O(dt^2).
    # We can approximate:
    # V[n, q] ≈ max_{d_b, d_a} { d_b*p_b + d_a*p_a - C_inv[q]
    #             + V[n+1,q] + p_b*(V[n+1,q+1]-V[n+1,q])
    #             + p_a*(V[n+1,q-1]-V[n+1,q]) }
    
    # This separates into:
    # V[n, q] ≈ -C_inv[q] + V[n+1,q]
    #             + max_{d_b} { d_b*p_b + p_b*(V[n+1,q+1]-V[n+1,q]) }
    #             + max_{d_a} { d_a*p_a + p_a*(V[n+1,q-1]-V[n+1,q]) }
    
    # With boundary conditions:
    # - If q = Q, no bid fill: max_{d_b} { d_b*p_b } (but actually we should
    #   set bid depth to inf or handle separately; for now, we allow it but
    #   the probability of fill is still p_b, just inventory doesn't change)
    #   Actually, the problem states "no bid fill at q = Q", so the reward is
    #   still d_b*p_b but inventory stays at Q. So:
    #   max_{d_b} { d_b*p_b + p_b*(V[n+1,Q]-V[n+1,Q]) } = max_{d_b} { d_b*p_b }
    #   Wait, that's not right. If no fill occurs, inventory doesn't change.
    #   So the term is: d_b*p_b + (1-p_b)*V[n+1,Q] + p_b*V[n+1,Q]
    #                = d_b*p_b + V[n+1,Q]
    #   So max over d_b of { d_b*p_b } + V[n+1,Q]
    #   But that would always choose the largest depth! That can't be right.
    
    # Let me reconsider. The issue is that if we're at q=Q and a bid fill
    # occurs, we can't increase inventory further. But the problem says "no
    # bid fill at q=Q", meaning the fill doesn't happen. So the probability
    # of a fill is still p_b, but the inventory transition is q -> q (not q+1).
    
    # So at q=Q:
    # V[n, Q] = max_{d_b, d_a} { d_b*p_b + d_a*p_a - C_inv[Q]
    #             + (1-p_b)(1-p_a)*V[n+1,Q] + p_b(1-p_a)*V[n+1,Q]
    #             + (1-p_b)p_a*V[n+1,Q-1] + p_b*p_a*V[n+1,Q] }
    #          = max_{d_b, d_a} { d_b*p_b + d_a*p_a - C_inv[Q]
    #             + (1-p_a)*V[n+1,Q] + p_a*V[n+1,Q-1] }
    #          = max_{d_b} { d_b*p_b } + max_{d_a} { d_a*p_a + p_a*(V[n+1,Q-1]-V[n+1,Q]) }
    #            - C_inv[Q] + V[n+1,Q]
    
    # Similarly at q=-Q for ask.
    
    # Let's implement this separable optimization.
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break
            
        V_next = V[n + 1]
        
        # For each q, compute the optimal bid and ask depths
        V_new = np.zeros(2 * Q + 1)
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            V_q = V_next[q_idx]
            
            # Bid side
            if q_idx < 2 * Q:  # q < Q, bid fill increases inventory
                delta_plus = V_next[q_idx + 1] - V_q
                bid_terms = depths * p_depths + p_depths * delta_plus
            else:  # q == Q, bid fill doesn't change inventory
                bid_terms = depths * p_depths
            
            # Ask side
            if q_idx > 0:  # q > -Q, ask fill decreases inventory
                delta_minus = V_next[q_idx - 1] - V_q
                ask_terms = depths * p_depths + p_depths * delta_minus
            else:  # q == -Q, ask fill doesn't change inventory
                ask_terms = depths * p_depths
            
            best_bid_idx = np.argmax(bid_terms)
            best_ask_idx = np.argmax(ask_terms)
            
            opt_bid[n, q_idx] = depths[best_bid_idx]
            opt_ask[n, q_idx] = depths[best_ask_idx]
            
            V_new[q_idx] = V_q - C_inv[q_idx] + bid_terms[best_bid_idx] + ask_terms[best_ask_idx]
        
        V[n, :] = V_new
    
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Use a log-spaced grid for better resolution near small depths
    # where the optimal depth typically lies. With separable 1D optimization,
    # we can afford a finer grid.
    n_points = 200
    depth_grid = np.geomspace(0.001, 8.0, n_points)
    depth_grid = np.maximum(depth_grid, 0.001)
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q'], depth_grid
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    min_depth = 0.001
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
