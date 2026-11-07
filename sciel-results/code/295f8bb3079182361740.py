"""
Optimal market making via pre-computed DP policy lookup.
Uses a fully vectorized DP over inventory levels for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 320.0  # safety margin before 330s kill


def _heuristic_depths(params, t, inventory):
    """Analytical fallback: time-varying base depth with inventory skew."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    q = inventory.astype(np.float64)
    t_remaining = np.maximum(T - t, 0.0)
    d0 = 1.0 / k
    skew_scale = gamma * sigma * sigma * t_remaining
    alpha = skew_scale / (A * k) if A > 0 and k > 0 else 0.0
    alpha = min(alpha, 6.0 / k)
    bid_depth = d0 + alpha * q
    ask_depth = d0 - alpha * q
    min_depth = 0.01 / k
    max_depth = 10.0 / k
    bid_depth = np.clip(bid_depth, min_depth, max_depth)
    ask_depth = np.clip(ask_depth, min_depth, max_depth)
    bid_depth = np.where(inventory >= Q, np.inf, bid_depth)
    ask_depth = np.where(inventory <= -Q, np.inf, ask_depth)
    bid_depth = np.where(np.isfinite(bid_depth) & (bid_depth >= -3.0/k), bid_depth, np.inf)
    ask_depth = np.where(np.isfinite(ask_depth) & (ask_depth >= -3.0/k), ask_depth, np.inf)
    return bid_depth, ask_depth


def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via backward DP.
    Fully vectorized over inventory levels.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    # High-resolution depth grid
    depth_min = 0.001
    depth_max = 10.0 / k
    n_depths = 500
    depth_grid = np.geomspace(depth_min, depth_max, n_depths)

    # Precompute fill probabilities and depth*prob
    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # Precompute arrays for broadcasting
    P_grid = p_fill
    DP_grid = depth_p
    
    # Precompute cost term for each q
    q_vals = q_range.astype(np.float64)
    cost_term = (gamma / 2.0) * sigma**2 * q_vals**2 * dt

    for n in range(steps - 1, -1, -1):
        # Time check
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback: fill remaining with heuristic
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            break

        Vn1 = V[n + 1]

        # Compute D1, Dm1, D2 for all q
        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]
        
        D1 = V_plus - Vn1      # V(q+1) - V(q)
        Dm1 = V_minus - Vn1    # V(q-1) - V(q)
        D2 = V_plus + V_minus - 2.0 * Vn1  # V(q+1) + V(q-1) - 2V(q)

        # Boundary handling: set P and DP to 0 where no fill is allowed
        # For q=Q (index n_q-1), no bid fill -> Pb=0, DPb=0
        # For q=-Q (index 0), no ask fill -> Pa=0, DPa=0
        
        # We construct the J tensor of shape (D, D, n_q)
        # J[i, j, q] = DPb[i] + DPa[j] + Vn1[q] + Pb[i]*D1[q] + Pa[j]*Dm1[q] - Pb[i]*Pa[j]*D2[q] - cost[q]
        
        # To handle boundaries, we can create masked versions of P and DP
        # But since D1 and Dm1 are already 0 at boundaries (due to V_plus[-1]=Vn1[-1] etc),
        # the terms Pb*D1 and Pa*Dm1 will be 0 at boundaries even if P is not 0.
        # However, the term DPb[i] is non-zero even if P is 0?
        # Wait, if P=0, then DP should be 0 because DP = d * P.
        # In our precomputed DP_grid, DP = d * P. So if P=0, DP=0.
        # But we are using the same P_grid and DP_grid for all q.
        # For q=Q, we need Pb=0.
        # So we must zero out the P and DP arrays for the specific q?
        # No, the broadcast uses P_grid[:, None, None] which is (D, 1, 1).
        # It is multiplied by D1[None, None, :] which is (1, 1, n_q).
        # So for q=Q, D1[Q]=0. So the term Pb*D1 is 0.
        # But the term DPb[i] is added independently.
        # If we allow a bid depth d_b > 0 at q=Q, we would earn d_b * P_b.
        # But P_b must be 0 at q=Q.
        # So we must force P_b = 0 and DP_b = 0 for q=Q.
        
        # How to do this efficiently?
        # We can create P_b_q of shape (D, n_q) where P_b_q[i, q] = P_grid[i] if q < Q else 0.
        # And DP_b_q of shape (D, n_q) where DP_b_q[i, q] = DP_grid[i] if q < Q else 0.
        
        # This creates two (D, n_q) arrays.
        P_b_q = np.tile(P_grid[:, None], (1, n_q))
        P_b_q[:, n_q - 1] = 0.0  # q = Q
        DP_b_q = np.tile(DP_grid[:, None], (1, n_q))
        DP_b_q[:, n_q - 1] = 0.0
        
        P_a_q = np.tile(P_grid[None, :], (n_q, 1)).T # Shape (D, n_q)
        P_a_q[:, 0] = 0.0  # q = -Q
        DP_a_q = np.tile(DP_grid[None, :], (n_q, 1)).T
        DP_a_q[:, 0] = 0.0

        # Now construct J
        # Shapes:
        # DP_b_q[:, :, None] : (D, n_q, 1) -> Wait, we want (D, D, n_q)
        # Let's rearrange.
        # J[i, j, q] = DP_b_q[i, q] + DP_a_q[j, q] + Vn1[q] + P_b_q[i, q]*D1[q] + P_a_q[j, q]*Dm1[q] - P_b_q[i, q]*P_a_q[j, q]*D2[q] - cost[q]
        
        # We can broadcast:
        # Term 1: DP_b_q[:, None, :] + DP_a_q[None, :, :] -> (D, D, n_q)
        # Term 2: Vn1[None, None, :] -> (1, 1, n_q)
        # Term 3: P_b_q[:, None, :] * D1[None, None, :] -> (D, D, n_q)
        # Term 4: P_a_q[None, :, :] * Dm1[None, None, :] -> (D, D, n_q)
        # Term 5: (P_b_q[:, None, :] * P_a_q[None, :, :]) * D2[None, None, :] -> (D, D, n_q)
        # Term 6: cost[None, None, :] -> (1, 1, n_q)
        
        J = (DP_b_q[:, None, :] + DP_a_q[None, :, :] + 
             Vn1[None, None, :] + 
             P_b_q[:, None, :] * D1[None, None, :] + 
             P_a_q[None, :, :] * Dm1[None, None, :] - 
             (P_b_q[:, None, :] * P_a_q[None, :, :]) * D2[None, None, :] - 
             cost[None, None, :])
        
        # Find max over i, j for each q
        flat_idx = np.argmax(J, axis=(0, 1))
        i_max, j_max = np.unravel_index(flat_idx, (n_depths, n_depths))
        
        V[n, :] = J[i_max, j_max, np.arange(n_q)]
        bid_table[n, :] = depth_grid[i_max]
        ask_table[n, :] = depth_grid[j_max]

    # Post-process boundaries to return inf for no quote
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    if time.time() - _START_TIME > TIME_LIMIT:
        return _heuristic_depths(params, t, inventory)

    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key not in _policy_cache:
        bid_table, ask_table = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_table, ask_table)
    else:
        bid_table, ask_table = _policy_cache[key]

    steps = params['steps']
    Q = params['Q']
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    q_index = inventory + Q
    q_index = np.clip(q_index, 0, 2 * Q)

    bid_depths = bid_table[n, q_index].copy()
    ask_depths = ask_table[n, q_index].copy()

    # Ensure valid values
    min_valid = -3.0 / params['k']
    bid_depths = np.where((bid_depths > min_valid) & np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where((ask_depths > min_valid) & np.isfinite(ask_depths), ask_depths, np.inf)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
