import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

# Cache for computed policies
_policy_cache = {}


def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths for all steps and inventory levels
    via backward induction on the discrete MDP.
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]

    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q

    # Depth grid - use a fine grid in the region where the optimum typically lies
    # For k=1.5: 10/k ≈ 6.67; for k=0.75: 10/k ≈ 13.33
    depth_max = min(15.0 / k, 50.0)  # cap at reasonable value
    depth_min = 0.001
    # Use more points near zero where the function is more sensitive
    n_grid = 150
    # Non-uniform grid: denser near zero
    t_grid = np.linspace(0, 1, n_grid)
    # Apply a power transform to concentrate points near 0
    depths = depth_min + (depth_max - depth_min) * (t_grid ** 1.5)
    depths = np.maximum(depths, depth_min)

    # Precompute fill probabilities for all depths
    # p_fill[depth_idx] = 1 - exp(-A * exp(-k * depth) * dt)
    exp_term = np.exp(-k * depths)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)

    # V[n, q_idx] = optimal value from step n with inventory q
    V = np.zeros((n_steps + 1, n_inv))

    # Action tables
    opt_bid = np.zeros((n_steps, n_inv))
    opt_ask = np.zeros((n_steps, n_inv))

    # Precompute the 2D arrays for interior states
    # p_fill_2d[i,j] = p_fill[i] * p_fill[j]
    p_2d = np.outer(p_fill, p_fill)
    bd_2d = depths[:, np.newaxis]  # (D, 1)
    ad_2d = depths[np.newaxis, :]  # (1, D)

    # For boundary states, we only need 1D
    # Precompute the "delta" for boundaries
    # At q=Q: only ask side matters. val = inv_cost + p_ad * (ad + V[q-1] - V[q]) + V[q]
    # At q=-Q: only bid side matters. val = inv_cost + p_bd * (bd + V[q+1] - V[q]) + V[q]

    # Backward induction with time guard
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - 5:
            # Fill remaining with a simple heuristic
            for remaining_n in range(n, -1, -1):
                for q_idx in range(n_inv):
                    q = q_idx - Q
                    base_depth = 2.0 / k
                    if q == Q:
                        opt_bid[remaining_n, q_idx] = np.inf
                        opt_ask[remaining_n, q_idx] = base_depth
                    elif q == -Q:
                        opt_bid[remaining_n, q_idx] = base_depth
                        opt_ask[remaining_n, q_idx] = np.inf
                    else:
                        # Adjust for inventory
                        risk_adj = gamma * sigma**2 * abs(q) * dt
                        adj_depth = base_depth * (1.0 + 0.5 * risk_adj)
                        if q > 0:
                            # Positive inventory: want to sell more -> deeper ask
                            opt_bid[remaining_n, q_idx] = adj_depth
                            opt_ask[remaining_n, q_idx] = base_depth * 0.8
                        elif q < 0:
                            # Negative inventory: want to buy more -> deeper bid
                            opt_bid[remaining_n, q_idx] = base_depth * 0.8
                            opt_ask[remaining_n, q_idx] = adj_depth
                        else:
                            opt_bid[remaining_n, q_idx] = base_depth
                            opt_ask[remaining_n, q_idx] = base_depth
            break

        # Process all inventory levels
        for q_idx in range(n_inv):
            q = q_idx - Q

            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma**2 * q**2 * dt

            if q == Q:
                # No bid fill possible (bid_depth = inf)
                # Only ask side matters
                V_q_minus = V[n + 1, q_idx - 1]
                V_q = V[n + 1, q_idx]

                # val = inv_cost + p_ad * (ad + V_q_minus - V_q) + V_q
                delta_val = depths + V_q_minus - V_q
                vals = inv_cost + p_fill * delta_val + V_q

                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_bd = np.inf
                best_ad = depths[max_idx]

            elif q == -Q:
                # No ask fill possible (ask_depth = inf)
                # Only bid side matters
                V_q_plus = V[n + 1, q_idx + 1]
                V_q = V[n + 1, q_idx]

                delta_val = depths + V_q_plus - V_q
                vals = inv_cost + p_fill * delta_val + V_q

                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_bd = depths[max_idx]
                best_ad = np.inf

            else:
                # Both sides active
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]
                V_q_minus = V[n + 1, q_idx - 1]

                # p_bid_2d shape (D, 1), p_ask_2d shape (1, D)
                p_bid_2d = p_fill[:, np.newaxis]  # (D, 1)
                p_ask_2d = p_fill[np.newaxis, :]  # (1, D)

                # Fill probabilities
                p_both = p_bid_2d * p_ask_2d  # (D, D)
                p_bid_only = p_bid_2d * (1.0 - p_ask_2d)  # (D, D)
                p_ask_only = (1.0 - p_bid_2d) * p_ask_2d  # (D, D)
                p_neither = (1.0 - p_bid_2d) * (1.0 - p_ask_2d)  # (D, D)

                # Earnings
                earnings = p_bid_2d * bd_2d + p_ask_2d * ad_2d  # (D, D)

                # Future value
                future = p_both * V_q + p_bid_only * V_q_plus + p_ask_only * V_q_minus + p_neither * V_q

                total = inv_cost + earnings + future

                # Find max
                max_idx = np.unravel_index(np.argmax(total), total.shape)
                best_val = total[max_idx]
                best_bd = depths[max_idx[0]]
                best_ad = depths[max_idx[1]]

            V[n, q_idx] = best_val
            opt_bid[n, q_idx] = best_bd
            opt_ask[n, q_idx] = best_ad

    result = (opt_bid, opt_ask)
    _policy_cache[cache_key] = result
    return result


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    n_paths = len(inventory)

    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    if step >= steps:
        step = steps - 1

    # Get or compute DP solution
    opt_bid, opt_ask = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)

    # Map inventory to index
    inv_idx = np.asarray(inventory, dtype=np.int64) + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)

    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()

    # Ensure valid depths: must be > -3/k
    min_depth = -3.0 / k + 1e-9
    bid_depths[bid_depths <= min_depth] = np.inf
    ask_depths[ask_depths <= min_depth] = np.inf

    # Ensure shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)

    return bid_depths, ask_depths
