"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation with a reduced depth grid.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # Depth grid: use a log-spaced grid of depths
    # Depths should be positive. We'll use a grid from very small to large.
    # The optimal depth is typically around 1/k or so, but can be larger.
    # Let's use a grid of about 50 points.
    # min_depth = 0.01, max_depth = 10/k
    min_depth = 0.01
    max_depth = 10.0 / k
    n_depths = 50
    depth_grid = np.geomspace(min_depth, max_depth, n_depths)

    # Inventory range: -Q to Q
    inv_range = np.arange(-Q, Q + 1)
    n_inv = len(inv_range)

    # Value function: V[n, i] where n is step (0..steps), i is inventory index
    # V[steps, i] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_inv))

    # Optimal actions: opt_bid[n, i], opt_ask[n, i]
    opt_bid = np.zeros((steps, n_inv))
    opt_ask = np.zeros((steps, n_inv))

    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 10:
            # Fallback: use 1/k for all
            opt_bid[n] = 1.0 / k
            opt_ask[n] = 1.0 / k
            continue

        for i_idx, q in enumerate(inv_range):
            # Inventory cost for this step
            inv_cost = 0.5 * gamma * sigma**2 * q**2 * dt

            # Try each depth in the grid
            # For bid: only if q < Q
            if q < Q:
                best_val_bid = -np.inf
                best_depth_bid = 1.0 / k
                for d in depth_grid:
                    # Probability of bid fill
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    # If filled, we earn depth d and inventory decreases by 1
                    # New inventory: q - 1
                    new_q = q - 1
                    # Check bounds
                    if new_q < -Q:
                        continue
                    new_i_idx = int(new_q + Q)
                    # Value: d * p_bid + (1 - p_bid) * 0 + V[n+1, new_i_idx] * p_bid + V[n+1, i_idx] * (1 - p_bid)
                    # Wait, if filled, we go to new_q. If not, we stay at q.
                    # Expected value = p_bid * (d + V[n+1, new_i_idx]) + (1 - p_bid) * V[n+1, i_idx]
                    val = p_bid * (d + V[n + 1, new_i_idx]) + (1 - p_bid) * V[n + 1, i_idx]
                    if val > best_val_bid:
                        best_val_bid = val
                        best_depth_bid = d
                opt_bid[n, i_idx] = best_depth_bid
                # Store the value for this side (we'll combine bid and ask values)
                # But the HJB is: max over bid, ask of [bid_value + ask_value - inv_cost]
                # Actually, bid and ask are chosen independently? No, they are chosen jointly.
                # But the fills are independent.
                # V[n, q] = max_{b,a} [ p_b(b)*(b + V[n+1, q-1]) + (1-p_b(b))*V[n+1, q] 
                #                 + p_a(a)*(a + V[n+1, q+1]) + (1-p_a(a))*V[n+1, q]
                #                 - inv_cost ]
                # This can be decomposed:
                # V[n, q] = max_b [ ... ] + max_a [ ... ] - inv_cost
                # Let's verify:
                # p_b(b)*(b + V[n+1, q-1]) + (1-p_b(b))*V[n+1, q] = V[n+1, q] + p_b(b)*(b + V[n+1, q-1] - V[n+1, q])
                # So max_b of this is V[n+1, q] + max_b [ p_b(b)*(b + V[n+1, q-1] - V[n+1, q]) ]
                # Similarly for ask.
                # So V[n, q] = 2*V[n+1, q] + max_b[...] + max_a[...] - inv_cost
                # This decomposition is valid because bid and ask choices are independent in the expectation.
            else:
                # No bid quote (or depth = inf, p_bid = 0)
                # Value contribution from bid side is V[n+1, i_idx]
                best_val_bid = V[n + 1, i_idx]
                opt_bid[n, i_idx] = np.inf

            if q > -Q:
                best_val_ask = -np.inf
                best_depth_ask = 1.0 / k
                for d in depth_grid:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                    new_q = q + 1
                    if new_q > Q:
                        continue
                    new_i_idx = int(new_q + Q)
                    val = p_ask * (d + V[n + 1, new_i_idx]) + (1 - p_ask) * V[n + 1, i_idx]
                    if val > best_val_ask:
                        best_val_ask = val
                        best_depth_ask = d
                opt_ask[n, i_idx] = best_depth_ask
            else:
                best_val_ask = V[n + 1, i_idx]
                opt_ask[n, i_idx] = np.inf

            # Combine
            # V[n, i_idx] = best_val_bid + best_val_ask - inv_cost
            # Wait, the decomposition was:
            # V[n, q] = V[n+1, q] + max_b[ p_b(b)*(b + V[n+1, q-1] - V[n+1, q]) ]
            #         + V[n+1, q] + max_a[ p_a(a)*(a + V[n+1, q+1] - V[n+1, q]) ]
            #         - inv_cost
            # = 2*V[n+1, q] + max_b[...] + max_a[...] - inv_cost
            # But my best_val_bid is already V[n+1, q] + max_b[...]
            # So V[n, i_idx] = best_val_bid + best_val_ask - V[n+1, i_idx] - inv_cost
            # Let's re-verify.
            # best_val_bid = p_b*(b + V[n+1, q-1]) + (1-p_b)*V[n+1, q]
            # best_val_ask = p_a*(a + V[n+1, q+1]) + (1-p_a)*V[n+1, q]
            # Sum = p_b*(b + V[n+1, q-1]) + (1-p_b)*V[n+1, q] + p_a*(a + V[n+1, q+1]) + (1-p_a)*V[n+1, q]
            # This is exactly the expected value before subtracting inv_cost.
            # So V[n, i_idx] = best_val_bid + best_val_ask - inv_cost
            V[n, i_idx] = best_val_bid + best_val_ask - inv_cost

    _OPT_TABLES[key] = (opt_bid, opt_ask, inv_range)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    opt_bid, opt_ask, inv_range = _get_optimal_table(params)

    n = int(t / (params["T"] / params["steps"]))
    if n >= params["steps"]:
        n = params["steps"] - 1

    # Map inventory to index
    inv_indices = inventory + params["Q"]
    # Clip to valid range
    inv_indices = np.clip(inv_indices, 0, len(inv_range) - 1)

    bid_depths = opt_bid[n, inv_indices]
    ask_depths = opt_ask[n, inv_indices]

    # Ensure depths are valid (>= 0, not NaN)
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0 / params["k"], bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0 / params["k"], ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
