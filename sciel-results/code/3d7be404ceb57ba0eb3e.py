"""
Optimal market making with inventory risk via exact Dynamic Programming.

Strategy:
  - Pre-compute the exact optimal policy using the Bellman equation over the
    discrete state space (step n, inventory q).
  - The state space is (steps=1000) x (2*Q+1 = 61 inventory levels).
  - For each state, we choose bid_depth and ask_depth to maximize the expected
    continuation value. Since the depth affects both the immediate reward (depth
    earned on fill) and the fill probability, we need to optimize over depth.
  - The fill probability for bid is p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt).
  - The immediate reward from a bid fill is bid_depth, and inventory increases by 1.
  - The immediate reward from an ask fill is ask_depth, and inventory decreases by 1.
  - The inventory cost at step n is (gamma/2) * sigma^2 * q^2 * dt, charged on
    inventory BEFORE fills.

  Since depth is continuous, we use a fine grid search over depth values for each
  state. To keep computation tractable within the time budget, we:
    1. Cache the policy per regime.
    2. Use a reasonable depth grid (e.g., 200 points in [0, max_depth]).
    3. Vectorize over inventory levels for each depth candidate.

  The Bellman equation:
    V(n, q) = - (gamma/2) * sigma^2 * q^2 * dt
              + max_{b, a} {
                  p_b * (b + V(n+1, q+1)) * I[q < Q]
                  + p_a * (a + V(n+1, q-1)) * I[q > -Q]
                  + (1 - p_b) * (1 - p_a) * V(n+1, q)
              }
    where p_b = 1 - exp(-A * exp(-k*b) * dt), p_a = 1 - exp(-A * exp(-k*a) * dt).

  Note: bid and ask fills are independent, so the joint probability of both filling
  is p_b * p_a. We must account for all four outcomes:
    - Only bid fills: p_b * (1 - p_a) -> reward b, inventory q+1
    - Only ask fills: (1 - p_b) * p_a -> reward a, inventory q-1
    - Both fill: p_b * p_a -> reward b + a, inventory q (bid +1, ask -1, net 0)
    - Neither: (1 - p_b) * (1 - p_a) -> reward 0, inventory q

  So:
    V(n, q) = - (gamma/2) * sigma^2 * q^2 * dt
              + max_{b, a} {
                  p_b*(1-p_a) * (b + V(n+1, q+1)) * I[q<Q]
                  + (1-p_b)*p_a * (a + V(n+1, q-1)) * I[q>-Q]
                  + p_b*p_a * (b + a + V(n+1, q))
                  + (1-p_b)*(1-p_a) * V(n+1, q)
              }
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_start_time = time.time()
TIME_BUDGET = 310  # safety margin before 330s kill

def _compute_optimal_policy(params):
    """
    Compute optimal bid/ask depths for all (step, inventory) states via DP.
    Returns (bid_depths, ask_depths) each of shape (steps, 2*Q+1).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_inv = 2 * Q + 1
    inv_idx = np.arange(-Q, Q + 1)  # inventory values

    # Value function: V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    bid_depth = np.zeros((steps, n_inv), dtype=np.float64)
    ask_depth = np.zeros((steps, n_inv), dtype=np.float64)

    # Depth grid for optimization
    # Reasonable range: from 0 to about 5/k
    max_depth = 5.0 / k
    n_depth = 150
    depth_grid = np.linspace(0.0, max_depth, n_depth)

    # Precompute exp(-k * depth) for the grid
    exp_neg_k_depth = np.exp(-k * depth_grid)
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_grid = 1.0 - np.exp(-A * exp_neg_k_depth * dt)

    # Inventory cost coefficient
    inv_cost_coeff = 0.5 * gamma * sigma * sigma * dt

    # DP backwards
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > TIME_BUDGET - 10:
            # Fallback: use 1/k for remaining
            for nn in range(n, steps):
                bid_depth[nn, :] = 1.0 / k
                ask_depth[nn, :] = 1.0 / k
            return bid_depth, ask_depth

        V_next = V[n + 1]  # shape (n_inv,)

        # For each inventory level, optimize over (b, a)
        # To speed up, we can vectorize over inventory for each pair of depths,
        # but that's n_depth^2 * n_inv per step = 150^2 * 61 * 1000 = ~1.36e9 ops
        # That's too slow. Instead, we optimize b and a separately if possible,
        # or use a coarser grid.

        # Actually, the objective is separable in b and a in the sense that:
        # Let's define for a given b:
        #   term_b_only = p_b * (1 - p_a) * (b + V_next[q+1])  [if q < Q]
        #   term_both = p_b * p_a * (b + a + V_next[q])
        #   term_neither = (1-p_b)*(1-p_a) * V_next[q]
        #   term_a_only = (1-p_b) * p_a * (a + V_next[q-1])  [if q > -Q]
        #
        # The total = term_b_only + term_a_only + term_both + term_neither
        #
        # This is not separable because p_b and p_a appear in cross terms.
        # However, we can try a nested optimization: for each b, find optimal a,
        # then find optimal b. Or use a coarser grid.

        # Let's use a coarser grid: 80 points
        n_depth_coarse = 80
        depth_grid_c = np.linspace(0.0, max_depth, n_depth_coarse)
        exp_neg_k_depth_c = np.exp(-k * depth_grid_c)
        p_grid_c = 1.0 - np.exp(-A * exp_neg_k_depth_c * dt)

        # For each inventory level, do a 2D grid search
        # But to vectorize, we can compute the value for all (b_idx, a_idx) pairs
        # for all inventory levels simultaneously.

        # V_next has shape (n_inv,)
        # V_next_shifted_up: V_next[q+1] for q in [-Q, Q-1], inf for q=Q
        # V_next_shifted_down: V_next[q-1] for q in [-Q+1, Q], inf for q=-Q

        V_next_up = np.full(n_inv, -np.inf)  # V(q+1)
        V_next_down = np.full(n_inv, -np.inf)  # V(q-1)

        # q from -Q to Q-1: index from 0 to n_inv-2
        V_next_up[:n_inv-1] = V_next[1:]  # V(q+1) for q = -Q..Q-1
        # q from -Q+1 to Q: index from 1 to n_inv-1
        V_next_down[1:] = V_next[:-1]  # V(q-1) for q = -Q+1..Q

        # Inventory cost
        inv_cost = inv_cost_coeff * inv_idx ** 2  # shape (n_inv,)

        # Now for each (b_idx, a_idx), compute the expected value for all q
        # This is a 2D grid of size n_depth_c^2 = 6400
        # For each pair, we compute a vector of length n_inv
        # Total: 6400 * 61 = ~390k ops per step, times 1000 steps = 390M ops
        # Still might be slow in pure Python loop over depth pairs.

        # Let's vectorize over depth pairs:
        # p_b has shape (n_depth_c,), p_a has shape (n_depth_c,)
        # We want to compute for all (i, j):
        #   p_b_i = p_grid_c[i]
        #   p_a_j = p_grid_c[j]
        #   b_i = depth_grid_c[i]
        #   a_j = depth_grid_c[j]
        #
        # For each q:
        #   val = inv_cost[q] + ... wait, V(n,q) = -inv_cost[q] + max(...)
        #   Actually V(n,q) = -inv_cost[q] + expected_future
        #   expected_future = max over (b,a) of:
        #     p_b*(1-p_a)*(b + V_up[q]) * I[q<Q]
        #     + (1-p_b)*p_a*(a + V_down[q]) * I[q>-Q]
        #     + p_b*p_a*(b+a+V_next[q])
        #     + (1-p_b)*(1-p_a)*V_next[q]

        # Let's compute this as a matrix operation.
        # For a fixed (i, j):
        #   term1 = p_grid_c[i] * (1 - p_grid_c[j]) * (depth_grid_c[i] + V_next_up) * (inv_idx < Q)
        #   term2 = (1 - p_grid_c[i]) * p_grid_c[j] * (depth_grid_c[j] + V_next_down) * (inv_idx > -Q)
        #   term3 = p_grid_c[i] * p_grid_c[j] * (depth_grid_c[i] + depth_grid_c[j] + V_next)
        #   term4 = (1 - p_grid_c[i]) * (1 - p_grid_c[j]) * V_next
        #   total = term1 + term2 + term3 + term4

        # We can vectorize over i and j by creating outer products.
        # But n_depth_c^2 = 6400, and for each we have a vector of length n_inv=61.
        # So we'd have a 3D array of shape (n_depth_c, n_depth_c, n_inv) = 6400*61 = 390k floats.
        # That's manageable.

        pb = p_grid_c  # (n_depth_c,)
        pa = p_grid_c  # (n_depth_c,)
        b_grid = depth_grid_c  # (n_depth_c,)
        a_grid = depth_grid_c  # (n_depth_c,)

        # Create outer arrays
        pb_i = pb[:, None]  # (n_depth_c, 1)
        pa_j = pa[None, :]  # (1, n_depth_c)
        b_i = b_grid[:, None]  # (n_depth_c, 1)
        a_j = a_grid[None, :]  # (1, n_depth_c)

        # term1: (n_depth_c, n_depth_c, n_inv)
        mask_up = (inv_idx < Q).astype(np.float64)  # (n_inv,)
        term1 = pb_i * (1 - pa_j) * (b_i + V_next_up[None, None, :]) * mask_up[None, None, :]
        # term1 shape: (n_depth_c, n_depth_c, n_inv)

        mask_down = (inv_idx > -Q).astype(np.float64)
        term2 = (1 - pb_i) * pa_j * (a_j + V_next_down[None, None, :]) * mask_down[None, None, :]

        term3 = pb_i * pa_j * (b_i + a_j + V_next[None, None, :])

        term4 = (1 - pb_i) * (1 - pa_j) * V_next[None, None, :]

        total = term1 + term2 + term3 + term4  # (n_depth_c, n_depth_c, n_inv)

        # Max over (i, j) for each q
        # total has shape (n_depth_c, n_depth_c, n_inv)
        # We want max over first two axes for each element in the third axis
        max_val, max_idx = np.max(total, axis=(0, 1))  # max_val shape (n_inv,)
        # max_idx is a flat index into (n_depth_c, n_depth_c)
        i_idx = max_idx // n_depth_coarse
        j_idx = max_idx % n_depth_coarse

        # V(n, q) = -inv_cost[q] + max_val[q]
        V[n] = -inv_cost + max_val
        bid_depth[n] = b_grid[i_idx]
        ask_depth[n] = a_grid[j_idx]

    return bid_depth, ask_depth


def _get_policy(params):
    """Get or compute the optimal policy for the given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_policy(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for the given state.
    """
    Q = params['Q']
    steps = params['steps']
    n = int(round(t / (params['T'] / steps)))
    n = max(0, min(n, steps - 1))

    bid_depth_table, ask_depth_table = _get_policy(params)

    # inventory is an array of shape (4000,) with values in [-Q, Q]
    # Convert to index: idx = inventory + Q
    inv_idx = np.clip(inventory, -Q, Q) + Q
    inv_idx = inv_idx.astype(np.int64)

    bid = bid_depth_table[n, inv_idx]
    ask = ask_depth_table[n, inv_idx]

    return bid, ask
# EVOLVE-BLOCK-END
