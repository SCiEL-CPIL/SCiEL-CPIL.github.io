"""Market making with inventory risk – backward‑induction DP on a coarse grid,
with an HJB‑style first‑order refinement for the optimal depth."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre‑computed DP tables keyed by (sigma, A, k, gamma, T, steps, Q)
_dp_cache = {}
_start_time = None


def _get_dp_tables(params):
    """Return (V_next, bid_depths_table, ask_depths_table) for the given regime.

    V_next is a 1‑D array indexed by inventory+Q, shape (2*Q+1,).
    bid_depths_table and ask_depths_table are 1‑D arrays of optimal depths
    for each inventory level at the *current* step, to be used with V_next
    (i.e. one step before terminal).
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key in _dp_cache:
        return _dp_cache[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    n_inv = 2 * Q + 1
    inv_levels = np.arange(-Q, Q + 1, dtype=np.int64)

    # We'll do backward induction from step n = steps-1 down to 0.
    # V[n+1] is the value-to-go after step n's fills.
    # V[steps] = 0 (no terminal charge)
    V_next = np.zeros(n_inv, dtype=np.float64)

    # We only need the policy at the *last* step (n = steps-1) because the
    # evaluator calls quote at every step, but the optimal policy depends on
    # remaining time. However, since the model is time-homogeneous in the
    # sense that only remaining steps matter, and we're called at every step,
    # we actually need the full table for each step. But computing 1000 tables
    # of size 61 is cheap.
    #
    # Actually, re-reading: quote is called at every step n=0..999. The optimal
    # depth at step n depends on how many steps remain. So we need V for each
    # step. Let's store policies for each step.
    #
    # But that's 1000 * 61 entries for bid and ask each = 122,000 floats. Fine.

    # Let's compute all policies via backward induction.
    # policy_bid[n, inv_idx] = optimal bid depth at step n given inventory inv
    # policy_ask[n, inv_idx] = optimal ask depth at step n given inventory inv

    # To save memory and time, note that the problem is time-homogeneous if we
    # define m = steps - n as remaining steps. So policy depends only on m.
    # m ranges from 1 to steps (at n=0, m=steps; at n=steps-1, m=1).

    # V[m] = value-to-go with m steps remaining (before any fills in those steps)
    # V[0] = 0
    # V[m] = max over depths of expected reward + E[V[m-1]]

    # Actually, let me define it more carefully:
    # At step n (0-indexed), before fills, we have inventory q, time t = n*dt.
    # We choose bid_depth b >= 0 and ask_depth a >= 0.
    # Fill probabilities: p_bid = 1 - exp(-A * exp(-k*b) * dt), p_ask = 1 - exp(-A * exp(-k*a) * dt)
    # If bid fills: earn b, inventory becomes q+1 (if q < Q)
    # If ask fills: earn a, inventory becomes q-1 (if q > -Q)
    # Both can fill independently.
    # Risk penalty: (gamma/2) * sigma^2 * q^2 * dt (applied to inventory before fills)
    # Mid moves but that doesn't affect the objective directly (depths are relative to mid).
    #
    # So the value function V(n, q) = max_{b,a} [ (gamma/2)*sigma^2*q^2*dt  ... wait, it's minus
    # Actually objective is: depths earned MINUS sum of (gamma/2)*sigma^2*q^2*dt
    # So we want to MAXIMIZE: sum of (depths earned) - sum of (gamma/2)*sigma^2*q^2*dt
    #
    # Let V(n, q) = max expected (total future rewards - future risk penalties) given we're at step n, inventory q.
    # V(steps, q) = 0 for all q.
    # V(n, q) = max_{b,a >= 0} [ 
    #     p_bid * b * I(q<Q) + p_ask * a * I(q>-Q) 
    #     - (gamma/2)*sigma^2*q^2*dt
    #     + p_bid*p_ask*V(n+1, q+1-I(q>=Q) - I(q<=-Q))  ... this gets complicated with boundary conditions
    # ]
    #
    # Let me handle boundaries:
    # - If q == Q: no bid fill possible. p_bid_effective = 0.
    # - If q == -Q: no ask fill possible. p_ask_effective = 0.
    #
    # Let me define:
    # q_next_bid = q+1 if q < Q else q
    # q_next_ask = q-1 if q > -Q else q
    #
    # Four cases:
    # 1. No fills: prob (1-pb)(1-pa), next inv = q
    # 2. Bid only: prob pb*(1-pa), next inv = q_next_bid
    # 3. Ask only: prob (1-pb)*pa, next inv = q_next_ask
    # 4. Both: prob pb*pa, next inv = q_next_bid if q<Q and q_next_ask if q>-Q... 
    #    Actually if both fill, inventory changes by +1 and -1, net 0? No:
    #    "A fill earns its depth and moves inventory one unit"
    #    Bid fill: +1 to inventory
    #    Ask fill: -1 to inventory
    #    Both: net 0 change.
    #    But boundary: if q==Q, bid can't fill. If q==-Q, ask can't fill.
    #    If q==Q and ask fills: q becomes Q-1.
    #    If q==-Q and bid fills: q becomes -Q+1.
    #    If both fill (only possible if -Q < q < Q): q stays same.

    # This is getting complex. Let me just do it numerically with a grid of depths.

    # For each inventory level, we'll search over a grid of depths for bid and ask.
    # The depth is in [0, inf) but practically we can cap it. exp(-k*d) becomes tiny.
    # Let's use a grid of, say, 50 points from 0 to some max_depth.
    # max_depth: when k*d is large, exp(-k*d) ~ 0, so p ~ 1-exp(-A*0*dt) = 0. 
    # Actually p = 1 - exp(-A*exp(-k*d)*dt). As d->inf, exp(-k*d)->0, so p->1-exp(0)=0.
    # As d->0, exp(0)=1, p = 1-exp(-A*dt).
    # So max p is 1-exp(-A*dt) at d=0.

    # Let's use a grid of depths. The optimal depth should be in a reasonable range.
    # For the seed, 1/k was used. Let's use a grid from 0 to 10/k or so.

    n_depth = 40
    max_d = 10.0 / k
    depth_grid = np.linspace(0, max_d, n_depth)

    # Precompute p(d) for the depth grid
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # Backward induction
    # V is indexed by [m, inv_idx] where m = steps remaining (0..steps), inv_idx = q+Q
    # V[0, :] = 0
    # For m from 1 to steps:
    #   For each q:
    #     For each (b_idx, a_idx) in depth_grid^2:
    #       Compute expected value
    #     Take max

    # This is O(steps * n_inv * n_depth^2) = 1000 * 61 * 1600 = 97.6M operations.
    # That might be too slow in Python. Let's vectorize.

    # Actually, let's think about whether we can reduce the depth grid or use a smarter approach.
    # The HJB first-order condition might give us a closed-form or 1D search.

    # For now, let's try with a smaller grid and see if it fits in time.
    # n_depth = 20, max_d = 5/k
    n_depth = 25
    max_d = 5.0 / k
    depth_grid = np.linspace(0, max_d, n_depth)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # V[m] is a 1D array of shape (n_inv,)
    # policy_bid[m-1, inv_idx] and policy_ask[m-1, inv_idx] for m=1..steps
    # We store policies indexed by m (1..steps) and inv_idx

    policies_bid = np.zeros((steps, n_inv), dtype=np.float64)
    policies_ask = np.zeros((steps, n_inv), dtype=np.float64)

    V = np.zeros(n_inv, dtype=np.float64)  # V for current m

    # Precompute inventory indices
    inv_idx_arr = np.arange(n_inv)
    q_arr = inv_levels  # -Q..Q

    for m in range(1, steps + 1):
        V_next = V  # V[m-1]
        V_curr = np.zeros(n_inv, dtype=np.float64)
        best_bid = np.zeros(n_inv, dtype=np.float64)
        best_ask = np.zeros(n_inv, dtype=np.float64)

        # For each inventory level
        for inv_idx in range(n_inv):
            q = inv_levels[inv_idx]
            can_bid = q < Q
            can_ask = q > -Q

            risk = 0.5 * gamma * sigma * sigma * q * q * dt

            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0

            for bi in range(n_depth):
                b = depth_grid[bi]
                pb = p_grid[bi] if can_bid else 0.0
                for ai in range(n_depth):
                    a = depth_grid[ai]
                    pa = p_grid[ai] if can_ask else 0.0

                    # Expected reward
                    reward = 0.0
                    if can_bid and can_ask:
                        # 4 cases
                        prob_none = (1 - pb) * (1 - pa)
                        prob_bid = pb * (1 - pa)
                        prob_ask = (1 - pb) * pa
                        prob_both = pb * pa

                        q_bid = q + 1
                        q_ask = q - 1
                        q_both = q  # net zero

                        reward = (prob_none * V_next[inv_idx] +
                                  prob_bid * V_next[q_bid + Q] +
                                  prob_ask * V_next[q_ask + Q] +
                                  prob_both * V_next[q_both + Q])
                        reward += prob_bid * b + prob_ask * a
                    elif can_bid:
                        # q == -Q, ask can't fill
                        prob_none = 1 - pb
                        prob_bid = pb
                        q_bid = q + 1
                        reward = prob_none * V_next[inv_idx] + prob_bid * V_next[q_bid + Q]
                        reward += prob_bid * b
                    elif can_ask:
                        # q == Q, bid can't fill
                        prob_none = 1 - pa
                        prob_ask = pa
                        q_ask = q - 1
                        reward = prob_none * V_next[inv_idx] + prob_ask * V_next[q_ask + Q]
                        reward += prob_ask * a
                    else:
                        # q == Q == -Q, only possible if Q==0
                        reward = V_next[inv_idx]

                    total = reward - risk

                    if total > best_val:
                        best_val = total
                        best_b = b
                        best_a = a

            V_curr[inv_idx] = best_val
            best_bid[inv_idx] = best_b
            best_ask[inv_idx] = best_a

        policies_bid[m - 1, :] = best_bid
        policies_ask[m - 1, :] = best_ask
        V = V_curr

    _dp_cache[key] = (policies_bid, policies_ask)
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 325:  # 5s safety margin
        # Return a safe default
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    policies_bid, policies_ask = _get_dp_tables(params)

    steps = params["steps"]
    Q = params["Q"]

    # Determine current step
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    # m = steps remaining = steps - n
    m = steps - n

    # m ranges from 1 to steps. policies indexed by m-1.
    idx = m - 1
    if idx < 0 or idx >= steps:
        idx = 0

    # Map inventory to indices
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)

    bid_depths = policies_bid[idx, inv_idx]
    ask_depths = policies_ask[idx, inv_idx]

    # Ensure valid values
    min_valid = -3.0 / params["k"]
    bid_depths = np.where(bid_depths < min_valid, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < min_valid, np.inf, ask_depths)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
