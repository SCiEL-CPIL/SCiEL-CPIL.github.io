"""Market making with inventory risk – approximate optimal quoting via DP on a coarse
inventory/time grid, with analytic inventory-shift heuristic as fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache

_T0 = time.time()
_DEADLINE = 295.0  # safety margin before 300s kill

# Cache for solved policies: key -> (bid_table, ask_table) or None
_POLICY_CACHE = {}


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _heuristic_depths(inventory, params):
    """Fallback: linear inventory shift around 1/k, clipped."""
    k = params["k"]
    Q = params["Q"]
    gamma = params["gamma"]
    sigma = params["sigma"]
    T = params["T"]
    steps = params["steps"]
    A = params["A"]
    t = params.get("_t", 0.0)
    dt = T / steps
    n = int(round(t / dt)) if dt > 0 else 0
    remaining = max(steps - n, 1)

    # Simple linear shift: bid = 1/k + c*q, ask = 1/k - c*q
    # c chosen to balance inventory cost vs spread
    # gamma/2 * sigma^2 * q^2 * dt is the cost per step
    # We want to reduce inventory, so shift toward the side that brings q toward 0
    # If q > 0 (long), widen bid (lower bid), narrow ask
    # If q < 0 (short), narrow bid, widen ask
    # Shift magnitude proportional to q, scaled by risk aversion
    shift_scale = gamma * sigma**2 * dt * Q / (A * (1.0 / k))  # heuristic scale
    shift = shift_scale * inventory.astype(float)
    inv_k = 1.0 / k
    bid = inv_k + shift
    ask = inv_k - shift
    max_d = 3.0 / inv_k  # 3/k
    bid = np.clip(bid, 0.0, max_d)
    ask = np.clip(ask, 0.0, max_d)
    # At Q, no bid; at -Q, no ask
    q_arr = inventory.astype(float)
    bid = np.where(q_arr >= Q, np.inf, bid)
    ask = np.where(q_arr <= -Q, np.inf, ask)
    return bid, ask


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve a coarse DP over inventory in [-Q, Q] and time steps 0..steps-1.
    We discretize inventory on the integer grid [-Q, Q] (61 values for Q=30).
    For each step and inventory, we choose bid_depth and ask_depth from a small
    set of candidate depths to maximize expected reward.
    
    Returns (bid_table, ask_table) where tables are indexed by (inventory_idx, step).
    inventory_idx = inventory + Q, in [0, 2Q].
    """
    if _remaining_budget() < 90.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_d = 3.0 * inv_k

    # Candidate depths: use a small grid
    n_depths = 16
    depth_grid = np.linspace(0.0, max_d, n_depths)
    # Add 1/k explicitly
    depth_grid = np.unique(np.concatenate([depth_grid, [inv_k]]))
    n_depths = len(depth_grid)

    # Transition probabilities
    # P_bid_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    # P_ask_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    p_bid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    p_ask = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

    # V[n, q_idx] = max expected reward from step n onward, starting with inventory q
    # q_idx = q + Q, q in [-Q, Q]
    nq = 2 * Q + 1
    V = np.zeros((steps + 1, nq))

    # bid_table[i, n, q_idx] = optimal bid depth index
    # ask_table[i, n, q_idx] = optimal ask depth index
    bid_table = np.zeros((steps, nq), dtype=np.float64)
    ask_table = np.zeros((steps, nq), dtype=np.float64)

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if _remaining_budget() < 60.0:
            return None

        # Inventory cost at this step: gamma/2 * sigma^2 * q^2 * dt
        q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
        inv_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt  # shape (nq,)

        # For each inventory level q, choose (i_bid, i_ask) to maximize:
        #   depth_bid * p_bid[i_bid] * 1[q < Q]
        # + depth_ask * p_ask[i_ask] * 1[q > -Q]
        # - inv_cost[q]
        # + E[V[n+1, q']]
        # where q' depends on fills

        # We'll vectorize over q and depth pairs
        # For each q, the expected next value:
        #   (1-pb)*(1-pa)*V[n+1, q]
        # + pb*(1-pa)*V[n+1, q-1]  (bid fill, q decreases by 1)
        # + (1-pb)*pa*V[n+1, q+1]  (ask fill, q increases by 1)
        # + pb*pa*V[n+1, q]       (both fill, net 0)
        # but with boundary: no bid fill at q=Q, no ask fill at q=-Q

        # Precompute V_next = V[n+1]
        Vn1 = V[n + 1]

        # We need to handle boundaries:
        # V_next[q-1] is valid for q > -Q, else 0 (but q=-Q has no ask fill anyway)
        # V_next[q+1] is valid for q < Q, else 0

        # Let's compute the full reward matrix R[q_idx, i_bid, i_ask]
        # But that's nq * n_depths * n_depths which is 61*17*17 ~ 17k, fine.

        # Actually, let's loop over q for clarity and safety
        for qi in range(nq):
            q = qi - Q  # actual inventory

            # Inventory cost
            cost = inv_cost[qi]

            # Determine valid fills
            can_bid = (q < Q)
            can_ask = (q > -Q)

            # Build reward matrix for this q
            # R[i_bid, i_ask] = reward from fills + expected V_next - cost
            # Fill rewards:
            #   bid fill: +depth_grid[i_bid], inventory -> q-1
            #   ask fill: +depth_grid[i_ask], inventory -> q+1

            pb = p_bid  # (n_depths,)
            pa = p_ask  # (n_depths,)

            # Expected next value for each (i_bid, i_ask)
            # no fill: prob (1-pb)*(1-pa), q stays
            # bid only: pb*(1-pa), q -> q-1
            # ask only: (1-pb)*pa, q -> q+1
            # both: pb*pa, q stays

            # V_next indices
            q_idx_next_same = qi
            q_idx_next_bid = qi - 1 if q > -Q else None  # bid fill reduces q
            q_idx_next_ask = qi + 1 if q < Q else None   # ask fill increases q

            # Handle boundaries
            if q >= Q:
                # No bid fill possible
                # Only ask fill possible (if q > -Q, which is true since q>=Q>=0)
                # R[i_bid, i_ask] = depth_ask*pa[i_ask] + (1-pa)*Vn1[qi] + pa*Vn1[qi+1 if qi+1<nq else qi] - cost
                # But bid depth doesn't matter for reward, only for the fact that we can't fill
                # We still need to choose a bid depth, but it doesn't affect reward
                # Set bid to 0 (or any value, doesn't matter)
                # Actually, we should set bid to np.inf to indicate no quote? 
                # But the problem says "no bid fill at q=Q", not "no bid quote".
                # Let's just set bid depth to 0.
                ask_valid = can_ask
                if ask_valid:
                    q_ask_next = min(qi + 1, nq - 1)
                    exp_next = (1 - pa) * Vn1[qi] + pa * Vn1[q_ask_next]
                    reward = depth_grid[np.newaxis, :] * pa[np.newaxis, :] + exp_next[np.newaxis, :] - cost
                    # Maximize over i_ask; i_bid is free
                    best_ia = np.argmax(reward)
                    best_val = reward[best_ia]
                    # Set bid to 0
                    bid_table[n, qi] = 0.0
                    ask_table[n, qi] = depth_grid[best_ia]
                    V[n, qi] = best_val
                else:
                    bid_table[n, qi] = 0.0
                    ask_table[n, qi] = 0.0
                    V[n, qi] = Vn1[qi] - cost
                continue

            if q <= -Q:
                # No ask fill possible
                bid_valid = can_bid
                if bid_valid:
                    q_bid_next = max(qi - 1, 0)
                    exp_next = (1 - pb) * Vn1[qi] + pb * Vn1[q_bid_next]
                    reward = depth_grid[np.newaxis, :] * pb[np.newaxis, :] + exp_next[np.newaxis, :] - cost
                    best_ib = np.argmax(reward)
                    best_val = reward[best_ib]
                    bid_table[n, qi] = depth_grid[best_ib]
                    ask_table[n, qi] = 0.0
                    V[n, qi] = best_val
                else:
                    bid_table[n, qi] = 0.0
                    ask_table[n, qi] = 0.0
                    V[n, qi] = Vn1[qi] - cost
                continue

            # Interior: both fills possible
            q_bid_next = qi - 1
            q_ask_next = qi + 1

            # exp_next[i_bid, i_ask] = (1-pb[i_b])(1-pa[i_a])*Vn1[qi]
            #                      + pb[i_b](1-pa[i_a])*Vn1[qi-1]
            #                      + (1-pb[i_b])pa[i_a]*Vn1[qi+1]
            #                      + pb[i_b]pa[i_a]*Vn1[qi]
            # = Vn1[qi] + pb[i_b](1-pa[i_a])(Vn1[qi-1]-Vn1[qi]) + (1-pb[i_b])pa[i_a](Vn1[qi+1]-Vn1[qi])
            # Let d_bid = Vn1[qi-1] - Vn1[qi], d_ask = Vn1[qi+1] - Vn1[qi]
            d_bid = Vn1[q_bid_next] - Vn1[qi]
            d_ask = Vn1[q_ask_next] - Vn1[qi]

            # exp_next[i_b, i_a] = Vn1[qi] + pb[i_b](1-pa[i_a])*d_bid + (1-pb[i_b])pa[i_a]*d_ask
            # Fill reward: depth_grid[i_b]*pb[i_b] + depth_grid[i_a]*pa[i_a]
            # Total = depth_grid[i_b]*pb[i_b] + depth_grid[i_a]*pa[i_a] + Vn1[qi] + pb[i_b](1-pa[i_a])*d_bid + (1-pb[i_b])pa[i_a]*d_ask - cost

            # Vectorize over i_b and i_a
            # Let's build the matrix
            # R[i_b, i_a] = dg[i_b]*pb[i_b] + dg[i_a]*pa[i_a] + Vn1[qi] + pb[i_b]*(1-pa[i_a])*d_bid + (1-pb[i_b])*pa[i_a]*d_ask - cost
            dg = depth_grid
            # Terms:
            # T1[i_b, i_a] = dg[i_b]*pb[i_b]
            T1 = (dg * pb)[:, np.newaxis]
            # T2[i_b, i_a] = dg[i_a]*pa[i_a]
            T2 = (dg * pa)[np.newaxis, :]
            # T3[i_b, i_a] = Vn1[qi]
            T3 = np.full((n_depths, n_depths), Vn1[qi])
            # T4[i_b, i_a] = pb[i_b]*(1-pa[i_a])*d_bid
            T4 = (pb[:, np.newaxis] * (1 - pa)[np.newaxis, :]) * d_bid
            # T5[i_b, i_a] = (1-pb[i_b])*pa[i_a]*d_ask
            T5 = ((1 - pb)[:, np.newaxis] * pa[np.newaxis, :]) * d_ask
            # T6 = -cost
            T6 = -cost

            R = T1 + T2 + T3 + T4 + T5 + T6

            # Find argmax
            idx = np.argmax(R)
            ib, ia = np.unravel_index(idx, R.shape)
            V[n, qi] = R[ib, ia]
            bid_table[n, qi] = depth_grid[ib]
            ask_table[n, qi] = depth_grid[ia]

    return bid_table, ask_table


def quote(t, inventory, mid, params):
    """
    Quote using cached DP policy or fallback heuristic.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    key = (sigma, A, k, gamma, T, steps, Q)

    if key not in _POLICY_CACHE:
        result = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        _POLICY_CACHE[key] = result

    result = _POLICY_CACHE[key]

    if result is None:
        # Fallback to heuristic
        params_copy = dict(params)
        params_copy["_t"] = t
        return _heuristic_depths(inventory, params_copy)

    bid_table, ask_table = result

    dt = T / steps
    n = int(round(t / dt))
    if n < 0:
        n = 0
    if n >= steps:
        n = steps - 1

    inv_clipped = np.clip(inventory, -Q, Q)
    q_idx = (inv_clipped + Q).astype(np.int64)

    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]

    max_d = 3.0 / k
    bid_depths = np.clip(bid_depths, 0.0, max_d)
    ask_depths = np.clip(ask_depths, 0.0, max_d)

    # Boundary conditions: no bid at Q, no ask at -Q
    q_arr = inventory.astype(float)
    bid_depths = np.where(q_arr >= Q, np.inf, bid_depths)
    ask_depths = np.where(q_arr <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
