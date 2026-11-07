"""Market-making with inventory risk: DP over inventory grid with time-varying policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache DP value tables per parameter regime.
_V_TABLES = {}

# Global deadline guard (300 s hard limit).
_T0 = time.time()
_DEADLINE = 295.0  # seconds of wall clock budget for the whole run.


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete DP for the optimal policy.
    Returns a dict with:
      - 'bid_shifts': array of shape (steps, nq) where bid_shifts[n, i] is the shift
        added to the base bid depth at step n for inventory index i.
      - 'ask_shifts': array of shape (steps, nq) similarly for ask.
      - 'base_bid': scalar base bid depth (1/k)
      - 'base_ask': scalar base ask depth (1/k)
      - 'qs': inventory grid
      - 'q_step': grid spacing
    """
    if _remaining_budget() < 30.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    q_step = 2.0  # grid spacing for inventory
    q_min = float(-Q)
    q_max = float(Q)
    nq = int(round((q_max - q_min) / q_step)) + 1
    qs = q_min + q_step * np.arange(nq)  # (nq,)

    # V for n_remaining = 0 (terminal)
    V = np.zeros((nq,))

    # We'll store the optimal bid/ask shifts for each step and inventory level.
    # bid_shifts[n, i] is the shift from base at step n (0-indexed, n=0 is first step)
    # for inventory index i.
    bid_shifts = np.zeros((steps, nq))
    ask_shifts = np.zeros((steps, nq))

    # Precompute qs array
    q_arr = qs

    for n in range(steps, 0, -1):
        # n is the number of steps remaining (including current)
        # V is V(q, n-1) i.e. value with n-1 steps remaining
        Vprev = V.copy()

        # For each possible inventory q, we want to find the optimal bid_depth and ask_depth.
        # The optimal depth is a trade-off between immediate reward (depth) and continuation value.
        # We'll do a local search over depth values.

        # Base depths: bid_base = 1/k, ask_base = 1/k
        # We consider deviations from base. Let's search over a grid of shifts.
        # The shift is added to base: bid = base + shift_bid, ask = base + shift_ask
        # Constraints: bid >= 0, ask >= 0, bid <= 3/k, ask <= 3/k

        # For efficiency, we'll search over a coarse grid of shifts.
        # The shift range: bid in [0, 3/k] => shift_bid in [-1/k, 2/k]
        # ask in [0, 3/k] => shift_ask in [-1/k, 2/k]

        # Let's use a grid of shifts. Number of points: 20 per dimension.
        n_shifts = 20
        shift_range = np.linspace(-inv_k, 2.0 * inv_k, n_shifts)
        # shift_grid: (n_shifts, n_shifts) for (shift_bid, shift_ask)
        shift_bid_grid, shift_ask_grid = np.meshgrid(shift_range, shift_range, indexing='ij')
        shift_bid_flat = shift_bid_grid.ravel()
        shift_ask_flat = shift_ask_grid.ravel()

        # Compute bid and ask depths for all combinations
        bid_depths = inv_k + shift_bid_flat  # (n_shifts^2,)
        ask_depths = inv_k + shift_ask_flat  # (n_shifts^2,)

        # Clip to valid range
        bid_depths = np.clip(bid_depths, 0.0, 3.0 * inv_k)
        ask_depths = np.clip(ask_depths, 0.0, 3.0 * inv_k)

        # Probabilities
        pb_raw = 1.0 - np.exp(-A * np.exp(-k * bid_depths) * dt)  # (n_shifts^2,)
        pa_raw = 1.0 - np.exp(-A * np.exp(-k * ask_depths) * dt)  # (n_shifts^2,)

        # Reward: bid + ask - risk charge
        # Risk charge depends on q, so we compute it per q.

        # For each q index, compute the optimal shift.
        # Vprev is (nq,)
        # For each q_idx, we need V(q-1) and V(q+1) from Vprev.

        q_idx = np.arange(nq)
        lo = np.clip(q_idx - 1, 0, nq - 1)
        hi = np.clip(q_idx + 1, 0, nq - 1)

        # V(q-1) and V(q+1) via linear interpolation (q_step=2, so q±1 is halfway)
        Vq_minus = 0.5 * Vprev + 0.5 * Vprev[lo]  # (nq,)
        Vq_plus = 0.5 * Vprev + 0.5 * Vprev[hi]   # (nq,)

        # For each q, compute the total value for each shift combination.
        # Value = reward + continuation
        # reward = bid + ask - 0.5 * gamma * sigma^2 * q^2 * dt
        # continuation = pb * V(q-1) + pa * V(q+1) + (1-pb-pa) * V(q)

        # We'll vectorize over q and shifts.
        # q_arr: (nq,)
        # bid_depths, ask_depths, pb_raw, pa_raw: (n_shifts^2,)

        # Risk charge: (nq,)
        risk_charge = 0.5 * gamma * sigma * sigma * q_arr ** 2 * dt

        # For each q, compute value for each shift combo.
        # value[q, s] = (bid_depths[s] + ask_depths[s]) - risk_charge[q]
        #               + pb_raw[s] * Vq_minus[q] + pa_raw[s] * Vq_plus[q]
        #               + (1 - pb_raw[s] - pa_raw[s]) * Vprev[q]

        # This is a (nq, n_shifts^2) matrix. nq=31, n_shifts^2=400. That's 12400 elements. Fine.

        # Compute the value matrix
        # term1: (nq, 1) + (1, n_shifts^2) -> (nq, n_shifts^2)
        term1 = (bid_depths[np.newaxis, :] + ask_depths[np.newaxis, :]) - risk_charge[:, np.newaxis]
        term2 = pb_raw[np.newaxis, :] * Vq_minus[:, np.newaxis]
        term3 = pa_raw[np.newaxis, :] * Vq_plus[:, np.newaxis]
        term4 = (1.0 - pb_raw[np.newaxis, :] - pa_raw[np.newaxis, :]) * Vprev[:, np.newaxis]

        value_matrix = term1 + term2 + term3 + term4  # (nq, n_shifts^2)

        # Find the best shift index for each q
        best_idx = np.argmax(value_matrix, axis=1)  # (nq,)

        # Extract the optimal shifts
        opt_shift_bid = shift_bid_flat[best_idx]  # (nq,)
        opt_shift_ask = shift_ask_flat[best_idx]  # (nq,)

        # Store in bid_shifts and ask_shifts for this step
        # n is steps remaining, so step index = steps - n
        step_idx = steps - n
        bid_shifts[step_idx, :] = opt_shift_bid
        ask_shifts[step_idx, :] = opt_shift_ask

        # Update V: V_new[q] = value_matrix[q, best_idx[q]]
        V = value_matrix[np.arange(nq), best_idx]

        # Check time budget
        if n % 20 == 0 and _remaining_budget() < 20.0:
            return None

    return {
        'bid_shifts': bid_shifts,
        'ask_shifts': ask_shifts,
        'base_bid': inv_k,
        'base_ask': inv_k,
        'qs': qs,
        'q_step': q_step,
        'q_min': q_min,
        'q_max': q_max,
    }


def _interp_shift(q, qs, shifts):
    """
    Linearly interpolate shifts at continuous q values using grid qs.
    q: (N,) array of inventory values
    qs: (nq,) sorted grid
    shifts: (nq,) array of shift values on the grid
    Returns: (N,) array of interpolated shifts
    """
    q_step = qs[1] - qs[0]
    idx = (q - qs[0]) / q_step
    i0 = np.floor(idx).astype(int)
    i1 = i0 + 1
    i0 = np.clip(i0, 0, len(qs) - 1)
    i1 = np.clip(i1, 0, len(qs) - 1)
    frac = idx - np.floor(idx)
    frac = np.clip(frac, 0.0, 1.0)
    return (1.0 - frac) * shifts[i0] + frac * shifts[i1]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _V_TABLES:
        table = _solve_dp(sigma, A, k, gamma, T, steps, Q)
        _V_TABLES[key] = table
    table = _V_TABLES[key]

    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k

    if table is None:
        # Fall back to analytic shift (no DP)
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        q = inventory.astype(float)
        bid = np.clip(inv_k + q * shift, 0.0, max_depth)
        ask = np.clip(inv_k - q * shift, 0.0, max_depth)
        bid = np.nan_to_num(bid, nan=inv_k, posinf=max_depth, neginf=0.0)
        ask = np.nan_to_num(ask, nan=inv_k, posinf=max_depth, neginf=0.0)
        bid = np.clip(bid, 0.0, max_depth)
        ask = np.clip(ask, 0.0, max_depth)
        return bid.astype(float), ask.astype(float)

    # Determine current step index
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = np.clip(step_idx, 0, steps - 1)

    q = inventory.astype(float)

    # Interpolate the optimal shifts for this step
    bid_shift = _interp_shift(q, table['qs'], table['bid_shifts'][step_idx])
    ask_shift = _interp_shift(q, table['qs'], table['ask_shifts'][step_idx])

    # Compute depths
    bid = table['base_bid'] + bid_shift
    ask = table['base_ask'] + ask_shift

    # Ensure valid values
    bid = np.nan_to_num(bid, nan=inv_k, posinf=max_depth, neginf=0.0)
    ask = np.nan_to_num(ask, nan=inv_k, posinf=max_depth, neginf=0.0)
    bid = np.clip(bid, 0.0, max_depth)
    ask = np.clip(ask, 0.0, max_depth)

    return bid.astype(float), ask.astype(float)
# EVOLVE-BLOCK-END
