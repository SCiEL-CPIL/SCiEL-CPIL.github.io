"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0

_OPT_TABLES = {}


def _solve_hjb(params):
    """
    Solve the discrete HJB equation via backward induction.
    Uses a fine depth grid with local refinement for sub-grid accuracy.
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

    n_inv = 2 * Q + 1
    inv_vals = np.arange(-Q, Q + 1, dtype=np.float64)

    # Depth grid: fine enough to capture the optimum
    # Optimal depths are typically in [1/k, 5/k] range
    n_grid = 800
    depth_min = 0.001
    depth_max = 20.0 / k
    # Use a combination of linear and log spacing for better coverage
    depths = np.geomspace(depth_min, depth_max, n_grid)

    # Pre-compute fill probabilities
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    fill_probs = np.clip(fill_probs, 0.0, 1.0)

    # Value function
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid_idx = np.zeros((steps, n_inv), dtype=np.int32)
    opt_ask_idx = np.zeros((steps, n_inv), dtype=np.int32)

    # Holding costs
    holding_cost = 0.5 * gamma * sigma**2 * inv_vals**2 * dt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 20.0:
            # Fallback
            fallback_idx = n_grid // 2
            opt_bid_idx[n, :] = fallback_idx
            opt_ask_idx[n, :] = fallback_idx
            continue

        V_next = V[n + 1, :]

        # Bid side: V_next[i+1] for i < n_inv-1
        V_next_bid = np.zeros(n_inv)
        V_next_bid[:n_inv - 1] = V_next[1:]

        # bid_vals[i, j] = fill_probs[j]*depths[j] + V_next[i] + fill_probs[j]*(V_next_bid[i] - V_next[i])
        # = V_next[i] + fill_probs[j]*(depths[j] + V_next_bid[i] - V_next[i])
        # = V_next[i] + fill_probs[j]*(depths[j] + delta_bid[i])
        delta_bid = V_next_bid - V_next  # shape (n_inv,)
        # bid_vals shape: (n_inv, n_grid)
        bid_vals = V_next[:, None] + fill_probs[None, :] * (depths[None, :] + delta_bid[:, None])
        opt_bid_idx[n, :] = np.argmax(bid_vals, axis=1)

        # Ask side: V_next[i-1] for i > 0
        V_next_ask = np.zeros(n_inv)
        V_next_ask[1:] = V_next[:-1]
        delta_ask = V_next_ask - V_next
        ask_vals = V_next[:, None] + fill_probs[None, :] * (depths[None, :] + delta_ask[:, None])
        opt_ask_idx[n, :] = np.argmax(ask_vals, axis=1)

        # Compute V[n, i]
        best_bid = opt_bid_idx[n, :]
        best_ask = opt_ask_idx[n, :]
        p_bid = fill_probs[best_bid]
        p_ask = fill_probs[best_ask]
        d_bid = depths[best_bid]
        d_ask = depths[best_ask]

        EV_bid = np.zeros(n_inv)
        EV_bid[:n_inv - 1] = p_bid[:n_inv - 1] * (d_bid[:n_inv - 1] + V_next[1:])

        EV_ask = np.zeros(n_inv)
        EV_ask[1:] = p_ask[1:] * (d_ask[1:] + V_next[:-1])

        EV_none = (1.0 - p_bid) * (1.0 - p_ask) * V_next

        V[n, :] = -holding_cost + EV_bid + EV_ask + EV_none

    # Build depth tables
    bid_depths_table = depths[opt_bid_idx]
    ask_depths_table = depths[opt_ask_idx]

    # Local refinement using Newton's method on the first-order condition
    if time.time() < _DEADLINE - 40.0:
        for n in range(steps):
            for i in range(n_inv):
                # Bid refinement
                coarse_bid = bid_depths_table[n, i]
                if i < n_inv - 1:
                    V_nb = V[n + 1, i + 1]
                    V_ni = V[n + 1, i]
                    delta = V_nb - V_ni
                else:
                    delta = 0.0

                # Optimize: f(d) = (1-exp(-A*exp(-k*d)*dt))*d + p*(delta)
                # where p = 1-exp(-A*exp(-k*d)*dt)
                # f'(d) = A*exp(-k*d)*dt*exp(-k*d) * (d - (1/p - 1) ... )
                # Let's use a simpler approach: golden section search in a small window
                lo = max(0.0001, coarse_bid * 0.3)
                hi = min(depth_max, coarse_bid * 3.0)
                if hi <= lo:
                    hi = lo + 0.01

                # Golden section search
                gr = (np.sqrt(5) - 1) / 2
                a, b = lo, hi
                c = b - gr * (b - a)
                d = a + gr * (b - a)

                def _obj_bid(dd):
                    p = 1.0 - np.exp(-A * np.exp(-k * dd) * dt)
                    return p * (dd + delta)

                fc = _obj_bid(c)
                fd = _obj_bid(d)
                for _ in range(30):
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = _obj_bid(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = _obj_bid(d)
                best_bid_depth = (a + b) / 2
                bid_depths_table[n, i] = best_bid_depth

                # Ask refinement
                coarse_ask = ask_depths_table[n, i]
                if i > 0:
                    V_na = V[n + 1, i - 1]
                    V_ni = V[n + 1, i]
                    delta_a = V_na - V_ni
                else:
                    delta_a = 0.0

                lo = max(0.0001, coarse_ask * 0.3)
                hi = min(depth_max, coarse_ask * 3.0)
                if hi <= lo:
                    hi = lo + 0.01

                a, b = lo, hi
                c = b - gr * (b - a)
                d = a + gr * (b - a)

                def _obj_ask(dd):
                    p = 1.0 - np.exp(-A * np.exp(-k * dd) * dt)
                    return p * (dd + delta_a)

                fc = _obj_ask(c)
                fd = _obj_ask(d)
                for _ in range(30):
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = _obj_ask(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = _obj_ask(d)
                best_ask_depth = (a + b) / 2
                ask_depths_table[n, i] = best_ask_depth

    result = {
        'bid': bid_depths_table,
        'ask': ask_depths_table,
        'n_inv': n_inv,
    }
    _OPT_TABLES[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths using pre-computed HJB optimal policy.
    Returns (bid_depths, ask_depths) each of shape (len(inventory),).
    """
    table = _solve_hjb(params)

    Q = params["Q"]
    steps = params["steps"]

    # Determine current step index
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    # Map inventory to index: inv_idx = q + Q
    inv_idx = inventory + Q
    inv_idx = np.clip(inv_idx, 0, table['n_inv'] - 1)

    # Look up depths
    bid_depths = table['bid'][n, inv_idx].astype(np.float64)
    ask_depths = table['ask'][n, inv_idx].astype(np.float64)

    # Ensure depths are valid
    k = params["k"]
    min_depth = -3.0 / k + 1e-10

    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, 1.0 / k)

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
