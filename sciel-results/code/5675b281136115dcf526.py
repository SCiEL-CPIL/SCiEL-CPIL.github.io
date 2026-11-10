"""Optimal market making via precomputed backward-induction DP table."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s limit
_dp_cache = {}


def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """Precompute optimal bid/ask depth tables via backward induction.

    Returns (bid_tab, ask_tab) each of shape (steps+1, 2*Q+1) floats.
    bid_tab[n, q+Q] = optimal bid depth at step n, inventory q.
    ask_tab[n, q+Q] = optimal ask depth at step n, inventory q.
    np.inf means no quote.
    """
    dt = T / steps
    nq = 2 * Q + 1
    idx = np.arange(nq)
    q_vals = (idx - Q).astype(np.float64)

    # Cost for inventory q before step's fills
    cost = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt

    # Terminal value: no terminal charge
    V_next = np.zeros(nq)

    # Precompute depth candidates
    # Max useful depth: beyond this, fill prob ~ 1 - exp(-A*exp(-k*d)*dt) ~ A*exp(-k*d)*dt
    # For A=140, dt=0.001: A*dt = 0.14. Fill prob ~ 0.14*exp(-k*d).
    # At d = 1/k: fill ~ 0.14/k. At d = 2/k: fill ~ 0.14/(k*e^2) ~ 0.024/k.
    # Depth beyond ~5/k gives negligible fills; but we need fine resolution near 0.
    # Use a log-spaced grid plus a linear grid for small depths.
    max_d = 6.0 / k
    # Linear grid for small depths (high resolution where it matters)
    d_lin = np.linspace(0.0, 0.5 / k, 401)
    # Log-spaced for larger depths
    d_log = np.geomspace(0.5 / k, max_d, 400)
    # Combine and unique
    d_cand = np.unique(np.concatenate([d_lin, d_log]))
    # Ensure minimum depth is small
    d_cand = d_cand[d_cand > 1e-9]
    nd = len(d_cand)

    # Fill probability: p(d) = 1 - exp(-A * exp(-k*d) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * d_cand) * dt)  # shape (nd,)

    # Backward induction
    bid_tab = np.full((steps + 1, nq), np.inf)
    ask_tab = np.full((steps + 1, nq), np.inf)

    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > _DEADLINE:
            # Fill remaining with heuristic
            for nn in range(n, -1, -1):
                tau = (steps - nn) * dt
                alpha = gamma * sigma * sigma * tau
                d0 = 1.0 / k
                for qq in range(nq):
                    qv = qq - Q
                    bid_tab[nn, qq] = np.clip(d0 + alpha * qv, -3.0/k + 1e-6, 50.0/k)
                    ask_tab[nn, qq] = np.clip(d0 - alpha * qv, -3.0/k + 1e-6, 50.0/k)
                bid_tab[nn, nq-1] = np.inf  # q=Q: no bid
                ask_tab[nn, 0] = np.inf     # q=-Q: no ask
            return bid_tab, ask_tab

        V_cur = np.zeros(nq)

        # For each inventory level, compute optimal bid and ask depths
        # Bid: choosing depth d >= 0 gives immediate reward d (if filled) with prob p(d),
        # then transitions to q-1 with prob p(d), q with prob (1-p(d)).
        # Value = -cost[q] + sum_d [ p(d)*d + p(d)*V_next[q-1] + (1-p(d))*V_next[q] ]
        #      = -cost[q] + V_next[q] + sum_d p(d)*(d + V_next[q-1] - V_next[q])
        # Optimal: maximize over d of p(d)*(d + V_next[q-1] - V_next[q])
        # Note: d=0 (no bid) gives 0. So we take max(0, best).

        # For ask: choosing depth d >= 0 gives immediate reward -d (if filled, we sell at mid+d,
        # so we earn depth d... wait, let me re-read.
        # "A fill earns its depth" - bid fill at depth d below mid: we buy at mid-d, 
        # the "depth earned" is d (we bought cheaper). Ask fill at depth d above mid:
        # we sell at mid+d, depth earned is d.
        # Actually re-reading: "distances below and above mid to quote this step"
        # bid_depth is distance below mid, ask_depth is distance above mid.
        # "A fill earns its depth" - so if bid is hit, we earn bid_depth (bought low).
        # If ask is hit, we earn ask_depth (sold high).
        # Inventory: bid hit -> we bought -> inventory increases by 1 (q -> q+1)
        # Ask hit -> we sold -> inventory decreases by 1 (q -> q-1)
        # Wait, let me re-check: "A fill earns its depth and moves inventory one unit"
        # "no bid fill at q = Q, no ask fill at q = -Q"
        # So bid fill at q=Q is forbidden (can't buy more when at max long).
        # Ask fill at q=-Q is forbidden (can't sell more when at max short).
        # Bid fill: inventory goes from q to q+1 (we accumulated more).
        # Ask fill: inventory goes from q to q-1 (we reduced/shortened).

        # Bid value at inventory q:
        # If we quote bid at depth d:
        #   Fill prob p_b = 1 - exp(-A*exp(-k*d)*dt)
        #   If filled: earn d, inventory q -> q+1 (if q < Q)
        #   If not filled: earn 0, inventory stays q
        # Value_bid = p_b * (d + V_next[q+1]) + (1-p_b) * V_next[q]
        #          = V_next[q] + p_b * (d + V_next[q+1] - V_next[q])
        # Maximize over d >= 0 (including d=0 meaning no bid, p=0, value=V_next[q])

        # Ask value at inventory q:
        # If we quote ask at depth d:
        #   Fill prob p_a = 1 - exp(-A*exp(-k*d)*dt)
        #   If filled: earn d, inventory q -> q-1 (if q > -Q)
        #   If not filled: earn 0, inventory stays q
        # Value_ask = p_a * (d + V_next[q-1]) + (1-p_a) * V_next[q]
        #          = V_next[q] + p_a * (d + V_next[q-1] - V_next[q])
        # Maximize over d >= 0

        # For bid: q ranges 0..Q-1 (no bid at q=Q)
        # For ask: q ranges 1..Q (no ask at q=-Q)

        # Bid optimization for q = 0..Q-1 (indices 0..Q-1 in 0-based, actual q = idx-Q)
        for qq in range(Q):  # indices 0 to Q-1, actual q = qq - Q, range -Q to Q-1
            # q+1 index = qq+1
            delta_bid = V_next[qq + 1] - V_next[qq]  # V_next[q+1] - V_next[q]
            # Maximize p(d) * (d + delta_bid) over d in d_cand, also consider d=0 -> 0
            vals = p_fill * (d_cand + delta_bid)
            best_val = np.max(vals) if nd > 0 else 0.0
            if best_val > 0:
                best_idx = np.argmax(vals)
                bid_tab[n, qq] = d_cand[best_idx]
            else:
                bid_tab[n, qq] = 0.0  # No bid (depth 0 means no quote? or infinitesimal?)
                # Actually depth 0 means quoting at mid. p(0) = 1-exp(-A*dt) > 0.
                # But if the optimal is to not quote, we should use inf.
                # Let me reconsider: the problem says "np.inf means no quote"
                # So we should either quote at some depth d > 0 or use inf.
                # But quoting at d=0 gives fill prob 1-exp(-A*dt) and earns 0.
                # If delta_bid < 0, quoting bid hurts (we'd rather not be long).
                # In that case, no bid (inf) is optimal.
                bid_tab[n, qq] = np.inf

        # No bid at q = Q (index Q)
        bid_tab[n, Q] = np.inf

        # Ask optimization for q = -Q+1..Q (indices 1..2Q)
        for qq in range(1, nq):  # indices 1 to 2Q, actual q = qq - Q, range -Q+1 to Q
            # q-1 index = qq-1
            delta_ask = V_next[qq - 1] - V_next[qq]  # V_next[q-1] - V_next[q]
            # Maximize p(d) * (d + delta_ask) over d in d_cand
            vals = p_fill * (d_cand + delta_ask)
            best_val = np.max(vals) if nd > 0 else 0.0
            if best_val > 0:
                best_idx = np.argmax(vals)
                ask_tab[n, qq] = d_cand[best_idx]
            else:
                ask_tab[n, qq] = np.inf

        # No ask at q = -Q (index 0)
        ask_tab[n, 0] = np.inf

        # V_cur[q] = -cost[q] + max_bid_value + max_ask_value
        # Bid contribution: if bid_tab[n,qq] is finite, value = p(d)*(d + delta_bid)
        # Ask contribution: if ask_tab[n,qq] is finite, value = p(d)*(d + delta_ask)
        # If inf (no quote), contribution = 0

        bid_contrib = np.zeros(nq)
        ask_contrib = np.zeros(nq)

        # Bid contributions
        finite_bid = np.isfinite(bid_tab[n, :Q])
        if np.any(finite_bid):
            for qq in np.where(finite_bid)[0]:
                d = bid_tab[n, qq]
                p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                bid_contrib[qq] = p * (d + V_next[qq + 1] - V_next[qq])

        # Ask contributions
        finite_ask = np.isfinite(ask_tab[n, 1:])
        if np.any(finite_ask):
            for qq in np.where(finite_ask)[0] + 1:
                d = ask_tab[n, qq]
                p = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
                ask_contrib[qq] = p * (d + V_next[qq - 1] - V_next[qq])

        V_cur = -cost + bid_contrib + ask_contrib
        V_next = V_cur

    return bid_tab, ask_tab


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    # Cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _dp_cache:
        _dp_cache[key] = _compute_dp(sigma, A, k, gamma, T, steps, Q)

    bid_tab, ask_tab = _dp_cache[key]

    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps))

    q = np.asarray(inventory, dtype=np.int64)
    # Clip to valid range
    q = np.clip(q, -Q, Q)
    idx = q + Q  # 0 to 2Q

    d_bid = bid_tab[n, idx]
    d_ask = ask_tab[n, idx]

    # Ensure valid: values below -3/k are invalid
    min_depth = -3.0 / k
    d_bid = np.where(d_bid < min_depth, np.inf, d_bid)
    d_ask = np.where(d_ask < min_depth, np.inf, d_ask)

    # Replace 0.0 with inf (no quote) - actually 0 depth is a valid quote at mid
    # But in our DP, we set inf when not quoting. Let's keep as is.
    # Actually, depth 0 means quoting exactly at mid. That's valid.
    # But p(0) = 1-exp(-A*dt) which is significant. If the DP says 0, it means
    # the optimal is to quote at mid. That's fine.

    d_bid = np.asarray(d_bid, dtype=np.float64)
    d_ask = np.asarray(d_ask, dtype=np.float64)

    return (d_bid, d_ask)
# EVOLVE-BLOCK-END
