"""Market-making with inventory risk: DP over inventory grid with time-varying policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_V_TABLES = {}
_T0 = time.time()
_DEADLINE = 290.0


def _remaining_budget():
    return _DEADLINE - (time.time() - _T0)


def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the discrete DP. Returns a dict with:
      - qs: grid of inventory values
      - policies: array of shape (steps, nq) where policies[n, i] gives
                  (bid_depth, ask_depth) as a packed float64? No, separate arrays.
      - bid_policies: (steps, nq)
      - ask_policies: (steps, nq)
      - shift: analytic shift for fallback
      - inv_k: 1/k
    """
    if _remaining_budget() < 30.0:
        return None

    dt = T / steps
    inv_k = 1.0 / k
    max_depth = 3.0 * inv_k

    # Grid for inventory
    # Use integer grid for exactness since inventory is integer
    q_min = -Q
    q_max = Q
    nq = q_max - q_min + 1
    qs = np.arange(q_min, q_max + 1, dtype=np.float64)

    # Precompute the optimal shift approximation for fallback
    shift = gamma * sigma * sigma * 0.5 / (A * k)

    # DP: V[i] = value at current step for inventory qs[i]
    # We iterate backwards from step steps-1 down to 0
    # At each step, we need to choose bid_depth and ask_depth to maximize
    #   (bid + ask) - (gamma/2)*sigma^2*q^2*dt + E[V_next]
    #
    # The optimal depth choice for a given q and step n is a 1D optimization.
    # bid_depth in [0, max_depth], ask_depth in [0, max_depth]
    #
    # For speed, we can note that the optimal bid_depth depends only on q, n, and the
    # continuation values V(q-1) and V(q+1) (approximately, since fill changes inventory by 1).
    #
    # Actually, let's think more carefully. The fill probability for bid is:
    #   pb = 1 - exp(-A * exp(-k * bid_depth) * dt)
    # The expected continuation value after a bid fill is V(q+1) [inventory increases by 1 on bid fill]
    # Wait: bid fill means we buy, so inventory increases. Ask fill means we sell, inventory decreases.
    #
    # Let me re-read: "A fill earns its depth and moves inventory one unit"
    # "bid hit" -> we sell? Or we buy?
    # In market making, bid is the price we offer to buy, ask is the price we offer to sell.
    # If a bid is hit, someone buys from us at our bid price? No, typically:
    # - We quote bid (buy) and ask (sell)
    # - If bid is hit, we buy the asset, inventory increases
    # - If ask is hit, we sell the asset, inventory decreases
    #
    # "A fill earns its depth" - the depth is the distance from mid, which is the profit/loss.
    # If we buy at bid_depth below mid, we earn bid_depth (we bought cheap).
    # If we sell at ask_depth above mid, we earn ask_depth (we sold high).
    #
    # So the reward is bid_depth + ask_depth when fills occur.
    #
    # Inventory change: bid fill -> +1, ask fill -> -1
    #
    # Boundary: no bid fill at q = Q (can't buy more), no ask fill at q = -Q (can't sell more)
    #
    # The DP is:
    # V_n(q) = max_{b, a} [ (b + a) - (gamma/2)*sigma^2*q^2*dt
    #                         + pb(q,b)*V_{n-1}(q+1) + pa(q,a)*V_{n-1}(q-1)
    #                         + (1 - pb - pa)*V_{n-1}(q) ]
    # where pb = 1 - exp(-A*exp(-k*b)*dt), pa = 1 - exp(-A*exp(-k*a)*dt)
    # with pb = 0 if q = Q, pa = 0 if q = -Q
    #
    # This is a 2D optimization per (n, q), which is expensive.
    #
    # Key insight: the optimal b and a can be found by 1D search each, since the objective
    # separates in b and a except for the (1-pb-pa) term. Actually it doesn't fully separate.
    #
    # Alternative: use the fact that for small dt, the optimal shift is approximately
    #   delta_bid = shift * q, delta_ask = -shift * q (relative to inv_k)
    # i.e., bid_depth = inv_k + shift*q, ask_depth = inv_k - shift*q
    #
    # Let's try a hybrid: use the analytic shift as a starting point and do a local refinement.
    #
    # For now, let's implement the full DP with a coarse grid search over depths.
    #
    # Actually, let's first try the analytic solution and see if we can improve it.
    # The current score is 1.1215, which is already above 1.0.
    #
    # Let me implement a proper DP with numerical optimization of depths.
    #
    # For each (n, q), we need to maximize over b in [0, max_depth] and a in [0, max_depth].
    # This is a 2D problem, but we can use the fact that the optimal b depends on
    # the "shadow price" of inventory, which is related to V(q+1) - V(q) and V(q) - V(q-1).
    #
    # Let's define:
    #   dV_plus = V(q+1) - V(q)  (value of having one more unit)
    #   dV_minus = V(q) - V(q-1)  (value of having one less unit)
    #
    # The optimal bid_depth b* maximizes:
    #   b + pb(b) * dV_plus + (1 - pb(b)) * 0  [ignoring the ask side for a moment]
    #   = b + (1 - exp(-A*exp(-k*b)*dt)) * dV_plus
    #
    # Taking derivative w.r.t. b:
    #   1 + A*exp(-k*b)*dt * (-k) * exp(-k*b) * dt... wait, let me redo this.
    #
    # pb(b) = 1 - exp(-A*exp(-k*b)*dt)
    # dpb/db = -exp(-A*exp(-k*b)*dt) * A * (-k) * exp(-k*b) * dt
    #        = k*A*dt*exp(-k*b)*exp(-A*exp(-k*b)*dt)
    #
    # f(b) = b + pb(b) * dV_plus
    # f'(b) = 1 + dpb/db * dV_plus
    #       = 1 + k*A*dt*exp(-k*b)*exp(-A*exp(-k*b)*dt) * dV_plus
    #
    # For the optimum, f'(b) = 0, but since f'(b) > 0 always (all terms positive if dV_plus > 0),
    # the maximum is at the boundary b = max_depth if dV_plus > 0.
    # If dV_plus < 0, we want to minimize pb, so b = 0.
    #
    # This suggests the optimal policy is bang-bang: either quote at max depth or no quote.
    # But that doesn't match the smooth shift solution. Let me reconsider.
    #
    # Ah, I forgot the ask side. The full objective is:
    #   b + a - (gamma/2)*sigma^2*q^2*dt
    #   + pb(b)*V(q+1) + pa(a)*V(q-1) + (1-pb-pa)*V(q)
    #   = b + a - (gamma/2)*sigma^2*q^2*dt
    #   + pb(b)*(V(q+1) - V(q)) + pa(a)*(V(q-1) - V(q))
    #   = b + a - (gamma/2)*sigma^2*q^2*dt
    #   + pb(b)*dV_plus + pa(a)*dV_minus
    #
    # So the bid and ask optimizations DO separate!
    #
    # Optimal b: maximize g(b) = b + pb(b) * dV_plus
    # Optimal a: maximize h(a) = a + pa(a) * dV_minus
    #
    # where dV_plus = V(q+1) - V(q) and dV_minus = V(q-1) - V(q)
    #
    # For g(b): if dV_plus >= 0, g is increasing in b (since pb is increasing in b), so b* = max_depth.
    # If dV_plus < 0, g is decreasing in b, so b* = 0.
    #
    # Wait, that can't be right. Let me check: if dV_plus < 0, having more inventory is bad,
    # so we want to avoid buying, hence b* = 0 (no bid quote). That makes sense.
    # If dV_plus > 0, having more inventory is good, so we want to buy, hence b* = max_depth.
    #
    # But this gives a bang-bang policy, not a smooth shift. The issue is that I'm not
    # accounting for the risk term properly. The risk term is fixed given q, so it doesn't
    # affect the depth choice. The depth choice only depends on dV_plus and dV_minus.
    #
    # Hmm, but the current program uses a smooth shift and gets a score of 1.1215.
    # Let me re-examine. Maybe the bang-bang policy is actually optimal for this discrete model?
    #
    # Let me just implement the DP with the separated 1D optimizations and see.
    #
    # Actually, I realize the issue: the "depth" is the distance from mid, and we earn the depth
    # when a fill occurs. But the depth also affects the fill probability. There's a trade-off:
    # deeper quotes earn more per fill but are less likely to fill.
    #
    # The objective for bid is: b + pb(b) * dV_plus
    # where pb(b) = 1 - exp(-A*exp(-k*b)*dt)
    #
    # Let's analyze this more carefully. Let x = exp(-k*b), so b = -ln(x)/k, x in (0, 1].
    # pb = 1 - exp(-A*x*dt)
    # g = -ln(x)/k + (1 - exp(-A*x*dt)) * dV_plus
    #
    # If dV_plus > 0: we want to maximize this. As x -> 0 (b -> inf), g -> inf. But b is capped at max_depth.
    # So if dV_plus > 0, b* = max_depth.
    # If dV_plus < 0: as x -> 1 (b -> 0), g -> 0. As x -> 0, g -> -inf. So b* = 0.
    # If dV_plus = 0: g = -ln(x)/k, maximized at x -> 0, so b* = max_depth. But this is a knife-edge.
    #
    # So the optimal policy is indeed bang-bang. Let me implement this.
    #
    # For the DP:
    # V_0(q) = 0 for all q
    # V_n(q) = max_b g(b) + max_a h(a) - (gamma/2)*sigma^2*q^2*dt
    #         where max_b g(b) = max_depth if dV_plus > 0 else 0
    #         and max_a h(a) = max_depth if dV_minus > 0 else 0
    #
    # Wait, but we also need to store the optimal b and a for the policy.
    #
    # Let me implement this.

    V = np.zeros(nq)
    bid_policies = np.zeros((steps, nq))
    ask_policies = np.zeros((steps, nq))

    for n in range(steps):
        # Compute dV_plus and dV_minus for each q
        # V(q+1) - V(q): for q < Q
        dV_plus = np.zeros(nq)
        dV_minus = np.zeros(nq)
        # q index i corresponds to qs[i] = q_min + i
        # V(q+1) = V[i+1] if i < nq-1, else V[i] (boundary)
        dV_plus[:-1] = V[1:] - V[:-1]
        dV_plus[-1] = 0.0  # at q=Q, no bid fill, so dV_plus doesn't matter
        dV_minus[1:] = V[:-1] - V[1:]
        dV_minus[0] = 0.0  # at q=-Q, no ask fill

        # Optimal bid: max_depth if dV_plus > 0, else 0
        bid_opt = np.where(dV_plus > 0, max_depth, 0.0)
        # At q = Q, no bid fill, so set to 0 (or max_depth, doesn't matter since pb=0)
        bid_opt[-1] = 0.0

        # Optimal ask: max_depth if dV_minus > 0, else 0
        ask_opt = np.where(dV_minus > 0, max_depth, 0.0)
        ask_opt[0] = 0.0

        bid_policies[n] = bid_opt
        ask_policies[n] = ask_opt

        # Compute V_n(q)
        # reward = bid + ask - risk
        risk = (gamma * 0.5 * sigma * sigma * qs ** 2 * dt)
        # Continuation: pb*V(q+1) + pa*V(q-1) + (1-pb-pa)*V(q)
        # pb = 1 - exp(-A*exp(-k*bid)*dt) if bid > 0, else 0
        # pa = 1 - exp(-A*exp(-k*ask)*dt) if ask > 0, else 0
        pb = np.where(bid_opt > 0, 1.0 - np.exp(-A * np.exp(-k * bid_opt) * dt), 0.0)
        pa = np.where(ask_opt > 0, 1.0 - np.exp(-A * np.exp(-k * ask_opt) * dt), 0.0)
        # Boundary: no bid fill at q=Q, no ask fill at q=-Q
        pb[-1] = 0.0
        pa[0] = 0.0

        V_q_plus = np.zeros(nq)
        V_q_minus = np.zeros(nq)
        V_q_plus[:-1] = V[1:]
        V_q_plus[-1] = V[-1]
        V_q_minus[1:] = V[:-1]
        V_q_minus[0] = V[0]

        cont = pb * V_q_plus + pa * V_q_minus + (1.0 - pb - pa) * V
        V_new = bid_opt + ask_opt - risk + cont
        V = V_new

        if n % 100 == 0 and _remaining_budget() < 20.0:
            return None

    return {
        'qs': qs,
        'bid_policies': bid_policies,
        'ask_policies': ask_policies,
        'shift': shift,
        'inv_k': inv_k,
        'q_min': q_min,
        'q_max': q_max,
    }


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
        # Fall back to analytic shift
        shift = gamma * sigma * sigma * 0.5 / (A * k)
        bid = np.clip(inv_k + inventory * shift, 0.0, max_depth)
        ask = np.clip(inv_k - inventory * shift, 0.0, max_depth)
        return bid.astype(float), ask.astype(float)

    # Determine current step index
    dt = T / steps
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)

    # Get policies for this step
    bid_pol = table['bid_policies'][n]
    ask_pol = table['ask_policies'][n]

    # Map inventory to grid index
    q_min = table['q_min']
    q_max = table['q_max']
    q = inventory.astype(np.int64)
    idx = np.clip(q - q_min, 0, q_max - q_min)

    bid = bid_pol[idx].astype(float)
    ask = ask_pol[idx].astype(float)

    # Ensure valid values
    bid = np.nan_to_num(bid, nan=0.0, posinf=max_depth, neginf=0.0)
    ask = np.nan_to_num(ask, nan=0.0, posinf=max_depth, neginf=0.0)
    bid = np.clip(bid, 0.0, max_depth)
    ask = np.clip(ask, 0.0, max_depth)

    return bid, ask
# EVOLVE-BLOCK-END
