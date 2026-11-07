"""
Strategy: Analytical Closed-Form Approximation of Optimal Depth.
Derives the optimal bid and ask depths using the first-order condition of the Bellman equation,
treating the depth as a continuous variable. This eliminates the need for grid search and
pre-computation, providing a "continuous" solution that avoids discretization error.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 320.0

def quote(t, inventory, mid, params):
    """
    Compute optimal bid and ask depths using analytical approximation.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n = int(np.round(t * steps))
    if n >= steps:
        n = steps - 1
    
    tau = (steps - 1 - n) * dt  # Time to horizon (approx)
    # Use a minimum tau to avoid division by zero or extreme values at the end
    tau_safe = np.maximum(tau, dt)
    
    q = np.asarray(inventory, dtype=np.float64)
    
    # Objective: Maximize E[depth] - cost
    # Bellman: V(q, tau) = max_d [ p(d)*d + (1-p(d))*V(q+1, tau-dt) ... ] - cost
    # Continuous approximation:
    # The optimal depth d* balances the marginal gain from increased fill probability/depth
    # against the marginal cost of inventory risk.
    
    # For a single step, if we ignore the value of future states (myopic),
    # we maximize d * (1 - exp(-A exp(-k d) dt))
    # Let p(d) = 1 - exp(-lambda exp(-k d)) where lambda = A dt
    # f(d) = d * p(d)
    # f'(d) = p(d) + d * p'(d)
    # p'(d) = exp(-lambda exp(-k d)) * lambda * exp(-k d) * k
    # p'(d) = (1 - p(d)) * lambda * exp(-k d) * k
    # Set f'(d) = 0:
    # p(d) + d * k * lambda * exp(-k d) * (1 - p(d)) = 0
    # This is transcendental.
    
    # However, we must consider inventory risk.
    # The "cost" of holding inventory q is approx C(q) = 0.5 * gamma * sigma^2 * q^2 * dt (immediate)
    # plus the change in value function due to inventory shift.
    # In a linear-quadratic approximation, the optimal action often depends linearly on state.
    
    # Let's use a heuristic derived from balancing "reward from fill" vs "risk penalty".
    # Reward ~ d * p(d)
    # Risk ~ gamma * sigma^2 * q * d (roughly, if fill moves q by 1, and cost is quadratic)
    # Actually, the value function V(q) is convex. V'(q) ~ gamma * sigma^2 * q * (something).
    
    # A common approximation for market making with inventory q:
    # Skew the quotes to push inventory towards 0.
    # If q > 0, lower bid (or raise ask) to sell.
    # If q < 0, raise bid (or lower ask) to buy.
    
    # Let's model the optimal depth as a function of a "pressure" term.
    # Pressure P = gamma * sigma^2 * q / (A * k) ??
    
    # Let's look at the first order condition more carefully for the infinite horizon stationary case.
    # V(q) = max_{db, da} [ db * pb(db) + da * pa(da) + E[V(q')] - C(q) ]
    # E[V(q')] = (1-pb)(1-pa)V(q) + pb(1-pa)V(q+1) + (1-pb)pa V(q-1) + pb pa V(q)
    # V(q) - E[V(q')] = pb(1-pa)(V(q) - V(q+1)) + (1-pb)pa(V(q) - V(q-1))
    # Let Delta+ = V(q) - V(q+1) and Delta- = V(q) - V(q-1).
    # For small dt, Delta+ ~ -V'(q) dt, Delta- ~ V'(q) dt.
    # If V(q) ~ c q^2, then V'(q) ~ 2 c q.
    # The term becomes: pb(1-pa)(-V'(q)) + (1-pb)pa(V'(q))
    # = V'(q) [ pa(1-pb) - pb(1-pa) ] = V'(q) [ pa - pb ]
    
    # So the effective cost/gain from inventory shift is V'(q) * (pa - pb).
    # We want to maximize: db*pb(db) + da*pa(da) + V'(q)(pa(da) - pb(db)) - C(q)
    
    # Let's approximate V'(q) using the immediate cost gradient.
    # C(q) = 0.5 gamma sigma^2 q^2 dt.
    # In steady state, V(q) should reflect the accumulated cost.
    # A reasonable estimate for the "shadow price" of inventory is S = gamma * sigma^2 * q.
    
    # Objective for bid depth db:
    # f_b(db) = db * pb(db) - S * pb(db)
    # where S = gamma * sigma^2 * q. (If q>0, S>0, so we penalize buying, lowering db)
    # f_b(db) = pb(db) * (db - S)
    # Maximize g(x) = (1 - exp(-lambda exp(-k x))) * (x - S)
    # Let y = x - S. Then x = y + S.
    # g(y) = (1 - exp(-lambda exp(-k(y+S)))) * y
    # If S is large and positive (high inventory), we want small db (or negative, but depth >= 0).
    # If S is negative (short inventory), we want large db.
    
    # Let's solve for optimal y for the function h(y) = y * (1 - exp(-B exp(-k y)))
    # where B = lambda * exp(-k S) = A dt * exp(-k S).
    
    # If B is small, p ~ B exp(-k y).
    # h(y) ~ y * B exp(-k y).
    # h'(y) = B (exp(-ky) - y k exp(-ky)) = B exp(-ky) (1 - ky).
    # Max at y = 1/k.
    # So optimal db ~ S + 1/k.
    # If S + 1/k < 0, optimal db = 0.
    
    # Let's refine this.
    # For ask depth da:
    # f_a(da) = da * pa(da) + S * pa(da)  (Note: selling reduces q, so if q>0, selling is good. V(q)-V(q-1) > 0. Term is +S*pa)
    # Wait, let's re-verify the sign.
    # V(q) - E[V(q')] ...
    # If we sell (ask hit), q -> q-1.
    # Term: pa * (V(q) - V(q-1)) ~ pa * V'(q).
    # If V(q) is convex (risk penalty), V'(q) > 0 for q > 0.
    # So selling gives a positive value boost.
    # So objective for ask: da * pa(da) + S * pa(da) = pa(da) (da + S).
    
    # So:
    # Bid optimal: max db such that db > S, roughly db ~ max(0, S + 1/k)
    # Ask optimal: max da such that da > -S, roughly da ~ max(0, -S + 1/k)
    
    # Let's calculate S for each path.
    S = gamma * sigma**2 * q
    
    # Base depth scale from fill probability
    # The "1/k" comes from the exponential decay.
    # More accurately, we can solve the transcendental equation numerically for a few key points
    # or use a better approximation.
    # Let's use the approximation:
    # Optimal depth d* satisfies: 1 - k d* (1 - exp(-k d*)) = exp(-lambda exp(-k d*)) ... complex.
    
    # Simple robust approximation:
    # d_bid = max(0, S + 1.0/k)
    # d_ask = max(0, -S + 1.0/k)
    
    # However, we must ensure depths are not too large.
    # Also, the "1/k" is the mode of the exponential distribution of the "effective" depth.
    
    # Let's add a small correction factor based on A and dt.
    # If A dt is very small, fills are rare, we should quote tighter (smaller depth) to ensure fill?
    # No, if fills are rare, the penalty for inventory is less urgent?
    # Actually, if fills are rare, the "reward" db*pb is small. The inventory cost S dominates.
    
    # Let's stick to the S + 1/k heuristic but clip it.
    # The problem states depth must be >= 0.
    # Also, depth shouldn't be unreasonably large. 3/k was a lower bound check in the prompt?
    # "Values below -3/k ... make the whole result invalid."
    # This implies depth should probably be >= -3/k? But depth is distance, so >= 0.
    # Wait, "distances below and above mid". Usually positive.
    # The prompt says: "distances below and above mid to quote this step; np.inf means no quote."
    # "Values below -3/k ... invalid". This suggests the output can be negative?
    # "distances below... mid". If bid_depth is negative, it means quoting *above* mid?
    # No, standard market making: bid = mid - depth, ask = mid + depth.
    # If depth is negative, bid = mid + |depth|, which is above mid. That's crossing the spread.
    # The constraint "below -3/k" is weird. Maybe it means the *value* of the depth variable.
    # Let's assume depth must be >= 0. The -3/k might be a typo in my understanding or refers to log-depth?
    # "distances ... one per simulated path".
    # Let's assume non-negative depths.
    
    d_bid = np.maximum(0.0, S + 1.0/k)
    d_ask = np.maximum(0.0, -S + 1.0/k)
    
    # Ensure no quote if inventory is at limit?
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # This implies we *can* quote, but it won't fill? Or we should not quote?
    # "no bid fill at q = Q" -> If q=Q, bid is ignored.
    # It doesn't say "no bid quote". But quoting is useless.
    # However, the objective includes "depths earned". If no fill, no depth earned.
    # So it's neutral. But to be safe, we can return 0 or inf.
    # The prompt says "np.inf means no quote".
    # If we quote and it doesn't fill, we earn 0.
    # If we don't quote, we earn 0.
    # So it doesn't matter for the objective, except for the "invalid" check.
    # Let's return 0 for no-quote scenarios if we want to be explicit, or inf.
    # Inf is safer if it implies "no position taken".
    
    # Apply boundary conditions
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    
    # Sanity check: depths should not be huge.
    # Max depth in grid was 3.0 in previous attempts.
    # 1/k for k=1.5 is 0.66. S can be large.
    # If q=30, sigma=2, gamma=0.05. S = 0.05 * 4 * 30 = 6.
    # d_bid = 6 + 0.66 = 6.66.
    # Is this valid?
    # The prompt says "Values below -3/k ... invalid". It doesn't give an upper bound.
    # But extremely large depths mean no fill.
    # If d is very large, p ~ 0.
    # The approximation holds for small p.
    
    return d_bid, d_ask

# EVOLVE-BLOCK-END
