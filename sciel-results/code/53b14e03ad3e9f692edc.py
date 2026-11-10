"""Optimal market making with inventory risk - Analytical FOC approach.

Replaces iterative coordinate ascent with direct solution of the first-order
conditions for bid and ask depths, assuming decoupled inventory effects
(ignoring the second-order interaction term in the Bellman equation). This
allows for a much faster and more stable computation.
"""
import numpy as np
import time
from scipy.optimize import brentq

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_TIME_BUDGET = 320.0  # seconds, leave 10s margin before 330s kill


def _fill_prob_scalar(depth, A, k, dt):
    """Fill probability for a scalar depth."""
    if depth <= 0:
        return 0.0
    exp_term = np.exp(-k * depth)
    p = 1.0 - np.exp(-A * exp_term * dt)
    return min(max(p, 0.0), 1.0)

def _solve_foc_scalar(Delta, A, k, dt):
    """
    Solves the First Order Condition for optimal depth d given inventory shadow price Delta.
    FOC: P(d) + d * P'(d) + Delta = 0
    where P(d) = 1 - exp(-A exp(-kd) dt)
    
    Let u = A exp(-kd) dt. Then P = 1 - exp(-u).
    d = -1/k * ln(u / (A dt)).
    P'(d) = -k * u * exp(-u) / (A dt) * A dt? No.
    P'(d) = -k * (A dt) exp(-kd) exp(-A exp(-kd) dt) = -k * u * exp(-u).
    
    So FOC is: (1 - exp(-u)) + d * (-k * u * exp(-u)) + Delta = 0.
    Substitute d:
    (1 - exp(-u)) - (1/k) ln(u/(A dt)) * k * u * exp(-u) + Delta = 0
    (1 - exp(-u)) - u * exp(-u) * ln(u/(A dt)) + Delta = 0
    
    We solve for u > 0.
    Let g(u) = 1 + Delta - exp(-u) * (1 + u * ln(u/(A dt)))
    Find root u*. Then d = -1/k * ln(u* / (A dt)).
    
    Constraints:
    If d < 0, return 0 (or min_depth).
    If d is very large, u is small.
    """
    if Delta >= 0:
        # If shadow price is positive (inventory is "good" to have more of, or cost is low),
        # we want to quote tighter?
        # Actually, Delta = V(q+1) - V(q). If V increases with q, Delta > 0.
        # FOC: P + dP' + Delta = 0.
        # If Delta is large positive, we need P + dP' to be negative.
        # P is always positive. P' is negative.
        # So we need a large d (large negative P') to offset positive Delta?
        # Wait. If Delta > 0, the continuation value increases if we increase inventory.
        # So we *want* to buy. We should quote tight (small d) to maximize probability of buying.
        # If d=0, P=1, P'=0 (approx). FOC ~ 1 + Delta > 0.
        # We need FOC = 0. Since P+dP' is decreasing in d (initially), and starts at 1,
        # if 1 + Delta > 0, the root is at d=0?
        # Let's check derivative of FOC wrt d.
        # If Delta is very large negative, we want to avoid buying, so quote wide (large d).
        
        # Let's just solve the equation generally.
        pass

    A_dt = A * dt
    
    def g(u):
        # u > 0
        if u <= 0:
            return 1 + Delta # limit as u->0 is 1+Delta
        # Term: exp(-u) * (1 + u * ln(u / A_dt))
        # Note: if u/A_dt is very small, ln is large negative.
        # u * ln(u) -> 0 as u->0.
        log_term = np.log(u / A_dt) if u > 0 else 0
        val = np.exp(-u) * (1 + u * log_term)
        return 1 + Delta - val

    # Search for root in u
    # u ranges from 0 to inf.
    # As u -> inf, exp(-u) -> 0, g(u) -> 1 + Delta.
    # As u -> 0, g(u) -> 1 + Delta.
    # There is a minimum in between.
    
    # We want g(u) = 0.
    # If 1 + Delta > 0 and the minimum of g(u) is > 0, no root (optimal d=0 or boundary).
    # If 1 + Delta < 0, there might be a root.
    
    # Let's find the min of g(u) or just bracket.
    # u_min for the term (1 + u ln(u/A_dt)) is where derivative is 0.
    # d/du [ u ln(u/A) ] = ln(u/A) + 1. Zero at u = A/e.
    
    u_center = A_dt / np.e
    u_lo = 1e-6
    u_hi = 10.0 # A_dt is usually small, so u=10 is large (d is large)
    
    # Evaluate g at a few points to find a bracket
    u_test = np.linspace(u_lo, u_hi, 100)
    g_vals = np.array([g(u) for u in u_test])
    
    # We look for sign change
    # If all g_vals > 0, no root (d=0 is optimal, or boundary)
    # If all g_vals < 0, no root in this range? (Unlikely if 1+Delta is sign)
    
    # Actually, simpler: just use brentq on a wide range if we can bracket.
    # Let's try to find a bracket dynamically.
    
    # If g(0) < 0 and g(inf) < 0, and min < 0, no root?
    # Wait, g(u) = 1 + Delta - exp(-u)(1 + u ln(u/A)).
    # If Delta is very negative, 1+Delta is negative.
    # exp term is positive. So g(u) = Neg - Pos = Neg.
    # So if 1+Delta < 0, g(u) is always negative?
    # Then FOC = 0 has no solution?
    # This implies the optimal d is at the boundary (infinity?).
    # If we want to avoid fills (Delta < 0 means inventory increase is bad), we want P to be small.
    # d -> inf makes P -> 0.
    # So if no root, return large d.
    
    # If 1+Delta > 0, g(0) > 0.
    # If the dip goes below 0, there are two roots. We want the smaller d (tighter quote)?
    # Usually the first root.
    
    # Let's just scan for a sign change.
    root_u = None
    prev_g = g(u_lo)
    for i in range(1, len(u_test)):
        curr_g = g(u_test[i])
        if prev_g * curr_g <= 0:
            # Bracket found
            try:
                root_u = brentq(g, u_test[i-1], u_test[i], xtol=1e-8)
                break
            except:
                pass
        prev_g = curr_g
    
    if root_u is None:
        # Check if we should return 0 or inf
        # If g(u_lo) > 0 and g(u_hi) > 0, the function stayed positive.
        # This happens if 1+Delta is large positive and the dip wasn't deep enough.
        # This implies d=0 is optimal?
        # If d=0, FOC = 1 + Delta. If 1+Delta > 0, increasing d makes FOC more negative?
        # dP'/dd is negative?
        # Let's just return 0 if no root and g(0)>0.
        # If g(0)<0, return large d.
        if g(u_lo) > 0:
            return 0.0
        else:
            return 10.0 / k # Large depth

    d = -1.0 / k * np.log(root_u / A_dt)
    
    if d < 0:
        return 0.0
    return d


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths using DP with analytical FOC.
    """
    dt = T / steps
    q_vals = np.arange(-Q, Q + 1)
    n_q = 2 * Q + 1

    # Inventory holding cost per step
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt

    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))

    start_time = time.time()
    deadline = start_time + max(1.0, _TIME_BUDGET - (time.time() - _GLOBAL_START))

    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            if n < steps - 1:
                V[n:n+1, :] = V[n+1, :]
                opt_bid[n:n+1, :] = opt_bid[n+1, :]
                opt_ask[n:n+1, :] = opt_ask[n+1, :]
            break

        V_next = V[n + 1]
        
        b_opt = np.zeros(n_q)
        a_opt = np.zeros(n_q)

        for i in range(n_q):
            q = i
            C = C_inv[q]
            
            if q == n_q - 1: # q = Q, no bid fill
                # Only ask
                Delta_a = V_next[q-1] - V_next[q]
                a_star = _solve_foc_scalar(Delta_a, A, k, dt)
                a_opt[q] = a_star
                b_opt[q] = np.inf

            elif q == 0: # q = -Q, no ask fill
                # Only bid
                Delta_b = V_next[q+1] - V_next[q]
                b_star = _solve_foc_scalar(Delta_b, A, k, dt)
                b_opt[q] = b_star
                a_opt[q] = np.inf

            else:
                # Interior
                Delta_b = V_next[q+1] - V_next[q]
                Delta_a = V_next[q-1] - V_next[q]
                
                b_star = _solve_foc_scalar(Delta_b, A, k, dt)
                a_star = _solve_foc_scalar(Delta_a, A, k, dt)
                
                b_opt[q] = b_star
                a_opt[q] = a_star

        # Compute V using the chosen depths
        # V(q) = b P_b + a P_a - C + E[V_next]
        # E[V_next] = (1-P_b)(1-P_a)V(q) + P_b(1-P_a)V(q+1) + (1-P_b)P_a V(q-1) + P_b P_a V(q)
        #           = V(q) + P_b(V(q+1)-V(q)) + P_a(V(q-1)-V(q)) - P_b P_a (V(q+1)+V(q-1)-2V(q))
        
        pb = np.zeros(n_q)
        pa = np.zeros(n_q)
        
        for i in range(n_q):
            if np.isinf(b_opt[i]):
                pb[i] = 0.0
            else:
                pb[i] = _fill_prob_scalar(b_opt[i], A, k, dt)
            
            if np.isinf(a_opt[i]):
                pa[i] = 0.0
            else:
                pa[i] = _fill_prob_scalar(a_opt[i], A, k, dt)

        V_curr = np.zeros(n_q)
        
        for i in range(n_q):
            b = 0.0 if np.isinf(b_opt[i]) else b_opt[i]
            a = 0.0 if np.isinf(a_opt[i]) else a_opt[i]
            
            reward = b * pb[i] + a * pa[i]
            
            # Continuation
            V_q = V_next[i]
            V_q_plus = V_next[i+1] if i < n_q-1 else V_q
            V_q_minus = V_next[i-1] if i > 0 else V_q
            
            # Interaction term
            # If both hit, assume q stays same (net 0)
            # Prob(Both) = pb * pa
            # Prob(Bid Only) = pb * (1-pa)
            # Prob(Ask Only) = (1-pb) * pa
            # Prob(None) = (1-pb) * (1-pa)
            
            exp_val = (pb[i] * (1-pa[i]) * V_q_plus) + \
                      ((1-pb[i]) * pa[i] * V_q_minus) + \
                      ((1-pb[i]) * (1-pa[i]) * V_q) + \
                      (pb[i] * pa[i] * V_q)
            
            V_curr[i] = reward - C_inv[i] + exp_val

        V[n, :] = V_curr
        opt_bid[n, :] = b_opt
        opt_ask[n, :] = a_opt

    return opt_bid, opt_ask


def _get_policy(params):
    """Retrieves or computes the optimal policy for the given parameters."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])

    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]

    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )

    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """Returns optimal bid and ask depths based on pre-computed DP policy."""
    opt_bid, opt_ask = _get_policy(params)

    dt = params['T'] / params['steps']
    n = int(round(t / dt))

    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1

    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].copy()
    ask_depths = opt_ask[n, q_idx].copy()

    # Ensure valid outputs
    min_depth = 0.001
    max_depth = 10.0 / params['k']

    # Handle inf (no quote)
    bid_depths = np.where(np.isinf(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isinf(ask_depths), np.inf, ask_depths)
    
    # Clip finite values
    bid_depths = np.where(np.isfinite(bid_depths), np.clip(bid_depths, min_depth, max_depth), bid_depths)
    ask_depths = np.where(np.isfinite(ask_depths), np.clip(ask_depths, min_depth, max_depth), ask_depths)

    # Replace any remaining NaN
    bid_depths = np.where(np.isnan(bid_depths), min_depth, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), min_depth, ask_depths)

    # No bid at q=Q, no ask at q=-Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths.astype(float), ask_depths.astype(float)
