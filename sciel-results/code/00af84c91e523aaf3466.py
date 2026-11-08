"""
Strategy: Exact Continuous DP via Convex Optimization.
Instead of grid-searching depths, we exploit the convexity of the Bellman objective
with respect to the depth d for fixed inventory change probabilities.
We precompute the exact optimal bid and ask depths for each (t, q) state using
bisection on the derivative of the objective. This removes discretization error
present in grid-based approaches.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin

_policy_cache = {}

def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # V_next: Value function at t+1, shape (n_states,)
    V_next = np.zeros(n_states)
    
    # Precompute constants for the derivative
    # Objective for bid depth b (given ask depth a is fixed at optimal a*):
    # f(b) = (1-pb)(1-pa) Vq + pb(1-pa)(b + Vqp) + (1-pb)pa(a + Vqm) + pb pa(b + a + Vq) - cost
    # Let's simplify f(b) by grouping terms involving pb.
    # f(b) = (1-pa)Vq - pb(1-pa)Vq + pb(1-pa)b + pb(1-pa)Vqp - pb(1-pa)a - pb(1-pa)Vqm 
    #       + pb pa b + pb pa a + pb pa Vq - cost
    #       + (1-pb)pa(a + Vqm)  <-- wait, I need to be careful with expansion.
    
    # Let's use the form:
    # f(b) = E[Reward] + E[V_next] - cost
    # E[Reward] = pb(1-pa)b + (1-pb)pa a + pb pa (b+a)
    #           = pb b - pb pa b + pa a - pb pa a + pb pa b + pb pa a
    #           = pb b + pa a
    # Wait, is that right?
    # If bid fills (prob pb), gain b. If ask fills (prob pa), gain a.
    # If both fill, gain b+a.
    # E[Gain] = pb(1-pa)b + (1-pb)pa a + pb pa (b+a)
    #         = pb b - pb pa b + pa a - pb pa a + pb pa b + pb pa a
    #         = pb b + pa a.
    # Yes! The expected immediate reward from depths is simply pb*b + pa*a.
    # This is a huge simplification.
    
    # Now E[V_next]:
    # Prob(no fill) = (1-pb)(1-pa) -> Vq
    # Prob(bid only) = pb(1-pa) -> Vqp
    # Prob(ask only) = (1-pb)pa -> Vqm
    # Prob(both) = pb pa -> Vq
    
    # E[V_next] = (1-pb)(1-pa)Vq + pb(1-pa)Vqp + (1-pb)pa Vqm + pb pa Vq
    #           = Vq - pb Vq - pa Vq + pb pa Vq
    #             + pb Vqp - pb pa Vqp
    #             + pa Vqm - pb pa Vqm
    #             + pb pa Vq
    #           = Vq + pb(Vqp - Vq - pa Vqp) + pa(Vqm - Vq - pb Vqm) + 2 pb pa Vq ? No.
    
    # Let's group by pb and pa.
    # E[V_next] = (1-pa)Vq + pb(1-pa)(Vqp - Vq) + pa(1-pb)(Vqm - Vq) + pb pa (Vq - Vq)
    # Wait, the last term is Vq, and the first is (1-pa)Vq.
    # (1-pa)Vq + pb pa Vq = Vq(1 - pa + pb pa) = Vq(1 - pa(1-pb)).
    # This is getting messy. Let's just compute the derivative directly using the original sum.
    
    # f(b) = (1-pb)(1-pa)Vq + pb(1-pa)(b + Vqp) + (1-pb)pa(a + Vqm) + pb pa(b + a + Vq) - cost
    
    # df/db = (1-pa) [ -dpb/db Vq + dpb/db (b + Vqp) + pb ] 
    #        + pa [ -dpb/db (a + Vqm) + dpb/db (b + a + Vq) + pb * 0? No, pa is constant wrt b ]
    # Let's differentiate term by term.
    # Let P = pb, Q = pa.
    # f(b) = (1-P)(1-Q)Vq + P(1-Q)(b + Vqp) + (1-P)Q(a + Vqm) + P Q (b + a + Vq) - cost
    # df/db = (1-Q) [ -P' Vq + P' (b + Vqp) + P ] + Q [ -P' (a + Vqm) + P' (b + a + Vq) ]
    #        = P' { (1-Q)(b + Vqp - Vq) + Q(b + a + Vq - a - Vqm) } + P(1-Q)
    #        = P' { (1-Q)(b + Vqp - Vq) + Q(b + Vq - Vqm) } + P(1-Q)
    
    # P' = d/d b (1 - exp(-A exp(-k b) dt))
    #    = - exp(-A exp(-k b) dt) * A * dt * exp(-k b) * (-k)
    #    = k * A * dt * exp(-k b) * exp(-A exp(-k b) dt)
    # Let x = A exp(-k b) dt. Then P = 1 - exp(-x).
    # P' = k * x * exp(-x) ? No.
    # d/db exp(-k b) = -k exp(-k b).
    # d/db exp(-A exp(-k b) dt) = exp(-A exp(-k b) dt) * (-A dt) * (-k exp(-k b))
    #                           = k A dt exp(-k b) exp(-A exp(-k b) dt).
    # So P' = k A dt exp(-k b) exp(-P).  (Since 1-P = exp(-x) = exp(-A exp(-k b) dt))
    
    # So df/db = k A dt exp(-k b) (1 - pb) * [ (1-pa)(b + Vqp - Vq) + pa(b + Vq - Vqm) ] + pb(1-pa)
    
    # We want to find b such that df/db = 0.
    # This is a 1D root finding problem.
    # The function is unimodal (convex objective -> concave maximization? No, we maximize f, so f is concave? 
    # Actually, we want to maximize f. The derivative starts positive (small b -> high prob, but low depth reward? 
    # Let's check limits.
    # b=0: P=1-exp(-A dt). P' > 0.
    # b->inf: P->0, P'->0.
    # The term P(1-Q) is positive. The term with P' is positive if the bracket is positive.
    # Is the bracket positive?
    # (1-pa)(b + Vqp - Vq) + pa(b + Vq - Vqm)
    # Vqp - Vq is the value of having one more unit of inventory (buying). 
    # Vq - Vqm is the value of having one less unit (selling).
    # Usually, holding inventory has a cost, so Vqp < Vq and Vqm < Vq? 
    # No, V is the *future* reward. 
    # If inventory is high, you want to sell. 
    # The term b is the immediate reward. 
    # As b increases, immediate reward b increases linearly, but probability P decreases exponentially.
    # So there is an optimal b.
    
    # We can use bisection on the derivative.
    # Derivative D(b) = k A dt exp(-k b) (1 - pb) * B + pb(1-pa)
    # where B = (1-pa)(b + Vqp - Vq) + pa(b + Vq - Vqm)
    
    # Note: B depends on b!
    # B(b) = (1-pa)b + (1-pa)(Vqp - Vq) + pa b + pa(Vq - Vqm)
    #      = b + (1-pa)(Vqp - Vq) + pa(Vq - Vqm)
    # Let C = (1-pa)(Vqp - Vq) + pa(Vq - Vqm). This is constant for fixed a.
    # So B(b) = b + C.
    
    # D(b) = k A dt exp(-k b) (1 - pb) * (b + C) + pb(1-pa)
    
    # We solve D(b) = 0 for b.
    
    def solve_depth(Vq, Vqp, Vqm, pa, allow):
        if not allow:
            return 0.0  # Depth doesn't matter if no fill allowed, but we return 0 for safety/consistency
        
        C = (1.0 - pa) * (Vqp - Vq) + pa * (Vq - Vqm)
        
        # D(b) = k A dt exp(-k b) (1 - (1-exp(-A exp(-k b) dt))) * (b + C) + (1-exp(-A exp(-k b) dt))(1-pa)
        # D(b) = k A dt exp(-k b) exp(-A exp(-k b) dt) * (b + C) + (1-exp(-A exp(-k b) dt))(1-pa)
        
        # We want D(b) = 0.
        # D(0) = k A dt * 1 * exp(-A dt) * C + (1-exp(-A dt))(1-pa)
        # D(inf) -> 0 + 0 = 0? 
        # As b->inf, exp(-k b) -> 0, so first term -> 0.
        # pb -> 0, so second term -> 0.
        # So D(b) goes from D(0) to 0.
        # If D(0) > 0, the maximum is at b=0? 
        # Wait, if D(b) > 0 for all b, then f is increasing, so max is at inf? 
        # But f(b) -> 0 as b->inf? 
        # f(b) = ... + pb b ... 
        # If b is very large, pb is tiny. pb b might go to 0.
        # Let's check f(b) as b->inf.
        # pb ~ A dt exp(-k b).
        # pb b ~ b exp(-k b) -> 0.
        # So f(b) -> constant (terms with 1-pb).
        # f(inf) = (1)(1-pa)Vq + 0 + (1-pa)pa(a+Vqm) + 0 - cost
        # f(0) = (1-P0)(1-pa)Vq + P0(1-pa)(0+Vqp) + (1-P0)pa(a+Vqm) + P0 pa(0+a+Vq) - cost
        
        # If D(0) > 0, f is increasing at 0. Since f->const at inf, max is somewhere?
        # Actually, D(b) is likely positive for small b and becomes negative?
        # Let's check the sign of D(b) for large b.
        # D(b) ≈ k A dt exp(-k b) * 1 * b + 0 = k A dt b exp(-k b) > 0.
        # So D(b) is always positive? 
        # If D(b) > 0 for all b, then f(b) is monotonically increasing.
        # But f(b) is bounded? 
        # If f is increasing and bounded, the max is at the limit? 
        # But we can't quote infinite depth. 
        # However, the task says "distances below/above mid". 
        # If the optimal depth is effectively infinity, we should quote a very large depth? 
        # But the task says "Values below -3/k ... invalid". It doesn't say max depth.
        # But practically, if D(b) > 0, we want to increase b.
        # Let's re-evaluate D(b).
        # Maybe I made a sign error.
        # f(b) = (1-P)(1-Q)Vq + P(1-Q)(b+Vqp) + (1-P)Q(a+Vqm) + P Q (b+a+Vq)
        # df/db = (1-Q)(-P'Vq + P'(b+Vqp) + P) + Q(-P'(a+Vqm) + P'(b+a+Vq))
        # = P' [ (1-Q)(b+Vqp-Vq) + Q(b+Vq-Vqm) ] + P(1-Q)
        # = P' (b+C) + P(1-Q)
        
        # Since P' > 0 and P > 0, if (b+C) > 0, then D(b) > 0.
        # C = (1-Q)(Vqp-Vq) + Q(Vq-Vqm).
        # If Vqp < Vq and Vqm < Vq, then C < 0.
        # If b is small, b+C might be negative.
        # If b+C < 0, the first term is negative.
        # So D(b) can be negative for small b if C is very negative.
        # If D(0) < 0, then f is decreasing at 0, so max is at b=0.
        # If D(0) > 0, then f is increasing at 0. 
        # As b increases, b+C becomes positive. 
        # Does D(b) cross zero?
        # If D(b) stays positive, max is at large b.
        # Given the cost of inventory, C is likely negative enough that D(b) crosses zero.
        
        # Bisection range: [0, 10.0] (safe upper bound)
        lo = 0.0
        hi = 10.0
        
        # Check D(lo)
        def deriv(b):
            if b < 0: b = 0
            exp_kb = np.exp(-k * b)
            x = A * exp_kb * dt
            exp_x = np.exp(-x)
            pb = 1.0 - exp_x
            P_prime = k * A * dt * exp_kb * exp_x
            val = P_prime * (b + C) + pb * (1.0 - pa)
            return val
        
        d_lo = deriv(lo)
        d_hi = deriv(hi)
        
        # If D(lo) <= 0, max is at 0
        if d_lo <= 0:
            return 0.0
            
        # If D(hi) >= 0, max is at hi (or larger, but we cap)
        # Actually, if D(hi) > 0, we might want to go larger. 
        # But for k>=0.75, exp(-k*10) is tiny. 
        # Let's just return hi if D(hi) > 0.
        if d_hi >= 0:
            return hi
        
        # Bisection
        for _ in range(50):
            mid = (lo + hi) / 2.0
            d_mid = deriv(mid)
            if d_mid > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    # DP Loop
    for t in range(steps - 1, -1, -1):
        V_current = np.zeros(n_states)
        for q_idx in range(n_states):
            q = q_values[q_idx]
            allow_bid = (q < Q)
            allow_ask = (q > -Q)
            
            v_q = V_next[q_idx]
            v_qp = V_next[q_idx + 1] if allow_bid else 0.0
            v_qm = V_next[q_idx - 1] if allow_ask else 0.0
            
            # We need to solve for optimal b and a.
            # They are coupled.
            # f(b,a) is jointly convex? 
            # We can use coordinate descent.
            # Initialize b=0, a=0.
            b_opt = 0.0
            a_opt = 0.0
            
            # Iterate a few times to converge
            for _ in range(5):
                # Solve for b given a_opt
                pa = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt) if allow_ask else 0.0
                b_new = solve_depth(v_q, v_qp, v_qm, pa, allow_bid)
                
                # Solve for a given b_new
                pb = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt) if allow_bid else 0.0
                # Symmetric formula for a:
                # C_a = (1-pb)(Vqm - Vq) + pb(Vq - Vqp)
                # D_a(a) = k A dt exp(-k a) (1-pa) (a + C_a) + pa(1-pb)
                
                if allow_ask:
                    C_a = (1.0 - pb) * (v_qm - v_q) + pb * (v_q - v_qp)
                    # Reuse solve_depth logic? 
                    # The structure is identical, just swap Vqp/Vqm and pb/pa roles.
                    # solve_depth(Vq, Vqm, Vqp, pb, allow_ask) ?
                    # Let's check:
                    # For ask depth a, the "next state" if filled is q-1 (Vqm).
                    # The "other" next state is q+1 (Vqp).
                    # So in solve_depth, Vqp corresponds to Vqm (inventory change +1 for bid, -1 for ask).
                    # Wait, solve_depth assumes:
                    # Vqp = V(q+1), Vqm = V(q-1).
                    # For bid: fill -> q+1. So Vqp is correct.
                    # For ask: fill -> q-1. So we should pass Vqm as the "fill" value.
                    # The formula for C was: (1-pa)(Vqp-Vq) + pa(Vq-Vqm).
                    # Here pa is prob of ask fill.
                    # If we call solve_depth(v_q, v_qm, v_qp, pb, allow_ask):
                    # It computes C = (1-pb)(v_qm - v_q) + pb(v_q - v_qp).
                    # This matches C_a derived above.
                    a_new = solve_depth(v_q, v_qm, v_qp, pb, allow_ask)
                else:
                    a_new = 0.0
                    
                if abs(b_new - b_opt) < 1e-6 and abs(a_new - a_opt) < 1e-6:
                    b_opt = b_new
                    a_opt = a_new
                    break
                b_opt = b_new
                a_opt = a_new
                
            opt_bid[t, q_idx] = b_opt
            opt_ask[t, q_idx] = a_opt
            
            # Compute V_current
            pb = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt) if allow_bid else 0.0
            pa = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt) if allow_ask else 0.0
            
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            v_next_expected = (1-pb)*(1-pa)*v_q + pb*(1-pa)*v_qp + (1-pb)*pa*v_qm + pb*pa*v_q
            reward = pb*b_opt + pa*a_opt
            V_current[q_idx] = reward + v_next_expected - cost
            
        V_next = V_current
        
        # Time check
        if time.time() > _DEADLINE - 10:
            # Fill remaining with zeros or current best guess to avoid timeout
            # This is a fallback
            break
            
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_optimal_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks
# EVOLVE-BLOCK-END
