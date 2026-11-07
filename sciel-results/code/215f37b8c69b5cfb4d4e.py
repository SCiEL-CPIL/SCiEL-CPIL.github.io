"""Exact DP lookup for optimal market making with inventory risk."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for computed policies
_policy_cache = {}
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill


def _compute_policy(params):
    """Compute the optimal policy via backward induction (Bellman equation).
    
    Returns (bid_depths_table, ask_depths_table) where each is shape (steps+1, 2*Q+1)
    indexed by [step, inventory_index].
    """
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps
    
    # State space: inventory in [-Q, Q], so 2*Q+1 levels
    n_inv = 2 * Q + 1
    
    # Precompute fill probabilities for a grid of depths
    # We need to find optimal depths. The optimal depth will be in a range we can enumerate.
    # Fill prob: 1 - exp(-A * exp(-k * depth) * dt)
    # Let's precompute for a fine grid of depths
    
    # What range of depths makes sense?
    # depth > 0. If depth is very large, fill prob -> 0.
    # If depth is very small (near 0), fill prob -> 1 - exp(-A*dt) which could be large.
    # But depth must be > 0 (or >= 0). Actually depth can be any positive real.
    # Let's think about the objective: we earn depth when filled, but pay inventory cost.
    # The optimal depth balances expected fill reward against inventory risk.
    
    # For the DP, at each step and inventory level, we choose bid_depth and ask_depth.
    # The value function V(n, q) = max over depths of [expected continuation + immediate reward]
    
    # Let's discretize depth. What's a good range?
    # At k=1.5, depth 1/k = 0.667. At k=0.75, depth 1/k = 1.333.
    # Let's use a range from a small epsilon to some max.
    # Actually, let's think more carefully. The fill probability is:
    # p = 1 - exp(-A * exp(-k*d) * dt)
    # For the regimes:
    # Regime 1: A=140, k=1.5, dt=0.001 -> A*dt=0.14, so p = 1-exp(-0.14*exp(-1.5*d))
    # Regime 2: same A, k but gamma=0.5
    # Regime 3: A=50, k=0.75, dt=0.001 -> A*dt=0.05, p = 1-exp(-0.05*exp(-0.75*d))
    
    # The depth that maximizes expected reward minus risk... let's just use a fine grid.
    # Let's say depth in [0.001, 5.0] with fine resolution.
    
    # Actually, for exactness, let's think about what depths are optimal.
    # The problem is separable in bid and ask? No, because they share inventory.
    # But given inventory q, the bid depth and ask depth choices are coupled through the
    # next-state distribution.
    
    # Let me set up the DP properly.
    # V(n, q) = max_{d_b, d_a} E[ reward + V(n+1, q') ]
    # where reward = d_b * I(bid fill) + d_a * I(ask fill) - (gamma/2)*sigma^2*q^2*dt
    # and q' = q + I(bid fill) - I(ask fill)  (bid fill increases inventory, ask fill decreases)
    
    # Wait, let me re-read: "A fill earns its depth and moves inventory one unit"
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # So if bid fills, inventory increases by 1 (q -> q+1), and we earn bid_depth.
    # If ask fills, inventory decreases by 1 (q -> q-1), and we earn ask_depth.
    
    # The mid moves but that doesn't affect the depth-based reward directly since depths
    # are measured relative to mid.
    
    # So the transition is:
    # p_b = 1 - exp(-A * exp(-k * d_b) * dt)  [if q < Q, else 0]
    # p_a = 1 - exp(-A * exp(-k * d_a) * dt)  [if q > -Q, else 0]
    # These are independent.
    
    # q' = q + 1 with prob p_b*(1-p_a)  [bid fills, ask doesn't]
    # q' = q - 1 with prob (1-p_b)*p_a  [ask fills, bid doesn't]
    # q' = q + 0 with prob (1-p_b)*(1-p_a) + p_b*p_a  [both or neither]
    # Wait, both fill: q' = q + 1 - 1 = q. Neither: q' = q.
    
    # Immediate reward: d_b * I(bid) + d_a * I(ask) - (gamma/2)*sigma^2*q^2*dt
    # E[reward] = d_b * p_b + d_a * p_a - (gamma/2)*sigma^2*q^2*dt
    
    # V(n, q) = max_{d_b, d_a} { d_b*p_b + d_a*p_a - (gamma/2)*sigma^2*q^2*dt 
    #             + (1-p_b)*(1-p_a)*V(n+1, q) 
    #             + p_b*(1-p_a)*V(n+1, q+1)   [if q < Q]
    #             + (1-p_b)*p_a*V(n+1, q-1)   [if q > -Q]
    #             + p_b*p_a*V(n+1, q) }
    
    # Simplify the V terms:
    # = d_b*p_b + d_a*p_a - (gamma/2)*sigma^2*q^2*dt
    #   + V(n+1,q)*[(1-p_b)(1-p_a) + p_b*p_a] + V(n+1,q+1)*p_b*(1-p_a) + V(n+1,q-1)*(1-p_b)*p_a
    
    # This is a 2D maximization over (d_b, d_a) for each (n, q). That's expensive.
    # But note: the objective is separable-ish. Let me check if we can separate.
    
    # Actually, let's define:
    # f(d_b, d_a) = d_b*p_b(d_b) + d_a*p_a(d_a) - c 
    #   + V_{q+1}*p_b(d_b)*(1-p_a(d_a)) + V_{q-1}*(1-p_b(d_b))*p_a(d_a) + V_q*[(1-p_b)(1-p_a)+p_b*p_a]
    
    # where c = (gamma/2)*sigma^2*q^2*dt, V_{q+1}=V(n+1,q+1), etc.
    
    # This is not trivially separable because of the cross terms p_b*(1-p_a), etc.
    
    # However, for a fine grid, we can just do a 2D grid search.
    # n_inv = 61, steps = 1000. If we use a grid of, say, 50 depths for each side,
    # that's 2500 evaluations per (n,q) pair, times 61*1000 = 61000 pairs -> 152.5M evals.
    # That might be too slow in Python. Let me vectorize with NumPy.
    
    # Let's use a depth grid. What range?
    # For the regimes, 1/k ranges from 0.667 to 1.333. Let's use [0.01, 3.0] with 200 points.
    # Actually, let's be smarter. Let's use a coarser grid first and refine.
    
    # Let me use 100 depth points for each side. 100*100 = 10000 per (n,q).
    # 61*1000*10000 = 610M. Still too much in pure Python loops.
    
    # Let me vectorize over the depth grid for each (n,q).
    
    # Actually, let me reconsider. The key insight from the knowledge is that exact DP
    # achieves 1.347. Let me implement it efficiently.
    
    # Let's use a moderate grid. Say 50x50 = 2500 per (n,q).
    # Vectorize over all inventory levels for a given step.
    
    n_depth = 60
    depth_min = 0.01
    depth_max = 3.0
    depths = np.linspace(depth_min, depth_max, n_depth)
    
    # Precompute fill probabilities for each depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    fill_probs = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # V will be shape (steps+1, n_inv)
    V = np.zeros((steps + 1, n_inv))
    
    # Policy tables
    bid_policy = np.zeros((steps + 1, n_inv))  # index into depths
    ask_policy = np.zeros((steps + 1, n_inv))
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
        
        # For each inventory level, compute optimal (d_b, d_a)
        # Vectorize over inventory levels
        # V_next[q_idx] = V(n+1, q_idx)
        V_next = V[n + 1, :]
        
        # For each inventory index qi (0..n_inv-1), q = qi - Q
        # We need V(n+1, q-1), V(n+1, q), V(n+1, q+1)
        # q-1 corresponds to index qi-1, q+1 to qi+1
        # Boundary: if q = Q (qi = 2Q), no bid fill possible
        #            if q = -Q (qi = 0), no ask fill possible
        
        # Let's compute for all qi at once using vectorization
        # For each qi, the objective over (db_idx, da_idx) is:
        # db[db_idx]*pb[db_idx] + da[da_idx]*pa[da_idx] - c[qi]
        # + V_next[qi]*[(1-pb)(1-pa)+pb*pa]
        # + V_next[qi+1]*pb*(1-pa)  [if qi < 2Q]
        # + V_next[qi-1]*(1-pb)*pa  [if qi > 0]
        
        # This is complex to fully vectorize in 2D. Let me do it per qi but vectorize the depth grid.
        
        for qi in range(n_inv):
            q = qi - Q
            
            # Inventory cost
            c = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get next-state values
            V_q = V_next[qi]
            V_qp1 = V_next[qi + 1] if qi < 2 * Q else 0.0  # No bid fill at Q
            V_qm1 = V_next[qi - 1] if qi > 0 else 0.0  # No ask fill at -Q
            
            # If q = Q, bid fill is impossible, so we only optimize ask_depth
            # If q = -Q, ask fill is impossible, so we only optimize bid_depth
            
            if qi == 2 * Q:
                # Only optimize ask_depth
                # p_b = 0 always
                # Objective: da*pa - c + V_q*[(1-pa)] + V_qm1*pa
                # = da*pa - c + V_q - V_q*pa + V_qm1*pa
                # = da*pa - c + V_q + pa*(V_qm1 - V_q)
                obj = depths * fill_probs - c + V_q + fill_probs * (V_qm1 - V_q)
                best_idx = np.argmax(obj)
                bid_policy[n, qi] = depths[0]  # dummy, won't be used
                ask_policy[n, qi] = depths[best_idx]
            elif qi == 0:
                # Only optimize bid_depth
                # p_a = 0 always
                # Objective: db*pb - c + V_q*[(1-pb)] + V_qp1*pb
                # = db*pb - c + V_q - V_q*pb + V_qp1*pb
                # = db*pb - c + V_q + pb*(V_qp1 - V_q)
                obj = depths * fill_probs - c + V_q + fill_probs * (V_qp1 - V_q)
                best_idx = np.argmax(obj)
                bid_policy[n, qi] = depths[best_idx]
                ask_policy[n, qi] = depths[0]  # dummy
            else:
                # Both sides active
                # Full 2D grid
                # pb = fill_probs[:, None]  shape (n_depth, 1)
                # pa = fill_probs[None, :]  shape (1, n_depth)
                # db = depths[:, None]
                # da = depths[None, :]
                
                pb = fill_probs[:, None]
                pa = fill_probs[None, :]
                db = depths[:, None]
                da = depths[None, :]
                
                # Objective matrix
                obj = db * pb + da * pa - c
                obj += V_q * ((1 - pb) * (1 - pa) + pb * pa)
                obj += V_qp1 * pb * (1 - pa)
                obj += V_qm1 * (1 - pb) * pa
                
                # Find max
                max_flat = np.argmax(obj)
                best_db_idx, best_da_idx = np.unravel_index(max_flat, obj.shape)
                bid_policy[n, qi] = depths[best_db_idx]
                ask_policy[n, qi] = depths[best_da_idx]
        
        # Compute V[n, qi] for all qi
        for qi in range(n_inv):
            q = qi - Q
            c = 0.5 * gamma * sigma**2 * q**2 * dt
            V_q = V_next[qi]
            V_qp1 = V_next[qi + 1] if qi < 2 * Q else 0.0
            V_qm1 = V_next[qi - 1] if qi > 0 else 0.0
            
            db = bid_policy[n, qi]
            da = ask_policy[n, qi]
            
            pb = 1.0 - np.exp(-A * np.exp(-k * db) * dt) if qi < 2 * Q else 0.0
            pa = 1.0 - np.exp(-A * np.exp(-k * da) * dt) if qi > 0 else 0.0
            
            V[n, qi] = db * pb + da * pa - c \
                + V_q * ((1 - pb) * (1 - pa) + pb * pa) \
                + V_qp1 * pb * (1 - pa) \
                + V_qm1 * (1 - pb) * pa
    
    # Convert policy to depth arrays (already in depth units)
    # bid_policy and ask_policy are already depths
    return bid_policy, ask_policy


def _get_or_compute_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    bid_policy, ask_policy = _get_or_compute_policy(params)
    
    Q = params['Q']
    steps = params['steps']
    
    # Determine current step index
    T = params['T']
    dt = T / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)  # Clamp to valid range
    
    # inventory is in [-Q, Q], convert to index [0, 2Q]
    inv_idx = inventory + Q
    
    # Look up depths
    bid_depths = bid_policy[n, inv_idx]
    ask_depths = ask_policy[n, inv_idx]
    
    # Ensure no bid at Q, no ask at -Q (should already be handled in DP)
    # But let's be safe: set to np.inf if at boundary
    at_Q = (inventory == Q)
    at_mQ = (inventory == -Q)
    bid_depths[at_Q] = np.inf
    ask_depths[at_mQ] = np.inf
    
    # Ensure depths are valid: > -3/k, not -inf, not NaN
    k = params['k']
    min_valid = -3.0 / k
    
    # Replace invalid values
    invalid_bid = np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_valid)
    invalid_ask = np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_valid)
    
    if np.any(invalid_bid):
        bid_depths = np.where(invalid_bid, 1.0/k, bid_depths)
    if np.any(invalid_ask):
        ask_depths = np.where(invalid_ask, 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
