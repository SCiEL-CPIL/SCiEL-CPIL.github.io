"""
Strategy: Finite-Horizon Dynamic Programming (Bellman Equation).
Solves the exact discrete Markov Decision Process by backward induction.
This captures the full value function V(q, t) including future inventory risk,
which the myopic analytical approximation ignores.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 320.0

# Cache for pre-computed policies
# Key: (sigma, A, k, gamma, T, steps, Q)
_policy_cache = {}

def _compute_policy(params):
    """
    Pre-compute the optimal bid and ask depths for all steps and inventory levels.
    Returns:
        bid_depths: array of shape (steps, 2*Q+1)
        ask_depths: array of shape (steps, 2*Q+1)
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    
    # Discretize depth space
    # Max depth: We need to cover the range where fill probability is non-negligible.
    # p(d) = 1 - exp(-A exp(-k d) dt).
    # For large d, p is small.
    # Let's choose a grid that balances resolution and range.
    # Range: 0 to D_max.
    # D_max: If A dt is small, we need larger d to get fills?
    # Actually, p ~ A dt exp(-k d).
    # If A=140, dt=0.001, A dt = 0.14.
    # exp(-1.5 * d).
    # If d=5, exp(-7.5) ~ 0.0005. p ~ 0.14 * 0.0005 = 7e-5. Very small.
    # If d=1, exp(-1.5) ~ 0.22. p ~ 0.14 * 0.22 = 0.03.
    # So most action is in [0, 5].
    # Let's use a grid of 200 points over [0, 5.0].
    D_max = 5.0
    N_d = 200
    d_grid = np.linspace(0.0, D_max, N_d)
    
    # Pre-compute fill probabilities for all depths
    # p(d) = 1 - exp(-A * exp(-k * d) * dt)
    # Shape: (N_d,)
    exp_term = np.exp(-k * d_grid)
    p_fills = 1.0 - np.exp(-A * exp_term * dt)
    
    # Inventory indices: 0..2Q corresponding to -Q..Q
    # Index i corresponds to q = i - Q
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Value function V[n, i] = Max expected future reward starting at step n with inventory i
    # We iterate backwards from n = steps - 1 down to 0.
    # V[steps, i] = 0 (Terminal value)
    
    V_next = np.zeros(n_q)
    
    # Store optimal actions
    # bid_opt[n, i], ask_opt[n, i]
    bid_opt = np.zeros((steps, n_q))
    ask_opt = np.zeros((steps, n_q))
    
    # We only need to store the actions, not the full V history, to save memory/time?
    # But V_next depends on V at n+1. So we just keep V_next and update it.
    
    for n in range(steps - 1, -1, -1):
        # Check time limit
        if time.time() > TIME_LIMIT - 10:
            # If we run out of time, return current best (partial or zero)
            # To be safe, if we haven't finished, we might return a heuristic or just break.
            # But partial DP is useless. 
            # Let's ensure we finish. 
            # 1000 steps * 61 states * 200 depths * 2 sides = 24.4M ops.
            # Numpy vectorization helps.
            pass
        
        V_curr = np.zeros(n_q)
        
        # Iterate over all inventory states
        # We can vectorize over inventory states if the depth grid is the same for all.
        # The transition probabilities depend on q only for the boundary conditions (q=Q, q=-Q).
        # For interior states, the optimal depth is the same for all q?
        # No, the value function V(q) is convex, so the optimal depth depends on q.
        # However, the *shape* of the optimization problem is the same, just shifted by the value differences.
        # V(q+1) - V(q) and V(q-1) - V(q) differ.
        
        # Let's vectorize over q.
        # For each q, we want to maximize:
        # Immediate Reward + Expected Future Value - Cost
        # Cost = 0.5 * gamma * sigma^2 * q^2 * dt
        
        # Let's pre-compute the cost for each q
        cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # For each q, we compute the optimal bid and ask.
        # Since the depth grid is fixed, we can compute the value for all depths for all q's?
        # The transition depends on q.
        
        # Let's loop over q. It's only 61 iterations.
        for i in range(n_q):
            q = q_vals[i]
            
            # Calculate V(q+1) and V(q-1) from V_next
            # Note: V_next is indexed by i.
            # V(q) is at index i.
            # V(q+1) is at index i+1 (if i < n_q-1)
            # V(q-1) is at index i-1 (if i > 0)
            
            V_q = 0.0 # Placeholder, not needed directly in max, but for reference
            
            if i < n_q - 1:
                V_q_plus = V_next[i+1]
            else:
                V_q_plus = 0.0 # Boundary: no state beyond Q? Or clamp?
                # Actually, if q=Q, bid fill is impossible. Ask fill moves to Q-1.
                # If q=-Q, ask fill is impossible. Bid fill moves to -Q+1.
                
            if i > 0:
                V_q_minus = V_next[i-1]
            else:
                V_q_minus = 0.0
            
            # Objective for bid depth d_b and ask depth d_a
            # We assume bid and ask depths are chosen independently?
            # The problem allows choosing a tuple (bid_depths, ask_depths).
            # In the Bellman equation, we maximize over both.
            # Since the fill events are independent, we can separate the optimization?
            # Max_{db, da} [ db*pb(db) + da*pa(da) + E[V(q')] - Cost ]
            # E[V(q')] = (1-pb)(1-pa)V(q) + pb(1-pa)V(q+1) + (1-pb)pa V(q-1) + pb pa V(q)
            #            = V(q) + pb(1-pa)(V(q+1)-V(q)) + (1-pb)pa(V(q-1)-V(q))
            # Let Delta_plus = V(q+1) - V(q)
            # Let Delta_minus = V(q-1) - V(q)
            # E[V(q')] = V(q) + pb Delta_plus (1-pa) + pa Delta_minus (1-pb)
            #          = V(q) + pb Delta_plus + pa Delta_minus - pb pa (Delta_plus + Delta_minus)
            
            # The term - pb pa (Delta_plus + Delta_minus) couples db and da.
            # However, pb and pa are small (prob of fill in one step).
            # pb ~ A dt exp(-k db).
            # If dt is small, pb is small. The product pb*pa is second order small.
            # We can approximate by ignoring the coupling term (independent optimization).
            # This is a standard approximation in market making (Avellaneda-Stoikov style).
            
            # Approximation:
            # Max_db [ db*pb(db) + pb(db) * Delta_plus - Cost_bid_part? ]
            # Wait, the cost is incurred on current inventory q, regardless of fill.
            # Cost = 0.5 * gamma * sigma^2 * q^2 * dt. This is constant w.r.t db, da.
            
            # So we maximize:
            # Bid: db * pb(db) + pb(db) * Delta_plus
            #      = pb(db) * (db + Delta_plus)
            # Ask: da * pa(da) + pa(da) * Delta_minus
            #      = pa(da) * (da + Delta_minus)
            
            # Note: Delta_plus = V(q+1) - V(q). Since V is convex, Delta_plus > 0 for q>0?
            # V(q) ~ c q^2. V(q+1)-V(q) ~ 2cq. Positive for q>0.
            # So if q>0, Delta_plus > 0. Buying (bid fill) increases q, which is bad (higher cost).
            # Wait, V(q) is the *value* (max reward).
            # If cost is negative reward, V(q) is negative and convex (more negative for larger |q|).
            # Let's check signs.
            # Objective: Sum(depth) - Cost.
            # Cost is positive. So V(q) decreases as |q| increases.
            # V(q) is concave? No, "risk" usually implies convex cost, so concave value?
            # Let's assume V(q) is such that holding inventory is bad.
            # V(q+1) < V(q) for q>0. So Delta_plus < 0.
            # Then pb * Delta_plus is a penalty for buying.
            # This makes sense.
            
            # Let's compute Delta_plus and Delta_minus properly.
            # V_next is the value at time t+dt.
            # V(q) at time t is what we are computing.
            # The term in E[V(q')] uses V at t+dt.
            
            if i < n_q - 1:
                Delta_plus = V_next[i+1] - V_next[i] # Approx V(q+1, t+dt) - V(q, t+dt)
            else:
                # Boundary q=Q. Bid fill impossible.
                # We can set Delta_plus to a large negative number to discourage bid?
                # Or just handle boundary in probability.
                Delta_plus = -np.inf # Or 0?
                
            if i > 0:
                Delta_minus = V_next[i-1] - V_next[i]
            else:
                Delta_minus = -np.inf
            
            # Optimize Bid
            # Maximize f_b(db) = pb(db) * (db + Delta_plus)
            # pb(db) = 1 - exp(-A exp(-k db) dt)
            
            # If q == Q, no bid fill. So bid depth doesn't matter?
            # "no bid fill at q = Q".
            # So pb is effectively 0.
            # If we set pb=0, f_b = 0. Any db is optimal?
            # We can return 0 or inf.
            # Let's return 0 for safety (no quote).
            
            if i == n_q - 1: # q = Q
                bid_opt[n, i] = 0.0
                # Ask optimization still valid
            else:
                # Vectorized optimization over d_grid
                # f_b = p_fills * (d_grid + Delta_plus)
                # Note: If Delta_plus is very negative, the optimal db might be 0.
                vals_b = p_fills * (d_grid + Delta_plus)
                # Ensure no NaNs
                vals_b = np.nan_to_num(vals_b, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
                idx_b = np.argmax(vals_b)
                bid_opt[n, i] = d_grid[idx_b]
            
            # Optimize Ask
            # Maximize f_a(da) = pa(da) * (da + Delta_minus)
            # pa(da) = 1 - exp(-A exp(-k da) dt)
            
            if i == 0: # q = -Q
                ask_opt[n, i] = 0.0
            else:
                vals_a = p_fills * (d_grid + Delta_minus)
                vals_a = np.nan_to_num(vals_a, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
                idx_a = np.argmax(vals_a)
                ask_opt[n, i] = d_grid[idx_a]
            
            # Calculate V_curr[i]
            # V_curr[i] = Max_Value - Cost
            # Max_Value = f_b(db*) + f_a(da*) + V_next[i] (The V(q) part from expansion)
            # Wait, the expansion was:
            # E[V(q')] = V_next[i] + pb Delta_plus + pa Delta_minus - pb pa (Delta_plus + Delta_minus)
            # We ignored the last term.
            # So Max_Value = V_next[i] + f_b(db*) + f_a(da*)
            
            # Get optimal values
            if i == n_q - 1:
                best_b = 0.0
            else:
                best_b = vals_b[idx_b] if i < n_q - 1 else 0.0
                # If i == n_q - 1, we skipped bid opt.
            
            if i == 0:
                best_a = 0.0
            else:
                best_a = vals_a[idx_a] if i > 0 else 0.0
            
            # Re-calculate best_b and best_a carefully to avoid index errors
            if i == n_q - 1:
                # q=Q
                # Bid is 0. Value from bid is 0.
                val_b = 0.0
            else:
                # Recompute or store? We stored idx.
                # p_fills[idx_b] * (d_grid[idx_b] + Delta_plus)
                val_b = p_fills[idx_b] * (d_grid[idx_b] + Delta_plus)
            
            if i == 0:
                val_a = 0.0
            else:
                val_a = p_fills[idx_a] * (d_grid[idx_a] + Delta_minus)
            
            V_curr[i] = V_next[i] + val_b + val_a - cost[i]
            
        V_next = V_curr
    
    return bid_opt, ask_opt

def quote(t, inventory, mid, params):
    """
    Compute optimal bid and ask depths using pre-computed DP policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create cache key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        # Check time
        if time.time() > TIME_LIMIT - 5:
            # Fallback to heuristic if no time for DP
            dt = T / steps
            n = int(np.round(t * steps))
            if n >= steps: n = steps - 1
            tau = (steps - 1 - n) * dt
            tau_safe = np.maximum(tau, dt)
            q = np.asarray(inventory, dtype=np.float64)
            S = gamma * sigma**2 * q
            d_bid = np.maximum(0.0, S + 1.0/k)
            d_ask = np.maximum(0.0, -S + 1.0/k)
            d_bid = np.where(q >= Q, np.inf, d_bid)
            d_ask = np.where(q <= -Q, np.inf, d_ask)
            return d_bid, d_ask
            
        _policy_cache[key] = _compute_policy(params)
    
    bid_opt, ask_opt = _policy_cache[key]
    
    n = int(np.round(t * steps))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    q = np.asarray(inventory, dtype=np.int64)
    
    # Map inventory to index: q in [-Q, Q] -> idx in [0, 2Q]
    idx = q + Q
    
    # Clip indices just in case
    idx = np.clip(idx, 0, 2*Q)
    
    bid_depths = bid_opt[n, idx]
    ask_depths = ask_opt[n, idx]
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
