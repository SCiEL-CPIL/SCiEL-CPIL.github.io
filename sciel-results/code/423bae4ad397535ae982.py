"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

# Global cache for pre-computed optimal policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: tuple of (bid_depths_table, ask_depths_table)
# Tables are 2D arrays of shape (steps, Q+1) where index [n, q+Q] gives optimal depth
_policy_cache = {}

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid and ask depths for all time steps and inventory levels
    using backward induction on the Bellman equation.
    
    State: (n, q) where n is step index (0..steps-1), q is inventory (-Q..Q).
    Action: (d_bid, d_ask) where d_bid >= 0, d_ask >= 0.
    
    Objective: Maximize expected sum of (depths earned) - (gamma/2) * sigma^2 * q^2 * dt
    Note: The objective in the prompt is "depths earned minus sum of inventory risk".
    Wait, let's re-read the objective carefully.
    "Objective per path: depths earned minus sum over steps of (gamma / 2) sigma^2 q^2 dt"
    
    This is a bit unusual. Usually, it's cash + inventory valuation.
    Here, it seems we want to maximize the sum of depths we quote?
    No, "depths earned" usually means the price difference or the depth itself?
    In market making, if you quote at depth d, and it fills, you earn d (relative to mid).
    So if bid fills, you earn d_bid. If ask fills, you earn d_ask.
    The cost is the inventory risk term.
    
    So the reward at step n given state q and actions (db, da) is:
    R = p_bid * db + p_ask * da - (gamma / 2) * sigma^2 * q^2 * dt
    
    The next state q' depends on fills:
    - Bid fill: q' = q + 1 (if q < Q)
    - Ask fill: q' = q - 1 (if q > -Q)
    - No fill: q' = q
    - Both fill: q' = q (bid +1, ask -1)
    
    Probabilities:
    p_bid = 1 - exp(-A * exp(-k * db) * dt)
    p_ask = 1 - exp(-A * exp(-k * da) * dt)
    
    Since fills are independent:
    P(bid fill, no ask) = p_bid * (1 - p_ask)
    P(no bid, ask fill) = (1 - p_bid) * p_ask
    P(bid fill, ask fill) = p_bid * p_ask
    P(no fill) = (1 - p_bid) * (1 - p_ask)
    
    Value Function V(n, q):
    V(steps, q) = 0  # No terminal charge
    V(n, q) = max_{db, da >= 0} { R(n, q, db, da) + E[V(n+1, q')] }
    
    Since the problem is separable in a certain sense? No, the state transition couples db and da via the probabilities.
    However, for a fixed n and q, we can search for the optimal db and da.
    Given the small range of Q (30) and steps (1000), we can pre-compute this.
    But the action space is continuous.
    
    Insight: The optimal depth often satisfies a first-order condition.
    Let's analyze the derivative of the expected value w.r.t db.
    Let p_b = 1 - exp(-A e^{-k db} dt).
    dp_b/ddb = -exp(-A e^{-k db} dt) * (-A dt) * e^{-k db} * (-k)
             = -k A dt e^{-k db} exp(-A e^{-k db} dt)
    
    This looks complex. However, note that the value function V(n+1, q') is convex in q' (quadratic risk).
    The optimal policy in such MDPs with quadratic costs and exponential fill probabilities often has a simple structure.
    
    Alternative approach:
    Since Q is small (30) and steps is 1000, we can discretize the depth search?
    No, depth is continuous.
    
    Let's look at the structure again.
    V(n, q) = max_{db, da} [ p_b db + p_a da - C_q + 
        p_b(1-p_a)V(n+1, q+1) + (1-p_b)p_a V(n+1, q-1) + p_b p_a V(n+1, q) + (1-p_b)(1-p_a)V(n+1, q) ]
    
    Simplify the expectation term:
    E[V] = V(n+1, q) + p_b(1-p_a)(V(n+1, q+1) - V(n+1, q)) + p_a(1-p_b)(V(n+1, q-1) - V(n+1, q)) + p_b p_a (0)
    Let Delta+ = V(n+1, q+1) - V(n+1, q)
    Let Delta- = V(n+1, q-1) - V(n+1, q)
    
    Objective J(db, da) = p_b db + p_a da - C_q + V(n+1, q) + p_b(1-p_a)Delta+ + p_a(1-p_b)Delta-
    
    Since db and da appear in p_b and p_a, and the terms are coupled via p_b(1-p_a) etc., we can't fully decouple.
    However, if we assume the optimal db and da are close to 1/k (risk neutral), we can do a local search.
    Or, we can observe that for small dt, p_b and p_a are small.
    p_b ≈ A dt e^{-k db}.
    
    Let's try a numerical optimization for each (n, q) pair during pre-computation.
    Number of states: 1000 * 61 = 61,000.
    For each state, we need to maximize a 2-variable function.
    This might be slow if done naively for 3 regimes.
    
    Let's optimize the search.
    The function is smooth and unimodal.
    We can use a grid search or Newton's method.
    Given the time limit (300s), pre-computing for 3 regimes might be feasible if the inner loop is fast.
    
    Actually, there is a known result for this specific model (Avellaneda-Stoikov discrete).
    The optimal skew is linear in inventory.
    But the depth itself might change.
    
    Let's implement the DP with a coarse-to-fine search for db and da.
    Range of db, da: [0, 5/k] is likely enough.
    
    Wait, the prompt says "distances below and above mid to quote".
    If I quote at depth d, the price is mid - d (bid) or mid + d (ask).
    
    Let's implement the DP.
    
    Steps:
    1. Initialize V_next = zeros(Q+1) for n = steps.
    2. Iterate n from steps-1 down to 0.
    3. For each q in -Q..Q:
       - If q == Q, bid is disabled (p_b = 0). Optimize da only.
       - If q == -Q, ask is disabled (p_a = 0). Optimize db only.
       - Else, optimize db and da.
       
    To speed up, we can vectorize the q dimension?
    The objective depends on V(n+1, q-1), V(n+1, q), V(n+1, q+1).
    These are local.
    
    Let's use a simple grid search for db and da for each state?
    61,000 states * (grid_size^2) evaluations.
    If grid_size = 10, that's 61,000 * 100 = 6.1M evaluations.
    Each evaluation is O(1) math.
    6.1M operations is very fast in Python if vectorized or even in a tight loop?
    A tight loop of 6M in Python takes ~0.1-0.5s.
    3 regimes -> ~1.5s.
    This is well within the 300s limit.
    
    We can even use a finer grid or a Newton step.
    Let's use a coarse grid to find the neighborhood, then a fine grid or Newton.
    Actually, just a fine grid over a reasonable range might be enough.
    Range: [0, 2/k].
    
    Let's refine the objective function calculation.
    C_q = (gamma / 2) * sigma^2 * q^2 * dt
    
    def calc_obj(db, da, q, V_next, p_b_base, p_a_base...):
        # p_b = 1 - exp(-A * exp(-k * db) * dt)
        # If db is large, p_b is small.
        
    We will pre-compute the tables `bid_depths[n, q_index]` and `ask_depths[n, q_index]`.
    """
    
    dt = T / steps
    C_q = (gamma / 2.0) * sigma**2 * np.arange(-Q, Q+1)**2 * dt
    
    # V_next is array of shape (Q+1,)
    V_next = np.zeros(Q+1)
    
    bid_table = np.zeros((steps, Q+1))
    ask_table = np.zeros((steps, Q+1))
    
    # Pre-compute exponentials for speed?
    # exp(-k * d)
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(Q+1)
        for q_idx in range(Q+1):
            q = q_idx - Q
            
            # Get neighbors
            V_q = V_next[q_idx]
            V_qm1 = V_next[q_idx - 1] if q_idx > 0 else V_q  # Boundary: no transition?
            # Actually, if q == -Q, ask fill is impossible. So q-1 state is not reached via ask.
            # But in the expectation, if q == -Q, p_a must be 0.
            # So we handle boundaries by forcing depth to inf or p to 0.
            
            if q == Q:
                # No bid fill possible. Optimize da only.
                # p_b = 0
                # Obj = p_a * da + V_q + p_a * (V_qm1 - V_q)  # Wait, ask fill moves q -> q-1
                # q=Q, ask fill -> Q-1.
                # Delta_minus = V_next[Q-1] - V_next[Q]
                Delta_minus = V_next[q_idx - 1] - V_q
                Delta_plus = 0 # Not used
                
                # Maximize f(da) = p_a(da) * (da + Delta_minus) - C_q
                # p_a = 1 - exp(-A * exp(-k * da) * dt)
                
                # Grid search for da
                # Range [0, 2/k]
                das = np.linspace(0, 2.0/k, 20)
                pas = 1.0 - np.exp(-A * np.exp(-k * das) * dt)
                objs = pas * (das + Delta_minus) - C_q[q_idx]
                best_idx = np.argmax(objs)
                best_da = das[best_idx]
                
                # Refine? For now, grid is okay. Let's make it finer.
                # 20 points might be coarse. Let's do 50.
                # Actually, let's just use a slightly finer grid for all.
                
                bid_table[n, q_idx] = np.inf # No bid
                ask_table[n, q_idx] = best_da
                V_curr[q_idx] = objs[best_idx]
                
            elif q == -Q:
                # No ask fill possible. Optimize db only.
                Delta_plus = V_next[q_idx + 1] - V_q
                # Obj = p_b * (db + Delta_plus) - C_q
                dbs = np.linspace(0, 2.0/k, 50)
                pbs = 1.0 - np.exp(-A * np.exp(-k * dbs) * dt)
                objs = pbs * (dbs + Delta_plus) - C_q[q_idx]
                best_idx = np.argmax(objs)
                best_db = dbs[best_idx]
                
                bid_table[n, q_idx] = best_db
                ask_table[n, q_idx] = np.inf # No ask
                V_curr[q_idx] = objs[best_idx]
                
            else:
                # Both sides possible
                Delta_plus = V_next[q_idx + 1] - V_q
                Delta_minus = V_next[q_idx - 1] - V_q
                
                # Obj(db, da) = p_b(db) * db + p_a(da) * da - C_q + 
                #             p_b(1-p_a)Delta_plus + p_a(1-p_b)Delta_minus
                #             + (1-p_b)(1-p_a)V_q + p_b p_a V_q  -- Wait, V_q is constant part?
                # E[V] = V_q + p_b(1-p_a)Delta_plus + p_a(1-p_b)Delta_minus
                # So Obj = p_b db + p_a da - C_q + V_q + p_b(1-p_a)Delta_plus + p_a(1-p_b)Delta_minus
                
                # We can simplify:
                # Obj = V_q - C_q + p_b [ db + (1-p_a)Delta_plus - p_a Delta_minus ] + p_a [ da + (1-p_b)Delta_minus ]
                # This is coupled.
                
                # Let's just do a 2D grid search.
                # Range [0, 2/k]
                # 50x50 grid = 2500 points.
                # 61,000 states * 2500 = 150M ops. Too slow in Python loop.
                
                # Optimization:
                # The optimal db depends on da and vice versa.
                # However, for small dt, p_b and p_a are small.
                # Linear approximation:
                # p_b ≈ A dt e^{-k db}
                # p_a ≈ A dt e^{-k da}
                
                # Let's try a coarser grid first, then refine.
                # Or, notice that the function is separable in first order.
                
                # Let's use a vectorized approach for the grid for each state?
                # No, the Delta terms change per state.
                
                # Alternative: Use the fact that the optimal depth is close to 1/k.
                # Search in [0.5/k, 1.5/k] with fine grid.
                
                # Let's reduce the grid size.
                # 20x20 = 400 points.
                # 61,000 * 400 = 24M ops.
                # In Python, 24M simple float ops might take ~1-2 seconds.
                # 3 regimes -> 6s. Acceptable.
                
                dbs = np.linspace(0.0, 2.0/k, 20)
                das = np.linspace(0.0, 2.0/k, 20)
                
                # Vectorize over the grid for this state
                # Create meshgrid
                DB, DA = np.meshgrid(dbs, das, indexing='ij')
                PB = 1.0 - np.exp(-A * np.exp(-k * DB) * dt)
                PA = 1.0 - np.exp(-A * np.exp(-k * DA) * dt)
                
                # Objective matrix
                Obj = (PB * DB + PA * DA - C_q[q_idx] + V_q + 
                       PB * (1.0 - PA) * Delta_plus + 
                       PA * (1.0 - PB) * Delta_minus)
                
                max_idx = np.unravel_index(np.argmax(Obj), Obj.shape)
                best_db = dbs[max_idx[0]]
                best_da = das[max_idx[1]]
                
                bid_table[n, q_idx] = best_db
                ask_table[n, q_idx] = best_da
                V_curr[q_idx] = Obj[max_idx]
                
        V_next = V_curr
        
    return bid_table, ask_table

def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        # Check time
        start_time = time.time()
        bid_table, ask_table = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
        
        # If this took too long, subsequent calls will be fast.
        # If it times out, we fallback to heuristic.
        # But 24M ops is fast.
        
    bid_table, ask_table = _policy_cache[key]
    
    n = int(round(t / (T / steps)))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
        
    # Map inventory to index
    # inventory is array of ints in [-Q, Q]
    # Clip to be safe
    inv_clipped = np.clip(inventory, -Q, Q)
    indices = inv_clipped + Q
    
    # Fetch depths
    bid_depths = bid_table[n, indices]
    ask_depths = ask_table[n, indices]
    
    # Ensure validity
    # Values below -3/k are invalid. Our depths are >= 0.
    # np.inf is allowed for no quote.
    
    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
