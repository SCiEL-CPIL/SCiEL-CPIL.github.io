import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 295.0
_cache = {}

def _solve_regime(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1)
    
    # Parameterization:
    # Let x = exp(-k * depth). depth = -ln(x)/k.
    # x ranges from exp(-k*depth_max) to 1.
    # Probability of fill p = 1 - exp(-A * x * dt).
    
    # Grid for x
    # We want to cover depths from ~0 to ~20/k.
    # For k=1.5, 20/k ~ 13.3.
    # For k=0.75, 20/k ~ 26.6.
    # Let's use a log-spaced grid for x to get better resolution at small x (large depths)
    # where the derivative of p wrt x is small?
    # Actually p = 1 - exp(-A dt x).
    # d p / d x = A dt exp(-A dt x).
    # d p / d depth = p' * dx/ddepth = A dt exp(-A dt x) * (-k x).
    # Let's just use a fine log grid.
    # x_min corresponds to depth_max.
    # Let's choose depth_max such that p is negligible or value is small.
    # If depth is very large, p is very small.
    # Let's try x in [1e-10, 1].
    
    # Number of grid points
    # We have ~300s. 3 regimes.
    # DP loop: 1000 steps * 61 states.
    # Inside: optimize over grid.
    # If grid size is G, cost is ~ 1000 * 61 * G.
    # If G=5000, cost ~ 300M ops. NumPy vectorized this is fast (<1s).
    # Let's use G=5000.
    
    G = 5000
    x_grid = np.logspace(-10, 0, G)
    # depths = -np.log(x_grid) / k
    
    # Precompute p_grid
    # p = 1 - exp(-A * dt * x)
    # Note: A*dt*x can be small.
    p_grid = 1.0 - np.exp(-A * dt * x_grid)
    depth_grid = -np.log(x_grid) / k
    
    # V_next[q] for q in -Q..Q
    V_next = np.zeros(n_q)
    
    # Store optimal depths for each step and inventory
    # bid_depths[n, i], ask_depths[n, i]
    bid_depths = np.zeros((N, n_q))
    ask_depths = np.zeros((N, n_q))
    
    # Backward induction
    for n in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE:
            # Return whatever we have so far? Or raise?
            # Better to return best effort.
            # But we need to fill the rest with something valid.
            # Let's break and fill with 0 or inf.
            # Actually, if we break, the arrays are partially filled.
            # Let's just return what we have.
            # The caller expects arrays of shape (N, n_q).
            # Uninitialized parts are 0. 0 is a valid depth (aggressive).
            # This might be suboptimal but valid.
            break
        
        V_curr = np.zeros(n_q)
        
        # Inventory cost
        inv_cost = (gamma / 2.0) * (sigma ** 2) * (q_vals ** 2) * dt
        
        # Shifted value functions
        # V_plus[i] = V_next[i+1] for i < n_q-1
        # V_minus[i] = V_next[i-1] for i > 0
        V_plus = np.zeros(n_q)
        V_minus = np.zeros(n_q)
        V_plus[:-1] = V_next[1:]
        V_minus[1:] = V_next[:-1]
        
        # Objective for bid:
        # max_x p(x) * (depth(x) + V_plus[q])
        # Let's vectorize over q.
        # p_grid shape (G,)
        # depth_grid shape (G,)
        # V_plus shape (n_q,)
        
        # We want to compute M_bid[i] = max_j p[j] * (depth[j] + V_plus[i])
        # This is equivalent to finding the upper envelope of lines?
        # No, p and depth are linked.
        # f_i(x) = p(x) * (depth(x) + V_plus[i])
        # We can just compute the matrix and take max.
        
        # Matrix P_D shape (G, n_q)
        # P_D[j, i] = p[j] * depth[j] + p[j] * V_plus[i]
        # This is rank 1 + rank 1?
        # p[j]*depth[j] is vector u.
        # p[j] is vector v.
        # V_plus[i] is vector w.
        # M = u[:, None] + v[:, None] * w[None, :]
        # This is efficient.
        
        u = p_grid * depth_grid
        v = p_grid
        
        M_bid = u[:, None] + v[:, None] * V_plus[None, :]
        # Find max index for each i
        max_idx_bid = np.argmax(M_bid, axis=0)
        max_val_bid = M_bid[max_idx_bid, np.arange(n_q)]
        opt_x_bid = x_grid[max_idx_bid]
        opt_depth_bid = depth_grid[max_idx_bid]
        
        # Mask for q=Q (index n_q-1): no bid
        # Set depth to inf? Or 0?
        # "np.inf means no quote".
        # So set depth to inf.
        # But we store depths.
        bid_depths[n, :] = opt_depth_bid
        bid_depths[n, -1] = np.inf # No bid at Q
        
        # Objective for ask:
        # max_x p(x) * (depth(x) + V_minus[q])
        M_ask = u[:, None] + v[:, None] * V_minus[None, :]
        max_idx_ask = np.argmax(M_ask, axis=0)
        max_val_ask = M_ask[max_idx_ask, np.arange(n_q)]
        opt_depth_ask = depth_grid[max_idx_ask]
        
        ask_depths[n, :] = opt_depth_ask
        ask_depths[n, 0] = np.inf # No ask at -Q
        
        # Update V_curr
        # V_curr[i] = -inv_cost[i] + max_val_bid[i] + max_val_ask[i]
        # Note: if masked, max_val should be 0?
        # If no bid, the term is 0.
        # In our calculation, M_bid for i=n_q-1 used V_plus[n_q-1] which is 0 (since V_plus is zero-padded).
        # And we maximized p*(depth+0). This is non-zero.
        # But we should NOT add this value if we are not allowed to bid.
        # So we must zero out max_val_bid for i=n_q-1.
        max_val_bid[-1] = 0.0
        max_val_ask[0] = 0.0
        
        V_curr = -inv_cost + max_val_bid + max_val_ask
        
        V_next = V_curr
    
    # If we broke early due to deadline, fill remaining steps with 0s (or last known good)
    # Since we iterate backwards, if we break at n=k, steps k-1..0 are 0.
    # This is a valid fallback.
    
    _cache[key] = {'bid': bid_depths, 'ask': ask_depths}
    return _cache[key]

def quote(t, inventory, mid, params):
    policy = _solve_regime(params)
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step index
    # t = n * dt. dt = T / steps.
    # n = t / dt = t * steps / T
    n = int(np.round(t * params['steps'] / params['T']))
    # Clamp n to valid range [0, steps-1]
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to indices
    # inventory is in [-Q, Q]
    # Index = inventory + Q
    indices = inventory + Q
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = policy['bid'][n, indices]
    ask_depths = policy['ask'][n, indices]
    
    # Ensure shapes and types
    bid_depths = np.asarray(bid_depths, dtype=np.float64)
    ask_depths = np.asarray(ask_depths, dtype=np.float64)
    
    # Check for invalid values
    # "Values below -3/k, -inf, NaN ... make the whole result invalid"
    # Our depths are from -log(x)/k. x in [1e-10, 1].
    # depth in [0, -log(1e-10)/k] = [0, 23/k].
    # So depths are positive.
    # We set some to inf.
    # So no negative values, no NaN, no -inf.
    
    return bid_depths, ask_depths
