"""
Optimal market making with inventory risk via exact Dynamic Programming.

The state space is (step, inventory) where step in [0, steps) and inventory in [-Q, Q].
For each state, we choose bid_depth and ask_depth to maximize the expected cumulative reward.
The reward per step is:
  - If a bid fill occurs: +bid_depth (earned depth) and inventory decreases by 1
  - If an ask fill occurs: -ask_depth (paid depth) and inventory increases by 1
  - Inventory penalty: -(gamma/2) * sigma^2 * q^2 * dt where q is inventory BEFORE fills

The fill probabilities are:
  P(bid fill) = 1 - exp(-A * exp(-k * bid_depth) * dt)
  P(ask fill) = 1 - exp(-A * exp(-k * ask_depth) * dt)
  (independent)

Constraints: no bid fill if inventory == Q, no ask fill if inventory == -Q.

We use backward DP from step = steps-1 down to step = 0.
At each state (n, q), we search over possible bid_depth and ask_depth values to find the optimal.

To make this tractable, we discretize the depth space. The optimal depth is likely in a bounded range.
We use a grid of depth values and find the best combination.

Key insight: the objective separates in a specific way. For a given state (n, q):
  V(n, q) = max_{b, a} [ 
      p_b * (b + V(n+1, q-1)) + p_a * (-a + V(n+1, q+1)) 
      + (1-p_b)*(1-p_a) * V(n, q)  -- wait, this is wrong
  ]

Actually, let me reconsider. The mid moves randomly, but the depth is chosen relative to mid.
The key is that the fill probabilities depend only on the depth, not on mid directly.
The inventory penalty depends on q.

Let me re-derive:
  V(n, q) = max_{b, a} [ 
      -(gamma/2) * sigma^2 * q^2 * dt
      + p_b * (b + E[V(n+1, q-1)]) 
      + p_a * (-a + E[V(n+1, q+1)])
      + (1-p_b)*(1-p_a) * E[V(n+1, q)]
  ]

where p_b = 1 - exp(-A * exp(-k*b) * dt), p_a = 1 - exp(-A * exp(-k*a) * dt)

And E[V(n+1, q')] is the expected value at the next step given inventory q', which depends on the random mid move.
But wait - the mid move doesn't affect the future value directly because the depth is chosen relative to mid each step.
The only thing that carries forward is the inventory level.

So actually:
  V(n, q) = max_{b, a} [ 
      -(gamma/2) * sigma^2 * q^2 * dt
      + p_b * (b + V(n+1, q-1)) 
      + p_a * (-a + V(n+1, q+1))
      + (1-p_b)*(1-p_a) * V(n+1, q)
  ]

This is a clean DP! The mid randomness doesn't affect the value function because we re-quote relative to mid each step.

Boundary conditions:
  V(steps, q) = 0 for all q (no terminal charge)
  If q == Q, we cannot have bid fills, so p_b = 0 (or we set b = inf)
  If q == -Q, we cannot have ask fills, so p_a = 0 (or we set a = inf)

Now, to solve this, for each state (n, q), we need to maximize over b and a.
The function to maximize is:
  f(b, a) = p_b(b) * (b + V(n+1, q-1)) + p_a(a) * (-a + V(n+1, q+1)) + (1-p_b(b))*(1-p_a(a)) * V(n+1, q)

where p_b(b) = 1 - exp(-A * exp(-k*b) * dt) and p_a(a) = 1 - exp(-A * exp(-k*a) * dt)

This can be separated! Let's define:
  g_b(b) = p_b(b) * (b + V(n+1, q-1)) + (1-p_b(b)) * V(n+1, q)  -- no, this isn't quite right because of the joint term

Let me expand:
  f(b, a) = p_b * (b + V_{q-1}) + p_a * (-a + V_{q+1}) + (1-p_b)(1-p_a) * V_q
  = p_b * b + p_b * V_{q-1} - p_a * a + p_a * V_{q+1} + V_q - p_b*V_q - p_a*V_q + p_b*p_a*V_q
  = p_b * (b + V_{q-1} - V_q) + p_a * (-a + V_{q+1} - V_q) + V_q + p_b * p_a * V_q

So f(b,a) = V_q + p_b(b) * (b + V_{q-1} - V_q) + p_a(a) * (-a + V_{q+1} - V_q) + p_b(b) * p_a(a) * V_q

This still couples b and a through the last term. But if V_q is small or we can handle it...

Actually, let's just do a 2D grid search over b and a for each state. The grid can be moderate.

Let me implement this with a reasonable grid size and see if it's fast enough.
"""

import numpy as np
import time

# Cache for pre-computed optimal policies
_policy_cache = {}
_start_time = None


def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (step, inventory) states using DP.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    n_inv = 2 * Q + 1  # number of inventory levels
    
    # Grid for depth search
    # Optimal depth is likely in [0, some_max]
    # Let's use a grid from 0 to 5/k with reasonable resolution
    max_depth = 5.0 / k
    n_grid = 50  # number of grid points
    depth_grid = np.linspace(0.0, max_depth, n_grid)
    
    # V[n, q] where q is indexed 0..2Q, actual inventory = q - Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Store optimal depths
    bid_opt = np.full((steps, n_inv), np.inf, dtype=np.float64)
    ask_opt = np.full((steps, n_inv), np.inf, dtype=np.float64)
    
    # Backward DP
    for n in range(steps - 1, -1, -1):
        for q_idx in range(n_inv):
            q = q_idx - Q  # actual inventory
            
            V_q = V[n + 1, q_idx]
            
            # Check if we can quote bid/ask
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                bid_opt[n, q_idx] = np.inf
                ask_opt[n, q_idx] = np.inf
                V[n, q_idx] = V_q
                continue
            
            # Inventory penalty
            penalty = -0.5 * gamma * sigma * sigma * q * q * dt
            
            best_val = -np.inf
            best_b = np.inf
            best_a = np.inf
            
            # Get next step values
            if can_bid:
                V_q_minus_1 = V[n + 1, q_idx - 1]
            else:
                V_q_minus_1 = 0.0
            
            if can_ask:
                V_q_plus_1 = V[n + 1, q_idx + 1]
            else:
                V_q_plus_1 = 0.0
            
            # Search over depth grid
            if can_bid and can_ask:
                for b in depth_grid:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    for a in depth_grid:
                        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                        
                        val = penalty + p_b * (b + V_q_minus_1) + p_a * (-a + V_q_plus_1) + (1 - p_b) * (1 - p_a) * V_q
                        
                        if val > best_val:
                            best_val = val
                            best_b = b
                            best_a = a
            elif can_bid:
                for b in depth_grid:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                    val = penalty + p_b * (b + V_q_minus_1) + (1 - p_b) * V_q
                    if val > best_val:
                        best_val = val
                        best_b = b
                best_a = np.inf
            else:  # can_ask only
                for a in depth_grid:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    val = penalty + p_a * (-a + V_q_plus_1) + (1 - p_a) * V_q
                    if val > best_val:
                        best_val = val
                        best_a = a
                best_b = np.inf
            
            V[n, q_idx] = best_val
            bid_opt[n, q_idx] = best_b
            ask_opt[n, q_idx] = best_a
        
        # Check time
        if time.time() - _start_time > 280:
            break
    
    return bid_opt, ask_opt


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Check deadline
    if time.time() - _start_time > 290:
        # Fallback: return simple depths
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    # Create cache key
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_policy(params)
    
    bid_opt, ask_opt = _policy_cache[key]
    
    steps = params['steps']
    Q = params['Q']
    
    # Determine current step
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)
    
    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q] -> index in [0, 2Q]
    
    # Get depths
    bid_depths = bid_opt[n, inv_idx]
    ask_depths = ask_opt[n, inv_idx]
    
    return bid_depths, ask_depths
