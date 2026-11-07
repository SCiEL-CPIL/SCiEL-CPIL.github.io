"""Market-making with inventory risk: Numerical Dynamic Programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 295.0
_DP_CACHE = {}

def _solve_dp(sigma, A, k, gamma, T, steps, Q):
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _DP_CACHE:
        return _DP_CACHE[key]

    dt = T / steps
    n_steps = steps
    n_states = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    # V[t, q_idx] = value function at step t, inventory state q_idx
    # t=0 is the last step (terminal), t=n_steps is the first step (initial)
    # We compute backwards from t=0 (terminal) to t=n_steps (initial)
    # V[0] = 0 (no terminal charge)
    V = np.zeros((n_steps + 1, n_states))
    
    # Pre-compute optimal actions for each step and state
    # optimal_bid[t, q_idx], optimal_ask[t, q_idx]
    opt_bid = np.zeros((n_steps + 1, n_states))
    opt_ask = np.zeros((n_steps + 1, n_states))
    
    # We will iterate t from 0 to n_steps-1 (backwards in time)
    # At step t (current time), we choose bid/ask.
    # The next time is t+1.
    # Inventory change:
    # If bid fills: q -> q+1, reward: depth_bid
    # If ask fills: q -> q-1, reward: depth_ask
    # If both fill: q -> q, reward: depth_bid + depth_ask (Wait, model says independent? "bid hit... ask... independently")
    # "A fill earns its depth and moves inventory one unit"
    # "no bid fill at q = Q, no ask fill at q = -Q"
    # Objective: depths earned - sum (gamma/2) sigma^2 q^2 dt
    
    # Let's re-read carefully: "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt), ask likewise, independently."
    # "A fill earns its depth and moves inventory one unit"
    # If both hit, do we earn both depths? Yes.
    # Inventory change: +1 (bid) - 1 (ask) = 0.
    
    # For a fixed state q at step t, we choose bid_depth b and ask_depth a.
    # p_b = 1 - exp(-A exp(-k b) dt)
    # p_a = 1 - exp(-A exp(-k a) dt)
    
    # Expected immediate reward:
    # R = p_b * b + p_a * a - 0.5 * gamma * sigma^2 * q^2 * dt
    # Note: The cost term depends on q BEFORE fills.
    
    # Expected future value:
    # E[V_{t+1}(q')]
    # q' = q + 1 with prob p_b(1-p_a)
    # q' = q - 1 with prob (1-p_b)p_a
    # q' = q with prob p_b p_a + (1-p_b)(1-p_a)
    
    # We need to maximize R + E[V_{t+1}(q')] over b, a.
    
    # Since b and a enter independently in p_b and p_a, but coupled in the joint probability,
    # we can't simply separate them unless we approximate.
    # However, the state space is small (61 states) and steps are 1000.
    # We can do a grid search over b and a for each state and step?
    # 1000 * 61 * (grid_b * grid_a). If grid is 20x20=400, that's 24M evaluations.
    # Each evaluation involves exp, log, etc. Might be slow in pure Python loops.
    
    # Vectorization strategy:
    # For each step t, we have V_next (shape n_states).
    # We want to find b*, a* for each q.
    # Since the problem is symmetric (if we flip q and swap bid/ask, and assume symmetric costs),
    # we might be able to reduce search. But let's just do it directly.
    
    # Grid for depths.
    # Depths should be positive.
    # Range: 0 to some max.
    # 1/k is a natural scale. Let's use 0 to 3/k.
    # Let's use 30 points for each.
    depth_grid = np.linspace(0.0, 3.0/k, 30)
    
    # Pre-compute p for all depths in grid
    # p_b_grid[i] = 1 - exp(-A exp(-k depth_grid[i]) dt)
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Loop over time steps backwards
    for t in range(n_steps - 1, -1, -1):
        V_next = V[t + 1]
        
        # For each state q_idx:
        for qi in range(n_states):
            q = q_indices[qi]
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # We need to maximize:
            # max_{b,a} [ p_b*b + p_a*a - cost + 
            #             p_b(1-p_a) V_next[qi+1] + 
            #             (1-p_b)p_a V_next[qi-1] + 
            #             (p_b p_a + (1-p_b)(1-p_a)) V_next[qi] ]
            
            # Boundary conditions:
            # If q == Q, no bid fill. So p_b_eff = 0.
            # If q == -Q, no ask fill. So p_a_eff = 0.
            
            if q == Q:
                # Only ask matters
                # Maximize: p_a*a - cost + (1-p_a)V_next[qi-1] + p_a V_next[qi]
                # (Note: q-1 is valid)
                best_val = -np.inf
                best_a = 0.0
                # Vectorized over a
                a_grid = depth_grid
                pa = p_grid
                reward = pa * a_grid - cost
                future = (1 - pa) * V_next[qi-1] + pa * V_next[qi]
                total = reward + future
                idx = np.argmax(total)
                best_a = a_grid[idx]
                opt_bid[t+1, qi] = 0.0 # Doesn't matter, but set 0
                opt_ask[t+1, qi] = best_a
                V[t+1, qi] = total[idx]
            elif q == -Q:
                # Only bid matters
                # Maximize: p_b*b - cost + (1-p_b)V_next[qi+1] + p_b V_next[qi]
                best_val = -np.inf
                best_b = 0.0
                b_grid = depth_grid
                pb = p_grid
                reward = pb * b_grid - cost
                future = (1 - pb) * V_next[qi+1] + pb * V_next[qi]
                total = reward + future
                idx = np.argmax(total)
                best_b = b_grid[idx]
                opt_bid[t+1, qi] = best_b
                opt_ask[t+1, qi] = 0.0
                V[t+1, qi] = total[idx]
            else:
                # General case
                # We can iterate over b, and for each b, optimize a?
                # Or full 2D grid.
                # 2D grid: 30x30 = 900 evaluations per state.
                # 61 states * 1000 steps * 900 = 55M evaluations.
                # This might be too slow in Python.
                
                # Let's try to vectorize over states for a fixed (b, a)?
                # No, b and a depend on q.
                
                # Alternative: Since p_b and p_a are independent functions of b and a,
                # and the future value term is bilinear in p_b, p_a:
                # E[V] = p_b(1-p_a) V_{q+1} + (1-p_b)p_a V_{q-1} + (1 - p_b - p_a + p_b p_a) V_q
                # Let V_p = V_{q+1}, V_m = V_{q-1}, V_0 = V_q
                # E[V] = p_b V_p - p_b p_a V_p + p_a V_m - p_b p_a V_m + V_0 - p_b V_0 - p_a V_0 + p_b p_a V_0
                # E[V] = V_0 + p_b (V_p - V_0) + p_a (V_m - V_0) - p_b p_a (V_p + V_m - 2V_0)
                
                # Objective:
                # J(b,a) = p_b b + p_a a - cost + V_0 + p_b (V_p - V_0) + p_a (V_m - V_0) - p_b p_a (V_p + V_m - 2V_0)
                
                # Let C1 = V_p - V_0
                # Let C2 = V_m - V_0
                # Let C3 = V_p + V_m - 2V_0
                # J(b,a) = V_0 - cost + p_b (b + C1) + p_a (a + C2) - p_b p_a C3
                
                # We want to max over b, a.
                # Note p_b = f(b), p_a = f(a).
                
                # We can loop over b (30 options), and for each b, find best a (30 options).
                # Total 900 ops per state.
                # But we can vectorize the "find best a" part for all b's at once?
                # For a fixed b, p_b is constant.
                # J(b,a) = Const(b) + p_a (a + C2 - p_b C3)
                # So for fixed b, we just need to max over a of: p_a(a) * (a + C2 - p_b C3)
                # This is a 1D maximization.
                
                V_p = V_next[qi+1]
                V_m = V_next[qi-1]
                V_0 = V_next[qi]
                C1 = V_p - V_0
                C2 = V_m - V_0
                C3 = V_p + V_m - 2*V_0
                
                # Loop over b
                best_total = -np.inf
                best_b = 0.0
                best_a = 0.0
                
                for i_b, b in enumerate(depth_grid):
                    p_b = p_grid[i_b]
                    # Term for b: p_b (b + C1)
                    term_b = p_b * (b + C1)
                    # Coefficient for p_a(a) * a: (1 - p_b * C3 / (a+C2) ? No)
                    # J = V_0 - cost + term_b + p_a (a + C2 - p_b C3)
                    # Let K_a = C2 - p_b * C3
                    # We want max_a [ p_a(a) * (a + K_a) ]
                    # Vectorized over a_grid
                    a_vals = depth_grid
                    p_a_vals = p_grid
                    K_a = C2 - p_b * C3
                    obj_a = p_a_vals * (a_vals + K_a)
                    idx_a = np.argmax(obj_a)
                    best_a_val = a_vals[idx_a]
                    best_obj_a = obj_a[idx_a]
                    
                    total_val = V_0 - cost + term_b + best_obj_a
                    
                    if total_val > best_total:
                        best_total = total_val
                        best_b = b
                        best_a = best_a_val
                
                opt_bid[t+1, qi] = best_b
                opt_ask[t+1, qi] = best_a
                V[t+1, qi] = best_total

    _DP_CACHE[key] = (opt_bid, opt_ask)
    return _DP_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP table.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple heuristic
        k = params['k']
        depth = 1.0 / k
        return np.full(len(inventory), depth), np.full(len(inventory), depth)

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimal actions table
    # opt_bid[step_idx, q_idx], opt_ask[step_idx, q_idx]
    # step_idx corresponds to the step number n.
    # In our DP, V[t] is value at time t.
    # We computed V[t+1] using V[t+2]... wait.
    # Let's trace indices.
    # V[0] is terminal.
    # Loop t from n_steps-1 down to 0.
    # Inside loop, we compute V[t+1, qi] using V[t+2, ...]? No.
    # My code:
    # for t in range(n_steps - 1, -1, -1):
    #    V_next = V[t + 1]
    #    ... compute V[t+1, qi] ...
    # So V[t+1] is the value at time step t+1?
    # If t is the loop variable, and we are at time step corresponding to index t in the "remaining steps" sense?
    # Let's clarify time indexing.
    # n=0 is first call. t=0.
    # n=999 is last call. t=999*dt.
    # DP usually works backwards.
    # Let V_k(q) be value function with k steps remaining.
    # V_0(q) = 0.
    # V_{k+1}(q) = max_{b,a} [ Reward + E[V_k(q')] ]
    # At step n (0-indexed), steps remaining = steps - n.
    # So we need V_{steps-n}.
    
    # In my code:
    # V is array of shape (n_steps+1, n_states).
    # V[0] is initialized to 0. This corresponds to V_0 (0 steps remaining).
    # Loop t from n_steps-1 down to 0.
    # If t = n_steps-1, we compute V[n_steps] using V[n_steps-1].
    # Wait, V_next = V[t+1].
    # If t = n_steps-1, V_next = V[n_steps]. But V[n_steps] was not computed yet?
    # No, V is initialized to zeros.
    # So V[n_steps] is 0.
    # Then we compute V[n_steps-1] using V[n_steps].
    # This means V[i] in my array corresponds to V_i (i steps remaining)?
    # Let's check:
    # t = n_steps-1: V_next = V[n_steps] (which is 0, i.e., V_0).
    # We compute V[n_steps-1]. This should be V_1 (1 step remaining).
    # So V[i] in array = V_{n_steps - i}? No.
    # V[n_steps-1] is computed from V[n_steps].
    # If V[n_steps] = V_0, then V[n_steps-1] = V_1.
    # Then t = n_steps-2: V_next = V[n_steps-1] = V_1.
    # We compute V[n_steps-2] = V_2.
    # So V[i] = V_{n_steps - i}.
    
    # At step n (0-indexed), steps remaining = steps - n.
    # So we need V_{steps-n}.
    # This corresponds to array index i such that n_steps - i = steps - n.
    # i = n_steps - (steps - n) = steps - steps + n = n.
    # So we just use V[n].
    
    # Let's verify.
    # n=0 (first step). Steps remaining = 1000.
    # We need V_1000.
    # Array index i = 0.
    # V[0] is the first element.
    # In my loop, V[0] is computed last (t=0).
    # V[0] is computed from V[1].
    # V[1] is V_{999}.
    # So V[0] is V_{1000}. Correct.
    
    opt_bid, opt_ask = _solve_dp(sigma, A, k, gamma, T, steps, Q)
    
    # Map inventory to index
    # q_indices = np.arange(-Q, Q+1)
    # idx = q + Q
    q_idx = inventory + Q
    
    # Clip indices to be safe, though inventory should be in range
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    # Get actions for step n
    # n is the current step index (0..999)
    n = int(t / (T / steps))
    # Ensure n is within bounds
    n = np.clip(n, 0, steps-1)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Ensure valid depths
    bid_depths = np.clip(bid_depths, 0.0, 5.0 / k)
    ask_depths = np.clip(ask_depths, 0.0, 5.0 / k)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
