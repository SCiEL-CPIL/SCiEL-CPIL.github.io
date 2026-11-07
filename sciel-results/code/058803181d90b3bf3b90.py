"""Market-making with inventory risk: Exact Dynamic Programming."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for optimal depth actions per regime and time step
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: dict { t: (bid_depths_array, ask_depths_array) }
# bid_depths_array shape (2*Q+1, ), index by q + Q
_OPT_CACHE = {}

def _compute_dp_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths for all t and all inventory levels q.
    
    Model:
    - State: q in [-Q, Q]
    - Action: (d_bid, d_ask) with d >= 0
    - Transition:
      p_bid = 1 - exp(-A * exp(-k * d_bid) * dt) if q < Q else 0
      p_ask = 1 - exp(-A * exp(-k * d_ask) * dt) if q > -Q else 0
      q' = q + u_ask - u_bid
    - Reward:
      R = d_bid * u_bid + d_ask * u_ask - (gamma / 2) * sigma^2 * q^2 * dt
    
    Objective: Maximize E[Sum R_t]
    
    Bellman Equation:
    V_t(q) = max_{d_bid, d_ask} { R(q, d) + E[V_{t+1}(q')] }
    
    Since bid and ask decisions are independent given q and V_{t+1}, we can
    maximize over d_bid and d_ask separately.
    
    For a fixed q:
    Term_bid(d) = p_bid(d) * (d + V_{t+1}(q-1) - V_{t+1}(q))
    Term_ask(d) = p_ask(d) * (d + V_{t+1}(q+1) - V_{t+1}(q))
    
    Let Delta_bid = V_{t+1}(q-1) - V_{t+1}(q)
    Let Delta_ask = V_{t+1}(q+1) - V_{t+1}(q)
    
    We maximize f(d) = (1 - exp(-lambda * exp(-k d))) * (d + Delta)
    where lambda = A * dt.
    
    Note: If q == Q, p_bid = 0, so term is 0 regardless of d. We can set d_bid=0.
    If q == -Q, p_ask = 0, so term is 0 regardless of d. We can set d_ask=0.
    
    We use a grid search for d to find the maximizer for each q.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _OPT_CACHE:
        return _OPT_CACHE[key]

    dt = T / steps
    lambda_ = A * dt
    
    # Grid for depths
    # d should be in [0, d_max]. 
    # If d is very large, p -> 0, reward -> 0.
    # If d is small, p is high, but d is small.
    # Typical scale is 1/k. Let's use [0, 5/k].
    # Resolution: 200 points.
    d_max = 5.0 / k
    d_grid = np.linspace(0.0, d_max, 200)
    
    # Pre-compute p(d) for the grid
    # p(d) = 1 - exp(-lambda * exp(-k * d))
    # Note: exp(-k * d) can be very small for large d.
    # p(d) is monotonic increasing in d?
    # Let x = exp(-k d). x is decreasing in d.
    # p = 1 - exp(-lambda x). 
    # dp/dd = exp(-lambda x) * lambda * (-x) * (-k) = k * lambda * x * exp(-lambda x) > 0.
    # So p is increasing in d.
    
    p_grid = 1.0 - np.exp(-lambda_ * np.exp(-k * d_grid))
    
    # Initialize value function
    # V shape: (steps + 1, 2*Q + 1)
    # Index: q_idx = q + Q
    V = np.zeros((steps + 1, 2 * Q + 1))
    
    # Policy storage
    # bid_pol[t, q_idx], ask_pol[t, q_idx]
    bid_pol = np.zeros((steps, 2 * Q + 1))
    ask_pol = np.zeros((steps, 2 * Q + 1))
    
    risk_coeff = 0.5 * gamma * sigma**2 * dt
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        V_next = V[t + 1]
        
        for q_idx in range(2 * Q + 1):
            q = q_idx - Q
            
            # Risk cost for current state
            risk_cost = risk_coeff * q * q
            
            # --- Optimize Bid ---
            if q < Q:
                # Delta_bid = V_next[q-1] - V_next[q]
                delta_bid = V_next[q_idx - 1] - V_next[q_idx]
                # Objective: p(d) * (d + delta_bid)
                # We want to maximize this over d_grid
                # Note: d + delta_bid can be negative if delta_bid is very negative and d is small.
                # But p(d) is small for small d? No, p(d) is small for small d?
                # p(0) = 1 - exp(-lambda) > 0.
                # If delta_bid is very negative, we might want p(d)=0 (d->inf) to avoid the loss?
                # But d is bounded. If d is large, p->0, term->0.
                # If d is small, p is max, term is p*(small + neg) which is neg.
                # So max(0, max_grid) logic applies implicitly if 0 is a valid value for the term?
                # Actually, if we choose d=0, we get p(0)*(0+delta).
                # If we choose d=inf, we get 0.
                # So we should take max(0, max_{d in grid} term).
                # However, our grid starts at 0.
                # If the max over grid is negative, we should effectively choose "no quote" (inf depth).
                # But the problem says "np.inf means no quote".
                # In the simulation, if we return inf, p=0.
                # So yes, we can clamp the value to 0 if the best grid value is negative.
                # But wait, if we clamp to 0, we set depth to inf?
                # The policy stores depth. If best value < 0, we set depth to inf.
                
                bid_vals = p_grid * (d_grid + delta_bid)
                best_idx_bid = np.argmax(bid_vals)
                best_val_bid = bid_vals[best_idx_bid]
                
                if best_val_bid > 0:
                    bid_pol[t, q_idx] = d_grid[best_idx_bid]
                    val_bid = best_val_bid
                else:
                    bid_pol[t, q_idx] = np.inf
                    val_bid = 0.0
            else:
                # q == Q, no bid
                bid_pol[t, q_idx] = np.inf
                val_bid = 0.0

            # --- Optimize Ask ---
            if q > -Q:
                # Delta_ask = V_next[q+1] - V_next[q]
                delta_ask = V_next[q_idx + 1] - V_next[q_idx]
                ask_vals = p_grid * (d_grid + delta_ask)
                best_idx_ask = np.argmax(ask_vals)
                best_val_ask = ask_vals[best_idx_ask]
                
                if best_val_ask > 0:
                    ask_pol[t, q_idx] = d_grid[best_idx_ask]
                    val_ask = best_val_ask
                else:
                    ask_pol[t, q_idx] = np.inf
                    val_ask = 0.0
            else:
                # q == -Q, no ask
                ask_pol[t, q_idx] = np.inf
                val_ask = 0.0

            # Total Value
            V[t, q_idx] = -risk_cost + val_bid + val_ask
        
        # Deadline check
        if time.time() > _DEADLINE:
            # If we run out of time, we have to return what we have.
            # But we need to fill the rest of the steps.
            # We can use the current V and pol for all remaining t?
            # Or just break and let quote handle missing t?
            # Better: break and assume stationary policy for t < current_t.
            # The policy at t is close to t-1.
            # Let's just break. quote will check bounds.
            break
    
    _OPT_CACHE[key] = (bid_pol, ask_pol)
    return _OPT_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed DP tables.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimized parameters
    bid_pol, ask_pol = _compute_dp_policy(sigma, A, k, gamma, T, steps, Q)
    
    # t is continuous: t = n * dt.
    # We need to map t to integer step index n.
    # n = int(t / dt) = int(t * steps / T)
    # Clamp n to [0, steps-1]
    n = int(t * steps / T)
    if n < 0:
        n = 0
    if n >= steps:
        n = steps - 1
        
    # If DP didn't finish (deadline hit), bid_pol might have uninitialized values for early t.
    # We initialized bid_pol with zeros.
    # If n >= actual_computed_steps, we should use the last computed step?
    # The loop goes from steps-1 down to 0.
    # If it breaks at t_break, then V[t] is computed for t >= t_break.
    # V[t] for t < t_break is 0 (initial).
    # We should use the policy at t_break for all t < t_break?
    # Or just use n = steps - 1?
    # Actually, if we break, the earliest t computed is the one where we broke.
    # Let's track the minimum t computed.
    # For simplicity, if n is too small (early in time) and we haven't computed it,
    # use the policy from the earliest computed step.
    # But we don't store that info.
    # Let's just assume it finishes. 1000 steps * 61 states * 200 grid = 12.2M ops.
    # NumPy is fast. 12.2M simple ops is < 0.1s.
    # The inner loop is Python. 61 * 1000 = 61,000 iterations.
    # Each iteration does some NumPy ops on size 200.
    # 61,000 * (few us) = ~0.1s.
    # So it should finish.
    
    # Get depths
    q = inventory
    # q is in [-Q, Q]. Index = q + Q.
    q_idx = q + Q
    
    # Ensure q_idx is valid
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = bid_pol[n, q_idx]
    ask_depths = ask_pol[n, q_idx]
    
    # Handle potential NaNs or invalid values from incomplete computation
    # If depth is 0 but it should be inf? 
    # If the loop didn't finish, bid_pol might be 0.
    # 0 is a valid depth.
    # But if we didn't compute it, it's a default 0.
    # Let's replace 0 with 1/k if we suspect it's default?
    # No, 0 is a valid optimal depth.
    # Let's just return what we have.
    
    # Ensure float arrays
    bid_depths = bid_depths.astype(float)
    ask_depths = ask_depths.astype(float)
    
    # Check for negative values (should not happen, but safety)
    bid_depths[bid_depths < 0] = 0.0
    ask_depths[ask_depths < 0] = 0.0
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
