"""Optimal market making with inventory risk via HJB backward induction."""
import numpy as np
import time

# Cache for pre-computed value functions per regime
_value_cache = {}
_start_time = None


def _compute_value_function(params):
    """Compute the HJB value function V[step, q] via backward induction.
    
    Returns V as a 2D array of shape (steps+1, 2*Q+1) where
    V[n, q+Q] is the value at step n with inventory q.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_q = 2 * Q + 1
    
    # V[n, idx] where idx = q + Q, q in [-Q, Q]
    V = np.zeros((steps + 1, n_q), dtype=np.float64)
    
    # Terminal: V[steps, :] = 0
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_next = V[n + 1]  # shape (n_q,)
        
        for qi in range(n_q):
            q = qi - Q  # actual inventory
            
            # Inventory holding cost for this step
            hold_cost = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # --- Optimize bid depth ---
            # If q == Q, no bid fill possible
            if q == Q:
                bid_val = -hold_cost
            else:
                # Maximize: p_b * (b + V_next[q-1+Q]) + (1-p_b)*(-hold_cost)
                # where p_b = A * exp(-k*b) * dt
                # Optimal: b* ≈ 1/k - V_next[q-1+Q] (from FOC, p_b small)
                # But we need b* >= 0 and b* > -3/k (validity)
                # Let's do a small 1D optimization for accuracy
                
                V_bid_next = V_next[q - 1 + Q]  # V(t+dt, q-1)
                
                # Objective: f(b) = A*exp(-k*b)*dt * (b + V_bid_next) - (1 - A*exp(-k*b)*dt)*hold_cost
                # = A*exp(-k*b)*dt * (b + V_bid_next + hold_cost) - hold_cost
                # Maximize g(b) = A*exp(-k*b)*dt * (b + V_bid_next + hold_cost)
                
                # FOC: -k*exp(-k*b)*(b + c) + exp(-k*b) = 0 where c = V_bid_next + hold_cost
                # => 1 - k*(b + c) = 0 => b = 1/k - c
                c = V_bid_next + hold_cost
                b_opt = 1.0 / k - c
                
                # Clamp to valid range: b >= 0 (and > -3/k, but 0 is safe)
                # Also b should be reasonable; if b_opt < 0, use b=0
                if b_opt < 0.0:
                    b_opt = 0.0
                
                # Compute objective at b_opt
                p_b = A * np.exp(-k * b_opt) * dt
                # Clamp p_b to [0,1]
                if p_b > 1.0:
                    p_b = 1.0
                bid_val = p_b * (b_opt + V_bid_next) + (1.0 - p_b) * (-hold_cost)
            
            # --- Optimize ask depth ---
            if q == -Q:
                ask_val = -hold_cost
            else:
                V_ask_next = V_next[q + 1 + Q]  # V(t+dt, q+1)
                
                c = V_ask_next + hold_cost
                a_opt = 1.0 / k - c
                
                if a_opt < 0.0:
                    a_opt = 0.0
                
                p_a = A * np.exp(-k * a_opt) * dt
                if p_a > 1.0:
                    p_a = 1.0
                ask_val = p_a * (a_opt + V_ask_next) + (1.0 - p_a) * (-hold_cost)
            
            # Combined: fills are independent
            # E[reward] = (1-p_b)(1-p_a)*(-hold_cost) + p_b(1-p_a)*(b_opt + V_bid_next - hold_cost)
            #            + (1-p_b)*p_a*(a_opt + V_ask_next - hold_cost) + p_b*p_a*(b_opt+a_opt+V_bid_next+V_ask_next-hold_cost)
            # Simplify: = -hold_cost + p_b*(b_opt + V_bid_next) + p_a*(a_opt + V_ask_next)
            # Wait, that's not right because the hold cost is paid once, not per fill.
            # Actually: reward = (fills_earn) - hold_cost
            # E[reward] = E[fills_earn] - hold_cost
            # E[fills_earn] = p_b*(b_opt) + p_a*(a_opt) + p_b*p_a*0 (no double-earn issue, each fill earns its depth)
            # Wait, both can fill in same step. Each fill earns its depth.
            # E[fills_earn] = p_b*b_opt + p_a*a_opt (by linearity, since each fill independently earns its depth)
            # But we also need the continuation value:
            # E[continuation] = (1-p_b)(1-p_a)*V_next[q] + p_b(1-p_a)*V_next[q-1] + (1-p_b)p_a*V_next[q+1] + p_b*p_a*V_next[q]
            # (if both fill, inventory unchanged: q-1+1=q)
            
            if q == Q and q == -Q:
                # Can't happen unless Q=0
                V[n, qi] = -hold_cost
            elif q == Q:
                # Only ask possible
                V[n, qi] = ask_val
            elif q == -Q:
                # Only bid possible
                V[n, qi] = bid_val
            else:
                p_b = A * np.exp(-k * b_opt) * dt
                p_a = A * np.exp(-k * a_opt) * dt
                if p_b > 1.0: p_b = 1.0
                if p_a > 1.0: p_a = 1.0
                
                # E[immediate earnings]
                earn = p_b * b_opt + p_a * a_opt
                
                # E[continuation]
                V_cont = ((1.0 - p_b) * (1.0 - p_a) * V_next[q] +
                          p_b * (1.0 - p_a) * V_next[q - 1] +
                          (1.0 - p_b) * p_a * V_next[q + 1] +
                          p_b * p_a * V_next[q])
                
                V[n, qi] = earn - hold_cost + V_cont
    
    return V


def _get_value_function(params):
    """Get cached value function for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _value_cache:
        _value_cache[key] = _compute_value_function(params)
    return _value_cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    if time.time() - _start_time > 300:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_paths = len(inventory)
    inv = np.asarray(inventory, dtype=np.int64)
    
    # Get pre-computed value function
    V = _get_value_function(params)
    
    # Current step index
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    V_current = V[n]      # V(t, q)
    V_next = V[n + 1]     # V(t+dt, q)
    
    bid_depths = np.empty(n_paths, dtype=np.float64)
    ask_depths = np.empty(n_paths, dtype=np.float64)
    
    for i in range(n_paths):
        q = int(inv[i])
        qi = q + Q  # index into V
        
        # Holding cost
        hold_cost = (gamma / 2.0) * sigma * sigma * q * q * dt
        
        # --- Bid depth ---
        if q == Q:
            bid_depths[i] = np.inf
        else:
            V_bid_next = V_next[q - 1 + Q]
            c = V_bid_next + hold_cost
            b_opt = 1.0 / k - c
            if b_opt < 0.0:
                b_opt = 0.0
            bid_depths[i] = b_opt
        
        # --- Ask depth ---
        if q == -Q:
            ask_depths[i] = np.inf
        else:
            V_ask_next = V_next[q + 1 + Q]
            c = V_ask_next + hold_cost
            a_opt = 1.0 / k - c
            if a_opt < 0.0:
                a_opt = 0.0
            ask_depths[i] = a_opt
    
    # Ensure validity: no NaN, no values below -3/k
    min_valid = -3.0 / k
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < min_valid), 0.0, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < min_valid), 0.0, ask_depths)
    
    return bid_depths, ask_depths
