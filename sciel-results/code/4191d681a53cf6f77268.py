"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed optimal depths per (sigma, A, k, gamma, Q, steps)
_depth_cache = {}
_start_time = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill


def _solve_depth_for_cost(C, A, k, dt):
    """
    Solve for optimal depth d given marginal cost C.
    FOC: (1 - exp(-L)) + (d - C) * k * L * exp(-L) = 0
    where L = A * exp(-k*d) * dt.
    
    For C <= 0: optimal depth is 0 (we want to fill).
    For C > 0: solve the transcendental equation.
    """
    if C <= 0:
        return 0.0
    
    # Try Newton's method on the FOC
    # f(d) = (1 - exp(-L)) + (d - C) * k * L * exp(-L)
    # where L = A * exp(-k*d) * dt
    
    # Initial guess: d0 = C + 1/k (small-L approximation)
    d = C + 1.0 / k
    if d < 0:
        d = 0.0
    if d > 50.0 / k:
        d = 50.0 / k
    
    # Newton iteration
    for _ in range(20):
        L = A * np.exp(-k * d) * dt
        exp_neg_L = np.exp(-L)
        f = (1.0 - exp_neg_L) + (d - C) * k * L * exp_neg_L
        
        # f'(d) = dL/dd * [exp(-L) + (d-C)*k*exp(-L)*(-1)] + exp(-L)
        # dL/dd = -k * L
        # f'(d) = -k*L * [exp(-L) - (d-C)*k*exp(-L)] + exp(-L)
        #        = -k*L*exp(-L)*(1 - (d-C)*k) + exp(-L)
        #        = exp(-L) * [1 - k*L*(1 - (d-C)*k)]
        f_prime = exp_neg_L * (1.0 - k * L * (1.0 - (d - C) * k))
        
        if abs(f_prime) < 1e-15:
            break
        
        d_new = d - f / f_prime
        
        # Clamp to valid range
        if d_new < 0:
            d_new = 0.0
        if d_new > 50.0 / k:
            d_new = 50.0 / k
        
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    # If d is very large, return inf (effectively no quote)
    if d > 20.0 / k:
        return np.inf
    
    return max(0.0, d)


def _precompute_depths(sigma, A, k, gamma, Q, steps, T=1.0):
    """
    Precompute optimal bid/ask depths using time-dependent quadratic value function.
    Uses backward induction with proper value function updates.
    """
    dt = T / steps
    nq = 2 * Q + 1
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Value function coefficients: V(q, t) = a(t) * q^2 + b(t) * q
    # We'll do backward induction, updating a and b at each step.
    
    # Terminal condition: V(q, T) = 0
    a_next = np.zeros(steps + 1)
    b_next = np.zeros(steps + 1)
    a_next[steps] = 0.0
    b_next[steps] = 0.0
    
    # Store depths for each time step and inventory level
    bid_depths = np.zeros((nq, steps))
    ask_depths = np.zeros((nq, steps))
    
    # Backward induction
    for t in range(steps - 1, -1, -1):
        a_t1 = a_next[t + 1]
        b_t1 = b_next[t + 1]
        
        # For each inventory level, compute optimal depths
        for i in range(nq):
            q = i - Q
            
            # Marginal cost of bid fill (q -> q+1)
            # Delta_b = V(q+1, t+1) - V(q, t+1) = a*(2q+1) + b
            C_b = a_t1 * (2 * q + 1) + b_t1
            
            # Marginal cost of ask fill (q -> q-1)
            # Delta_a = V(q-1, t+1) - V(q, t+1) = -a*(2q-1) + b
            C_a = -a_t1 * (2 * q - 1) + b_t1
            
            # Solve for optimal depths
            d_b = _solve_depth_for_cost(C_b, A, k, dt)
            d_a = _solve_depth_for_cost(-C_a, A, k, dt)  # Note: -C_a because we want to decrease inventory
            
            bid_depths[i, t] = d_b
            ask_depths[i, t] = d_a
        
        # Update value function coefficients
        # V(q, t) = (gamma/2)*sigma^2*q^2*dt + E[reward + V(q_next, t+1)]
        # 
        # For the quadratic approximation, we need to match coefficients.
        # The key insight: the expected change in V due to fills is:
        # E[Delta V] = p_b*(1-p_a)*Delta_b + (1-p_b)*p_a*Delta_a
        # where Delta_b = a_t1*(2q+1) + b_t1, Delta_a = -a_t1*(2q-1) + b_t1
        #
        # The reward is d_b*p_b + d_a*p_a, which depends on q through the depths.
        #
        # For the quadratic fit, we compute V(q, t) for several q values and fit.
        
        # Compute V(q, t) for a few representative q values
        q_sample = np.array([-Q, -Q//2, 0, Q//2, Q], dtype=np.float64)
        V_vals = np.zeros(len(q_sample))
        
        for j, q in enumerate(q_sample):
            i = int(q + Q)
            d_b = bid_depths[i, t]
            d_a = ask_depths[i, t]
            
            if np.isinf(d_b):
                p_b = 0.0
                d_b_eff = 0.0
            else:
                L_b = A * np.exp(-k * d_b) * dt
                p_b = 1.0 - np.exp(-L_b)
                d_b_eff = d_b
            
            if np.isinf(d_a):
                p_a = 0.0
                d_a_eff = 0.0
            else:
                L_a = A * np.exp(-k * d_a) * dt
                p_a = 1.0 - np.exp(-L_a)
                d_a_eff = d_a
            
            # Penalty
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Reward from fills
            reward = d_b_eff * p_b + d_a_eff * p_a
            
            # Expected future value change
            C_b = a_t1 * (2 * q + 1) + b_t1
            C_a = -a_t1 * (2 * q - 1) + b_t1
            
            # E[Delta V] = p_b*(1-p_a)*C_b + (1-p_b)*p_a*C_a
            E_delta_V = p_b * (1 - p_a) * C_b + (1 - p_b) * p_a * C_a
            
            # V(q, t) = penalty + reward + V(q, t+1) + E[Delta V]
            # But V(q, t+1) = a_t1*q^2 + b_t1*q
            V_q_t1 = a_t1 * q**2 + b_t1 * q
            V_vals[j] = penalty + reward + V_q_t1 + E_delta_V
        
        # Fit quadratic: V(q, t) = a_t * q^2 + b_t * q
        # Using least squares on the sample points
        # A_q = [[q^2, q], [q^2, q], ...]
        A_mat = np.column_stack([q_sample**2, q_sample])
        # Solve A_mat @ [a_t, b_t] = V_vals
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A_mat, V_vals, rcond=None)
            a_next[t] = coeffs[0]
            b_next[t] = coeffs[1]
        except:
            a_next[t] = a_t1
            b_next[t] = b_t1
    
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """Inventory-aware quoting with precomputed optimal depths."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    Q = params["Q"]
    steps = params["steps"]
    T = params["T"]
    
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        # Return default depths
        default_depth = 1.0 / k
        bid_depths = np.full(4000, default_depth, dtype=np.float64)
        ask_depths = np.full(4000, default_depth, dtype=np.float64)
        bid_depths[inventory >= Q] = np.inf
        ask_depths[inventory <= -Q] = np.inf
        return bid_depths, ask_depths
    
    # Check cache
    key = (sigma, A, k, gamma, Q, steps, T)
    if key not in _depth_cache:
        _depth_cache[key] = _precompute_depths(sigma, A, k, gamma, Q, steps, T)
    
    bid_depths_table, ask_depths_table = _depth_cache[key]
    
    # Get current time step index
    dt = T / steps
    t_idx = int(t / dt)
    t_idx = min(t_idx, steps - 1)
    t_idx = max(t_idx, 0)
    
    # Map inventory to index
    q = inventory
    indices = (q + Q).astype(np.int64)
    indices = np.clip(indices, 0, 2 * Q)
    
    bid_depths = bid_depths_table[indices, t_idx].astype(np.float64)
    ask_depths = ask_depths_table[indices, t_idx].astype(np.float64)
    
    # Enforce no bid fill at q = Q, no ask fill at q = -Q
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Ensure no invalid values
    bid_depths = np.where(bid_depths < -3.0/k, np.inf, bid_depths)
    ask_depths = np.where(ask_depths < -3.0/k, np.inf, ask_depths)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
