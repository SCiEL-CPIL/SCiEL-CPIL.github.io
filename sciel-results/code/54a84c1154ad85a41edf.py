"""
Market making with inventory risk: quadratic value function approximation.

Structural approach: approximate V(q,t) = alpha(t) * q^2 (symmetric, no linear term).
Backward-induct alpha(t) using the first-order condition of the Bellman equation.
This captures time-varying risk aversion and gives an O(N) closed-form policy.
"""
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330.0
_TIME_MARGIN = 15.0

# Cache for alpha(t) values per regime
_alpha_cache = {}


def _compute_alpha(sigma, A, k, gamma, T, steps, Q):
    """
    Compute alpha(t) for t = 0, dt, 2dt, ..., T via backward induction.
    
    V(q,t) = alpha(t) * q^2
    Bellman: alpha(t)*q^2 = max_{b,a} [
        p_b*b + p_a*a - (gamma/2)*sigma^2*q^2*dt 
        + alpha(t+dt)*(q^2 + 2q*(p_b - p_a) + p_b + p_a)
    ]
    
    For small dt, p_b ≈ A*exp(-k*b)*dt, p_a ≈ A*exp(-k*a)*dt.
    
    The FOC for optimal b (treating a independently by symmetry of structure):
    d/db [p_b*b + alpha_next*(2q*p_b + p_b)] - 0 = 0
    (since -gamma/2*sigma^2*q^2*dt doesn't depend on b)
    
    dp_b/db = -k*A*exp(-k*b)*dt = -k*p_b
    
    d/db [p_b*(b + alpha_next*(2q+1))] = dp_b/db*(b + alpha_next*(2q+1)) + p_b*1
    = -k*p_b*(b + alpha_next*(2q+1)) + p_b
    = p_b * [1 - k*b - k*alpha_next*(2q+1)]
    
    Setting to 0: 1 - k*b - k*alpha_next*(2q+1) = 0
    => b = (1 - k*alpha_next*(2q+1)) / k = 1/k - alpha_next*(2q+1)
    
    Similarly for ask: a = 1/k - alpha_next*(2q-1)  [by symmetry, q->-q]
    Wait, let me redo. For ask, delta_q = -1:
    d/da [p_a*a + alpha_next*(-2q*p_a + p_a)] = 0
    dp_a/da = -k*p_a
    d/da [p_a*(a + alpha_next*(1-2q))] = -k*p_a*(a + alpha_next*(1-2q)) + p_a = 0
    => 1 - k*a - k*alpha_next*(1-2q) = 0
    => a = 1/k - alpha_next*(1-2q)
    
    Now, to find alpha(t), we substitute the optimal b, a back into the Bellman equation.
    But this is complex because b, a depend on q. Instead, we can compute alpha(t) 
    by evaluating the Bellman equation at a representative q, or by averaging.
    
    Actually, the cleanest approach: the Bellman equation with quadratic V should hold
    for all q. Let's substitute the optimal depths and match coefficients of q^2.
    
    At optimal b: b* = max(0, 1/k - alpha_next*(2q+1))
    At optimal a: a* = max(0, 1/k - alpha_next*(1-2q))
    
    For the regime parameters, let's check if depths are positive for all q in [-Q,Q].
    With alpha ~ O(0.1-1), 1/k ~ 0.67, and 2q+1 up to 61, we may get negative depths.
    
    This is getting complex. Let me use a simpler but still structurally different approach:
    precompute the optimal policy via the FOC for each q in [-Q,Q] at each time step,
    using the quadratic V approximation, and store as a lookup table.
    
    Actually, for speed, let me just compute alpha(t) numerically by solving the
    scalar equation that arises from the Bellman equation averaged over q.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _alpha_cache:
        return _alpha_cache[key]
    
    dt = T / steps
    n_steps = steps + 1  # t = 0, dt, ..., T
    alpha = np.zeros(n_steps)
    alpha[-1] = 0.0  # V(q, T) = 0 (no terminal charge)
    
    # For each time step backward, compute alpha(t) from alpha(t+dt)
    # Using the FOC: b*(q) = 1/k - alpha_next*(2q+1), a*(q) = 1/k - alpha_next*(1-2q)
    # 
    # The Bellman equation at state q:
    # alpha(t)*q^2 = p_b*b* + p_a*a* - (gamma/2)*sigma^2*q^2*dt
    #               + alpha_next*(q^2 + 2q*(p_b - p_a) + p_b + p_a)
    #
    # where p_b = 1-exp(-A*exp(-k*b*)*dt), p_a similarly.
    #
    # This must hold for all q. Let's evaluate at q=0 and q=1 to get equations,
    # or better, use the fact that with quadratic V, we can match the q^2 coefficient.
    #
    # Actually, the simplest robust approach: pick a grid of q values, compute the
    # implied alpha(t) from the Bellman equation at each q, and take the median.
    
    q_grid = np.array([0.0, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0])
    q_grid = q_grid[np.abs(q_grid) <= Q]
    if len(q_grid) == 0:
        q_grid = np.array([0.0])
    
    for i in range(n_steps - 2, -1, -1):
        alpha_next = alpha[i + 1]
        implied_alphas = []
        
        for q in q_grid:
            # Optimal depths
            b_star = 1.0 / k - alpha_next * (2.0 * q + 1.0)
            a_star = 1.0 / k - alpha_next * (1.0 - 2.0 * q)
            
            # Boundary conditions
            if q >= Q:
                b_star = np.inf
            if q <= -Q:
                a_star = np.inf
            
            # Clip to positive
            if b_star <= 0:
                b_star = np.inf
            if a_star <= 0:
                a_star = np.inf
            
            # Fill probabilities
            if np.isinf(b_star):
                p_b = 0.0
            else:
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_star) * dt)
            
            if np.isinf(a_star):
                p_a = 0.0
            else:
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_star) * dt)
            
            # Bellman RHS
            # Immediate reward
            reward_b = p_b * b_star if np.isfinite(b_star) else 0.0
            reward_a = p_a * a_star if np.isfinite(a_star) else 0.0
            
            # Risk penalty
            risk = (gamma / 2.0) * (sigma ** 2) * q * q * dt
            
            # Future value
            delta_q = p_b - p_a  # E[delta_q]
            var_term = p_b + p_a  # E[delta_q^2] approx for small p
            future = alpha_next * (q * q + 2.0 * q * delta_q + var_term)
            
            rhs = reward_b + reward_a - risk + future
            
            # alpha(t)*q^2 = rhs => alpha(t) = rhs / q^2 (if q != 0)
            if abs(q) > 1e-10:
                implied_alphas.append(rhs / (q * q))
            else:
                # At q=0: alpha(t)*0 = rhs => rhs should be 0 for consistency
                # Use a different approach: differentiate Bellman eq w.r.t. q at q=0
                # Or just skip q=0
                pass
        
        if implied_alphas:
            alpha[i] = np.median(implied_alphas)
            # Ensure non-negative (risk aversion should be non-decreasing backward)
            alpha[i] = max(0.0, alpha[i])
        else:
            alpha[i] = alpha_next
    
    _alpha_cache[key] = alpha
    return alpha


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) using quadratic V approximation.
    """
    N = len(inventory)
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # Deadline check
    if (time.time() - _START_TIME) > (_DEADLINE - _TIME_MARGIN):
        # Fallback: seed policy
        depth = 1.0 / k
        bid = np.full(N, depth, dtype=np.float64)
        ask = np.full(N, depth, dtype=np.float64)
        return bid, ask
    
    # Get alpha(t)
    alpha_arr = _compute_alpha(sigma, A, k, gamma, T, steps, Q)
    
    # Map t to index
    t_idx = int(round(t / dt))
    t_idx = min(t_idx, steps)
    alpha_t = alpha_arr[t_idx]
    
    q = np.asarray(inventory, dtype=np.float64)
    
    # Optimal depths from FOC
    bid_depth = 1.0 / k - alpha_t * (2.0 * q + 1.0)
    ask_depth = 1.0 / k - alpha_t * (1.0 - 2.0 * q)
    
    # Boundary conditions
    bid_depth[q >= Q] = np.inf
    ask_depth[q <= -Q] = np.inf
    
    # Clip negative depths to inf (no quote)
    bid_depth[bid_depth <= 0] = np.inf
    ask_depth[ask_depth <= 0] = np.inf
    
    # Safety: replace NaN, -inf
    bid_depth = np.where(np.isnan(bid_depth) | (bid_depth == -np.inf), np.inf, bid_depth)
    ask_depth = np.where(np.isnan(ask_depth) | (ask_depth == -np.inf), np.inf, ask_depth)
    
    # Ensure valid: >= -3/k (should be positive or inf)
    min_valid = -3.0 / k
    bid_depth = np.where(bid_depth < min_valid, np.inf, bid_depth)
    ask_depth = np.where(ask_depth < min_valid, np.inf, ask_depth)
    
    bid_depth = np.asarray(bid_depth, dtype=np.float64)
    ask_depth = np.asarray(ask_depth, dtype=np.float64)
    
    return bid_depth, ask_depth
