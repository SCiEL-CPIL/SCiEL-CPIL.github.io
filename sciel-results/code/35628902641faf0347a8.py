"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed DP tables per regime
_dp_cache = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s limit, leave 10s buffer

def _compute_analytical_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal bid/ask depths for each inventory level using DP 
    with analytical depth optimization based on first-order conditions.
    """
    dt = T / steps
    # Inventory levels: -Q to Q
    q_levels = np.arange(-Q, Q + 1)
    n_q = len(q_levels)
    
    # V[q_index] where q_index = q + Q
    # Initialize V for the last step (n = steps)
    # At n = steps, no more steps, so V = 0
    V = np.zeros(n_q, dtype=np.float64)
    
    # Store policies for all steps? 
    # The task asks for quote(t, ...) where t = n * dt.
    # We need to know the policy at each step n.
    # Since the policy depends on n (via V_{n+1}), we need to store 
    # bid_depths[n, q] and ask_depths[n, q].
    # Memory: 1000 steps * 61 inventory levels * 2 (bid/ask) * 8 bytes ~ 1MB. This is fine.
    
    bid_depths = np.full((steps, n_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, n_q), np.inf, dtype=np.float64)
    
    # Iterate backwards from n = steps - 1 to 0
    for n in range(steps - 1, -1, -1):
        # Check time
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            # Fallback to simple heuristic if running out of time
            # This shouldn't happen with analytical solution, but just in case
            break
            
        V_new = np.zeros(n_q, dtype=np.float64)
        
        for qi in range(n_q):
            q = q_levels[qi]
            
            # Inventory penalty term
            penalty = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            # Get V values for next step
            V_q = V[qi]
            
            # Calculate deltas
            # D_plus = V(q+1) - V(q)
            # D_minus = V(q-1) - V(q)
            
            # Bid side (q < Q)
            if qi < n_q - 1:
                D_plus = V[qi + 1] - V_q
                # Analytical optimal bid depth: b* = (1 - k * D_plus) / k
                b_opt = (1.0 - k * D_plus) / k
                if b_opt > 1e-6:
                    bid_depths[n, qi] = b_opt
                    # Calculate p_b
                    # p_b = 1 - exp(-A * exp(-k * b_opt) * dt)
                    # exp(-k * b_opt) = exp(-k * (1/k - D_plus)) = exp(-1 + k * D_plus) = e^-1 * exp(k * D_plus)
                    # Let's just compute directly to be safe
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                    V_q1 = V[qi + 1]
                else:
                    # No bid quote
                    p_b = 0.0
                    V_q1 = V_q # Doesn't matter as p_b is 0
            else:
                p_b = 0.0
                V_q1 = V_q
                
            # Ask side (q > -Q)
            if qi > 0:
                D_minus = V[qi - 1] - V_q
                # Analytical optimal ask depth: a* = (1 - k * D_minus) / k
                a_opt = (1.0 - k * D_minus) / k
                if a_opt > 1e-6:
                    ask_depths[n, qi] = a_opt
                    p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                    V_qm1 = V[qi - 1]
                else:
                    p_a = 0.0
                    V_qm1 = V_q
            else:
                p_a = 0.0
                V_qm1 = V_q

            # Calculate V_new[qi]
            # V_new = b*p_b + a*p_a - penalty + (1-p_b)(1-p_a)V_q + p_b(1-p_a)V_q1 + (1-p_b)p_a V_qm1 + p_b p_a V_q
            
            # Note: if p_b=0, terms with p_b vanish.
            # If we didn't quote bid, b*p_b is 0.
            # We need the actual b and a values used for the reward b*p_b.
            # If b_opt <= 1e-6, we set p_b=0, so reward is 0.
            
            # Let's reconstruct b and a for reward calculation
            if qi < n_q - 1:
                D_plus = V[qi + 1] - V_q
                b_opt = (1.0 - k * D_plus) / k
                if b_opt > 1e-6:
                    b_val = b_opt
                else:
                    b_val = 0.0
            else:
                b_val = 0.0
                
            if qi > 0:
                D_minus = V[qi - 1] - V_q
                a_opt = (1.0 - k * D_minus) / k
                if a_opt > 1e-6:
                    a_val = a_opt
                else:
                    a_val = 0.0
            else:
                a_val = 0.0

            # Reward from fills
            reward = b_val * p_b + a_val * p_a
            
            # Expected next value
            # P(no fill) = (1-p_b)(1-p_a)
            # P(bid fill) = p_b(1-p_a)
            # P(ask fill) = (1-p_b)p_a
            # P(both) = p_b p_a
            
            exp_next = (1.0 - p_b) * (1.0 - p_a) * V_q + \
                       p_b * (1.0 - p_a) * V_q1 + \
                       (1.0 - p_b) * p_a * V_qm1 + \
                       p_b * p_a * V_q
            
            V_new[qi] = reward - penalty + exp_next

        V = V_new

    return bid_depths, ask_depths

def _get_policy(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the optimal policy for given parameters."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _dp_cache:
        # Check time budget
        elapsed = time.time() - _START_TIME
        if elapsed > TIME_LIMIT - 10:
            # Return a simple heuristic if we are very close to timeout
            # This is a fallback, the analytical DP should be fast enough
            depth = 1.0 / k
            bid = np.full((steps, 2 * Q + 1), depth)
            ask = np.full((steps, 2 * Q + 1), depth)
            # Set inf at boundaries
            bid[:, 2*Q] = np.inf # q=Q
            ask[:, 0] = np.inf # q=-Q
            _dp_cache[key] = (bid, ask)
            return _dp_cache[key]
        
        bid, ask = _compute_analytical_policy(sigma, A, k, gamma, T, steps, Q)
        _dp_cache[key] = (bid, ask)
    return _dp_cache[key]

def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy lookup."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]

    bid_table, ask_table = _get_policy(sigma, A, k, gamma, T, steps, Q)

    # Determine current step index
    dt = T / steps
    n = int(round(t / dt))
    # Clamp n to valid range [0, steps-1]
    n = np.clip(n, 0, steps - 1)

    # Map inventory to index: q in [-Q, Q] -> index q + Q
    q_indices = (inventory + Q).astype(np.int64)
    # Clamp to valid range
    q_indices = np.clip(q_indices, 0, 2 * Q)

    # Retrieve depths for the current step and inventory levels
    bid_depths = bid_table[n, q_indices].astype(np.float64)
    ask_depths = ask_table[n, q_indices].astype(np.float64)

    # Ensure no negative depths or invalid values
    # The analytical solution might produce very small negative numbers due to floating point
    # We treat any depth <= 0 as no quote (inf)
    bid_depths[bid_depths <= 1e-6] = np.inf
    ask_depths[ask_depths <= 1e-6] = np.inf

    # Enforce boundary constraints: no bid at q=Q, no ask at q=-Q
    # These should already be inf from the DP, but double check
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf

    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
