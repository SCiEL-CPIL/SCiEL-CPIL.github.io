"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

sys.setrecursionlimit(100000)

_start_time = time.time()
_DEADLINE = 310.0  # safety margin before 330s limit

# Pre-computed policy tables for each regime
_policy_cache = {}

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q, n_grid=501):
    """
    Compute optimal bid/ask depths for each inventory level and time step
    using a semi-Lagrangian HJB approximation.
    
    Returns arrays bid_depths[q_idx, t_idx] and ask_depths[q_idx, t_idx]
    where q_idx maps to inventory levels from -Q to Q.
    """
    dt = T / steps
    tau = T  # total time
    
    # Inventory grid: -Q to Q
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    n_q = len(q_vals)
    
    # Time grid: 0 to T, but we need values at each step
    # We'll compute backwards from T to 0
    
    # Value function V[t, q] and derivatives
    # V[q] at current time step
    V = np.zeros(n_q)
    # dV/dq at current time step (first derivative)
    dV = np.zeros(n_q)
    
    # Pre-compute for each time step, store the optimal depths
    bid_table = np.zeros((n_q, steps))
    ask_table = np.zeros((n_q, steps))
    
    # We'll iterate backwards in time
    # At time T, V = 0 (no terminal charge)
    
    for n in range(steps - 1, -1, -1):
        t = n * dt
        tau_remaining = T - t  # time remaining
        
        # For each inventory level, compute optimal depths
        for i in range(n_q):
            q = q_vals[i]
            
            # First-order condition for optimal depth:
            # The objective contribution from quoting at depth d for bid:
            #   gain: d (if filled)
            #   cost: change in inventory -> affects future value
            #   
            # Probability of bid fill: p_b = 1 - exp(-A * exp(-k*d) * dt)
            # Expected immediate reward from bid: p_b * d
            # Expected inventory change from bid: -p_b (inventory decreases by 1)
            # 
            # For a small depth change dd, the change in expected value:
            #   dE[V] = p_b * d - p_b * (V[q-1] - V[q]) - p_b * dd * (dV/dq)
            #   Actually, let's think more carefully.
            #
            # The HJB equation in continuous approximation:
            #   V_t + (1/2)sigma^2 V_qq - gamma/2 * sigma^2 * q^2 
            #   + max_{d_b, d_a} [ (1-exp(-A e^{-k d_b} dt)) * (d_b + V_{q-1} - V_q)
            #                      + (1-exp(-A e^{-k d_a} dt)) * (d_a + V_{q+1} - V_q) ] = 0
            #
            # For the bid depth, the first-order condition:
            #   d/d d_b [ (1-exp(-A e^{-k d_b} dt)) * (d_b + delta_V_b) ] = 0
            #   where delta_V_b = V[q-1] - V[q] (change in value when inventory decreases by 1)
            #
            # Let p = 1 - exp(-A e^{-k d} dt)
            # d(p*(d + delta_V))/dd = p' * (d + delta_V) + p
            # where p' = d p / dd = A * k * exp(-k*d) * exp(-A e^{-k d} dt) * dt
            #          = A * k * exp(-k*d) * (1-p) * dt
            #
            # Setting to zero:
            #   A*k*exp(-k*d)*(1-p)*dt*(d + delta_V) + (1 - exp(-A e^{-k d} dt)) = 0
            #
            # This is implicit. For small dt, p ≈ A*exp(-k*d)*dt, so:
            #   A*k*exp(-k*d)*dt*(d + delta_V) + A*exp(-k*d)*dt ≈ 0
            #   k*(d + delta_V) + 1 ≈ 0
            #   d ≈ -1/k - delta_V
            #
            # For the ask:
            #   d ≈ -1/k - delta_V_a where delta_V_a = V[q+1] - V[q]
            #
            # So: d_bid ≈ -1/k - (V[q-1] - V[q]) = -1/k + V[q] - V[q-1]
            #     d_ask ≈ -1/k - (V[q+1] - V[q]) = -1/k + V[q] - V[q+1]
            
            # Approximate derivatives using central differences
            if i > 0:
                delta_V_b = V[i] - V[i-1]  # V[q] - V[q-1]
            else:
                delta_V_b = 0.0  # boundary
            
            if i < n_q - 1:
                delta_V_a = V[i] - V[i+1]  # V[q] - V[q+1]
            else:
                delta_V_a = 0.0  # boundary
            
            d_bid_opt = -1.0/k + delta_V_b
            d_ask_opt = -1.0/k + delta_V_a
            
            # Ensure depths are valid (not too small)
            min_depth = -3.0/k + 1e-9
            d_bid_opt = max(d_bid_opt, min_depth)
            d_ask_opt = max(d_ask_opt, min_depth)
            
            # Boundary conditions: no bid at q=Q, no ask at q=-Q
            if q >= Q:
                d_bid_opt = np.inf
            if q <= -Q:
                d_ask_opt = np.inf
            
            bid_table[i, n] = d_bid_opt
            ask_table[i, n] = d_ask_opt
        
        # Update value function for previous time step
        # V_t = -1/dt * (other terms)
        # We need to compute the max over depths of the fill terms
        # For simplicity, use the optimal depths we just computed
        
        V_new = np.zeros(n_q)
        for i in range(n_q):
            q = q_vals[i]
            
            # Inventory cost
            inv_cost = gamma / 2.0 * sigma**2 * q**2 * dt
            
            # Fill rewards using optimal depths
            d_b = bid_table[i, n]
            d_a = ask_table[i, n]
            
            if np.isfinite(d_b):
                p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
                # Inventory change: -1 (bid fill decreases inventory)
                if i > 0:
                    fill_b = p_b * (d_b + V[i-1] - V[i])
                else:
                    fill_b = p_b * d_b
            else:
                fill_b = 0.0
            
            if np.isfinite(d_a):
                p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
                # Inventory change: +1 (ask fill increases inventory)
                if i < n_q - 1:
                    fill_a = p_a * (d_a + V[i+1] - V[i])
                else:
                    fill_a = p_a * d_a
            else:
                fill_a = 0.0
            
            # Diffusion term: (1/2) sigma^2 V_qq
            if 0 < i < n_q - 1:
                V_qq = (V[i+1] - 2*V[i] + V[i-1])
                diff_term = 0.5 * sigma**2 * V_qq * dt
            else:
                diff_term = 0.0
            
            V_new[i] = V[i] - dt * (inv_cost - diff_term - fill_b - fill_a)
        
        V = V_new
    
    return bid_table, ask_table


def _get_policy_table(sigma, A, k, gamma, T, steps, Q):
    """Get or compute the policy table for given parameters."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Optimal market making quote with inventory skewing.
    
    Uses pre-computed policy table from semi-Lagrangian HJB approximation.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        k = params["k"]
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    q = np.asarray(inventory, dtype=np.int32)
    
    # Get policy table
    try:
        bid_table, ask_table = _get_policy_table(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        # Fallback to analytical approximation
        d_0 = 1.0 / k
        alpha = gamma * sigma * sigma * dt * max(1, steps - n)
        offset = 0.5 * alpha
        qf = q.astype(np.float64)
        d_bid = d_0 + alpha * qf + offset
        d_ask = d_0 - alpha * qf + offset
        min_depth = -3.0 / k + 1e-6
        max_depth = 50.0 / k
        d_bid = np.clip(d_bid, min_depth, max_depth)
        d_ask = np.clip(d_ask, min_depth, max_depth)
        d_bid = np.where(q >= Q, np.inf, d_bid)
        d_ask = np.where(q <= -Q, np.inf, d_ask)
        return (d_bid.astype(np.float64), d_ask.astype(np.float64))
    
    # Map inventory to table indices
    # Table index: q + Q (since q ranges from -Q to Q)
    q_idx = q + Q
    q_idx = np.clip(q_idx, 0, 2*Q)
    
    d_bid = bid_table[q_idx, n].astype(np.float64)
    d_ask = ask_table[q_idx, n].astype(np.float64)
    
    # Apply boundary conditions explicitly
    d_bid = np.where(q >= Q, np.inf, d_bid)
    d_ask = np.where(q <= -Q, np.inf, d_ask)
    
    # Ensure validity
    min_depth = -3.0 / k + 1e-9
    d_bid = np.where(np.isfinite(d_bid), np.maximum(d_bid, min_depth), d_bid)
    d_ask = np.where(np.isfinite(d_ask), np.maximum(d_ask, min_depth), d_ask)
    
    return (d_bid, d_ask)

# EVOLVE-BLOCK-END
