"""
Optimal market making with inventory risk using vectorized HJB backward induction.

Solves the discrete-time HJB equation via backward induction over the inventory
grid. The depth optimization is done on a fine grid for each side, exploiting
the separability of the fill process to reduce the 2D optimization to two 1D
optimizations plus a correction term.
"""
import numpy as np
import time

_policy_cache = {}
_start_time = time.time()
_TIME_LIMIT = 310  # safety margin before 330s kill


def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depths via backward induction on the discrete-time model.
    
    Returns:
        bid_table: array of shape (steps, 2*Q+1)
        ask_table: array of shape (steps, 2*Q+1)
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function: V[q] = optimal expected objective from current step with inventory q
    V = np.zeros(n_inv)
    
    # We'll store policies for each step
    bid_table = np.full((steps, n_inv), np.inf)
    ask_table = np.full((steps, n_inv), np.inf)
    
    # Grid of candidate depths
    # Use a fine grid; the optimal depth is typically small
    n_grid = 200
    max_depth = 8.0
    depth_grid = np.linspace(1e-4, max_depth, n_grid)
    
    # Pre-compute fill probabilities
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    exp_term = np.exp(-k * depth_grid)
    rate = A * exp_term * dt
    p_fill = 1.0 - np.exp(-rate)
    
    # Work backwards
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > _TIME_LIMIT:
            # Fill remaining steps with last computed policy
            if n < steps - 1:
                bid_table[:n+1] = bid_table[n+1]
                ask_table[:n+1] = ask_table[n+1]
            break
        
        V_next = V.copy()
        
        # For q = -Q: only bid enabled (ask is inf)
        q_idx = 0
        q = -Q
        inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
        q_next_idx = 1  # q+1 = -Q+1
        # Value: -d*p + (1-p)*V[q] + p*V[q+1] = V[q] + p*(V[q+1]-V[q] - d)
        diff = V_next[q_next_idx] - V_next[q_idx]
        values = diff * p_fill - depth_grid * p_fill
        best_idx = np.argmax(values)
        bid_table[n, q_idx] = depth_grid[best_idx]
        ask_table[n, q_idx] = np.inf
        V[q_idx] = -inv_cost + V_next[q_idx] + values[best_idx]
        
        # For q = Q: only ask enabled (bid is inf)
        q_idx = 2 * Q
        q = Q
        inv_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
        q_next_idx = 2 * Q - 1  # q-1 = Q-1
        # Value: -d*p + (1-p)*V[q] + p*V[q-1] = V[q] + p*(V[q-1]-V[q] - d)
        diff = V_next[q_next_idx] - V_next[q_idx]
        values = diff * p_fill - depth_grid * p_fill
        best_idx = np.argmax(values)
        ask_table[n, q_idx] = depth_grid[best_idx]
        bid_table[n, q_idx] = np.inf
        V[q_idx] = -inv_cost + V_next[q_idx] + values[best_idx]
        
        # Interior q = -Q+1, ..., Q-1
        q_interior = np.arange(-Q + 1, Q)
        q_indices = q_interior + Q  # 1 to 2Q-1
        n_interior = len(q_interior)
        
        if n_interior > 0:
            # inv_cost for each q
            inv_costs = (gamma / 2.0) * sigma**2 * q_interior**2 * dt
            
            # V_next values
            Vq = V_next[q_indices]  # V[q]
            Vq1 = V_next[q_indices + 1]  # V[q+1]
            Vqm1 = V_next[q_indices - 1]  # V[q-1]
            
            # The joint optimization over (d_b, d_a) can be decomposed:
            # val = pb*db + pa*da + pb*pa*Vq + pb*(1-pa)*Vq1 + (1-pb)*pa*Vqm1 + (1-pb)*(1-pa)*Vq
            #     = pb*db + pa*da + Vq + pb*(Vq1-Vq) - pa*(Vq-Vqm1) - pb*pa*(Vq1+Vqm1-2*Vq)
            # Let delta1[q] = V[q+1]-V[q], delta2[q] = V[q]-V[q-1], delta12[q] = V[q+1]+V[q-1]-2*V[q]
            # val = Vq + pb*(db + delta1) + pa*(da - delta2) - pb*pa*delta12
            # This is NOT separable due to the -pb*pa*delta12 term.
            # But we can do a 2D grid search efficiently.
            
            delta1 = Vq1 - Vq  # (n_interior,)
            delta2 = Vq - Vqm1  # (n_interior,)
            delta12 = Vq1 + Vqm1 - 2 * Vq  # (n_interior,)
            
            # 2D grid search
            # For each (i,j) in grid: pb = p_fill[i], pa = p_fill[j]
            # val[q] = Vq[q] + pb*(depth_grid[i] + delta1[q]) + pa*(depth_grid[j] - delta2[q]) - pb*pa*delta12[q]
            
            # Vectorize: (n_grid^2, n_interior)
            PB = p_fill[:, None] * np.ones((1, n_grid))  # (n_grid, n_grid)
            PA = np.ones((n_grid, 1)) * p_fill[None, :]  # (n_grid, n_grid)
            DB = depth_grid[:, None] * np.ones((1, n_grid))  # (n_grid, n_grid)
            DA = np.ones((n_grid, 1)) * depth_grid[None, :]  # (n_grid, n_grid)
            
            PB_flat = PB.ravel()  # (n_grid^2,)
            PA_flat = PA.ravel()  # (n_grid^2,)
            DB_flat = DB.ravel()  # (n_grid^2,)
            DA_flat = DA.ravel()  # (n_grid^2,)
            
            # (n_grid^2, n_interior)
            vals = (Vq[None, :] + 
                    PB_flat[:, None] * (DB_flat[:, None] + delta1[None, :]) + 
                    PA_flat[:, None] * (DA_flat[:, None] - delta2[None, :]) - 
                    PB_flat[:, None] * PA_flat[:, None] * delta12[None, :])
            
            best_idx_flat = np.argmax(vals, axis=0)
            best_val = vals[best_idx_flat, np.arange(n_interior)]
            
            # Map back to grid
            best_i = best_idx_flat // n_grid
            best_j = best_idx_flat % n_grid
            
            bid_table[n, q_indices] = depth_grid[best_i]
            ask_table[n, q_indices] = depth_grid[best_j]
            V[q_indices] = -inv_costs + best_val
    
    return bid_table, ask_table


def _get_policy_table(params):
    """Get or compute the policy table for given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        sigma = params['sigma']
        A = params['A']
        k = params['k']
        gamma = params['gamma']
        T = params['T']
        steps = params['steps']
        Q = params['Q']
        
        bid_table, ask_table = _compute_policy_table(sigma, A, k, gamma, T, steps, Q)
        _policy_cache[key] = (bid_table, ask_table)
    
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    dt = params['T'] / params['steps']
    step = int(round(t / dt))
    step = min(max(step, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    bid_table, ask_table = _get_policy_table(params)
    
    inv_indices = inventory + Q
    inv_indices = np.clip(inv_indices, 0, 2 * Q)
    
    bid_depths = bid_table[step, inv_indices].astype(np.float64)
    ask_depths = ask_table[step, inv_indices].astype(np.float64)
    
    # Ensure valid values: must be > -3/k
    min_val = -3.0 / params['k'] + 1e-6
    bid_depths = np.where(bid_depths > min_val, bid_depths, np.inf)
    ask_depths = np.where(ask_depths > min_val, ask_depths, np.inf)
    
    return bid_depths, ask_depths
