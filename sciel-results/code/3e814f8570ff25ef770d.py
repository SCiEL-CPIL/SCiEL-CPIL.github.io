"""Consumption-saving policy via backward induction with FOC-based continuous search."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    first-order condition (FOC) based continuous search for optimal consumption.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return 1.0 / c_safe
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, 1e-12), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, 1e-12)
            return np.power(c_safe, -gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # Helper: compute expected continuation value given m' for each z'
    def expected_continuation(m_prime, z, V_next_arr):
        """
        m_prime: shape (n_z,) - next period resources for each z'
        z: current state
        Returns scalar expected continuation value
        """
        m_prime_clipped = np.clip(m_prime, m_min, m_max)
        cont = np.zeros(n_z)
        for zp in range(n_z):
            cont[zp] = np.interp(m_prime_clipped[zp], m_grid, V_next_arr[zp, :])
        return np.dot(transition[z], cont)
    
    # Helper: compute objective and its derivative at consumption c
    def objective_and_deriv(c, m_val, z, V_next_arr):
        """
        Returns (value, derivative) of the Bellman objective at consumption c.
        """
        c_safe = np.maximum(c, 1e-12)
        m_minus_c = np.maximum(m_val - c_safe, 0.0)
        m_prime = R * m_minus_c + income  # shape (n_z,)
        m_prime_clipped = np.clip(m_prime, m_min, m_max)
        
        # Continuation value
        cont = np.zeros(n_z)
        for zp in range(n_z):
            cont[zp] = np.interp(m_prime_clipped[zp], m_grid, V_next_arr[zp, :])
        cont_val = np.dot(transition[z], cont)
        
        # Value
        val = utility(c_safe) + beta * cont_val
        
        # Derivative: u'(c) - beta * R * sum_z' trans[z,z'] * V'_next[z', m'_z']
        # V'_next is the derivative of the value function w.r.t. m
        mu_c = marginal_utility(c_safe)
        
        # Need V'_next: derivative of V_next w.r.t. m
        # Approximate via finite differences on the grid
        # V'_next[zp, m_idx] ≈ (V_next[zp, m_idx+1] - V_next[zp, m_idx-1]) / (m_grid[m_idx+1] - m_grid[m_idx-1])
        
        # For each z', we need V'_next[zp] evaluated at m_prime_clipped[zp]
        # Use interpolation of the derivative
        # First compute V'_next on the grid
        V_prime_next = np.zeros((n_z, n_m))
        for zp in range(n_z):
            V_prime_next[zp, 0] = (V_next_arr[zp, 1] - V_next_arr[zp, 0]) / (m_grid[1] - m_grid[0])
            V_prime_next[zp, -1] = (V_next_arr[zp, -1] - V_next_arr[zp, -2]) / (m_grid[-1] - m_grid[-2])
            if n_m > 2:
                V_prime_next[zp, 1:-1] = (V_next_arr[zp, 2:] - V_next_arr[zp, :-2]) / (m_grid[2:] - m_grid[:-2])
        
        # Interpolate V'_next at m_prime_clipped
        V_prime_at_mp = np.zeros(n_z)
        for zp in range(n_z):
            V_prime_at_mp[zp] = np.interp(m_prime_clipped[zp], m_grid, V_prime_next[zp, :])
        
        # Derivative of continuation w.r.t. c:
        # d/dc [beta * sum_z' trans[z,z'] * V_next[z', R*(m-c) + income[z']]]
        # = -beta * R * sum_z' trans[z,z'] * V'_next[z', m'_z']
        deriv = mu_c - beta * R * np.dot(transition[z], V_prime_at_mp)
        
        return val, deriv
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # Precompute V'_next for this iteration
            V_prime_next = np.zeros((n_z, n_m))
            for zp in range(n_z):
                V_prime_next[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (m_grid[1] - m_grid[0])
                V_prime_next[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (m_grid[-1] - m_grid[-2])
                if n_m > 2:
                    V_prime_next[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (m_grid[2:] - m_grid[:-2])
            
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                
                # Use bisection on the FOC: find c such that objective'(c) = 0
                # c in (0, m_val)
                c_lo = 1e-12
                c_hi = m_val
                
                # Check if the FOC has a solution in the interior
                _, deriv_lo = objective_and_deriv(c_lo, m_val, z, V_next)
                _, deriv_hi = objective_and_deriv(c_hi, m_val, z, V_next)
                
                if deriv_lo <= 0:
                    # Derivative is non-positive at lower bound, optimum at lower bound
                    c_opt = c_lo
                elif deriv_hi >= 0:
                    # Derivative is non-negative at upper bound, optimum at upper bound
                    c_opt = c_hi
                else:
                    # Bisection to find where derivative crosses zero
                    # The derivative is decreasing in c (concavity), so it goes from positive to negative
                    c_lo_b = c_lo
                    c_hi_b = c_hi
                    for _ in range(50):
                        c_mid = 0.5 * (c_lo_b + c_hi_b)
                        _, deriv_mid = objective_and_deriv(c_mid, m_val, z, V_next)
                        if deriv_mid > 0:
                            c_lo_b = c_mid
                        else:
                            c_hi_b = c_mid
                        if abs(c_hi_b - c_lo_b) < 1e-14 * m_val:
                            break
                    c_opt = 0.5 * (c_lo_b + c_hi_b)
                
                # Clamp to valid range
                c_opt = np.clip(c_opt, 1e-12, m_val)
                
                C_opt[z, m_idx] = c_opt
                V_curr[z, m_idx], _ = objective_and_deriv(c_opt, m_val, z, V_next)
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if c_val <= 0:
                    consumption[t, z, m_idx] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
# EVOLVE-BLOCK-END
