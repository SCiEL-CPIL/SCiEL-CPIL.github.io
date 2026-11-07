"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def solve_consumption(model, resource_grid, budget):
    """
    Solve consumption-saving policy using Chebyshev spectral collocation
    for the value function representation.
    
    Returns consumption[t, income_state, resource_index] with shape (6, 2, 101).
    """
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    # Resource grid
    m_grid = resource_grid.copy()
    
    # Chebyshev nodes on [0, 1] mapped to resource grid
    # We'll use the given resource_grid as our collocation points
    # But for polynomial interpolation, we need to map to [-1, 1]
    
    # Map resource_grid to [-1, 1]
    m_min = m_grid[0]
    m_max = m_grid[-1]
    # Use log mapping since resource_grid is geomspace
    log_m = np.log(m_grid)
    log_m_min = log_m[0]
    log_m_max = log_m[-1]
    
    # Chebyshev points in log-space
    # Actually, let's use the grid directly and fit polynomials in log(m) space
    
    # For each time step, we'll represent V(m, z) as a polynomial in log(m)
    # using Chebyshev basis
    
    # Number of Chebyshev coefficients
    n_cheb = min(25, n_grid)  # degree ~24
    
    # Precompute Chebyshev basis matrix
    # x = normalized log(m) in [-1, 1]
    x = 2.0 * (log_m - log_m_min) / (log_m_max - log_m_min) - 1.0
    
    # Chebyshev polynomial values T_k(x) for k=0..n_cheb-1
    T = np.zeros((n_grid, n_cheb))
    T[:, 0] = 1.0
    if n_cheb > 1:
        T[:, 1] = x
    for k in range(2, n_cheb):
        T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
    
    # Inverse of T matrix (for fitting coefficients)
    # T is n_grid x n_cheb, so we need pseudo-inverse or least squares
    # Since n_grid >= n_cheb, use lstsq
    T_inv = np.linalg.pinv(T)  # n_cheb x n_grid
    
    def crra_utility(c, gamma):
        """CRRA utility with floor on c."""
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    # Consumption grid for optimization
    # c = m * fraction, fraction in [0.001, 1]
    n_frac = budget  # use budget as number of consumption fractions
    frac_grid = np.linspace(0.001, 1.0, n_frac)
    
    # Value function coefficients for next period
    # V_next[z] = Chebyshev coefficients for state z
    V_next = np.zeros((n_states, n_cheb))
    
    # Store consumption policies
    consumption = np.zeros((horizon, n_states, n_grid))
    
    for t in range(horizon - 1, -1, -1):
        if t == horizon - 1:
            # Terminal: consume all resources
            for z in range(n_states):
                consumption[t, z, :] = m_grid
            # V at terminal is utility of consuming all
            # But we don't really need V at terminal for the policy
            continue
        
        for z in range(n_states):
            # For each income state z, compute optimal consumption
            # V_t(m, z) = max_c [ u(c) + beta * sum_z' trans[z,z'] * V_{t+1}(R*(m-c)+income[z'], z') ]
            
            # We need to evaluate V_next for each possible next state
            # For each m in m_grid, try each c = m * frac
            best_c = np.zeros(n_grid)
            best_val = np.zeros(n_grid)
            
            for i, m in enumerate(m_grid):
                c_candidates = m * frac_grid
                c_candidates = np.minimum(c_candidates, m)  # can't consume more than m
                c_candidates = np.maximum(c_candidates, 1e-12)
                
                # Utility of current consumption
                u_current = crra_utility(c_candidates, gamma)
                
                # Next resources for each c
                # m_next = R * (m - c) + income[z']
                # For each z', compute expected continuation value
                cont_val = np.zeros(len(c_candidates))
                for zp in range(n_states):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_next = R * (m - c_candidates) + income[zp]
                    m_next = np.maximum(m_next, 1e-12)
                    
                    # Evaluate V_next[zp] at m_next using Chebyshev interpolation
                    # Map m_next to x-space
                    log_m_next = np.log(m_next)
                    x_next = 2.0 * (log_m_next - log_m_min) / (log_m_max - log_m_min) - 1.0
                    x_next = np.clip(x_next, -1.0, 1.0)
                    
                    # Chebyshev polynomial values at x_next
                    T_next = np.zeros((len(x_next), n_cheb))
                    T_next[:, 0] = 1.0
                    if n_cheb > 1:
                        T_next[:, 1] = x_next
                    for k in range(2, n_cheb):
                        T_next[:, k] = 2.0 * x_next * T_next[:, k-1] - T_next[:, k-2]
                    
                    # V_next[zp] at these points
                    v_next_zp = T_next @ V_next[zp]
                    cont_val += prob * v_next_zp
                
                total_val = u_current + beta * cont_val
                
                # Find best
                idx_best = np.argmax(total_val)
                best_c[i] = c_candidates[idx_best]
                best_val[i] = total_val[idx_best]
            
            consumption[t, z, :] = best_c
            
            # Fit Chebyshev coefficients for V_t[:, z]
            # V_t(m, z) = best_val at m_grid points
            V_next[z] = T_inv @ best_val
    
    # Terminal period: consume all
    consumption[horizon - 1, :, :] = m_grid[None, None, :]
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, m_grid + 1e-10)
    
    # Ensure c[5] (last period) equals grid within 1e-9
    consumption[horizon - 1, :, :] = m_grid[None, None, :]
    
    # Ensure finite values
    consumption = np.where(np.isfinite(consumption), consumption, 1e-12)
    
    return consumption
# EVOLVE-BLOCK-END
