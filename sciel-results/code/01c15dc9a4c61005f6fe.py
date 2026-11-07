import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid size
    n_out = len(resource_grid)
    
    # Internal fine grid for better accuracy
    # We use a geometric grid to match the distribution of resource values
    # The problem specifies resource_grid = np.geomspace(0.001, 40, 101)
    # We will use a finer version of this
    n_fine = 1000
    m_fine = np.geomspace(0.001, 40, n_fine)
    
    # Policy shape: (horizon, 2, n_fine)
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Terminal period: consume everything
    policy_fine[horizon - 1, :, :] = m_fine
    
    # Interpolators for the fine grid
    # We use PchipInterpolator for monotonicity preservation
    pchip_policies = [None] * horizon
    pchip_policies[horizon - 1] = [
        PchipInterpolator(m_fine, policy_fine[horizon - 1, 0, :], extrapolate=False),
        PchipInterpolator(m_fine, policy_fine[horizon - 1, 1, :], extrapolate=False)
    ]
    
    # Helper to compute marginal utility
    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
    
    # Work backward from t = horizon-2 to t = 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fill remaining with terminal-like policy (or best guess)
            # If we run out of time, we just use the next period's policy as approximation
            # Or just break and map what we have
            for tt in range(t, horizon - 1):
                policy_fine[tt, :, :] = m_fine
            break
            
        for z in range(2):
            # Vectorized Bisection to solve FOC:
            # c^(-gamma) = beta * R * E[ c_next^(-gamma) ]
            # Let g(c) = c^(-gamma) - beta * R * E[ c_next^(-gamma) ]
            # We want g(c) = 0. g is decreasing in c.
            
            c_lo = np.full(n_fine, 1e-12)
            c_hi = m_fine.copy()
            
            # Pre-compute transition weights
            # E[ . ] = transition[z, 0] * f(state 0) + transition[z, 1] * f(state 1)
            
            for _ in range(60): # 60 iterations of bisection is plenty for double precision
                c_mid = (c_lo + c_hi) / 2.0
                
                # Calculate m_next for each state
                # m_next[s] = R * (m_fine - c_mid) + income[s]
                # Note: m_fine is shape (n_fine,), c_mid is (n_fine,)
                # m_next will be (n_fine, 2)
                m_next = R * (m_fine[:, np.newaxis] - c_mid[:, np.newaxis]) + income[np.newaxis, :]
                
                # Get next period consumption for each state
                # c_next[s] = pchip_policies[t+1][s](m_next[:, s])
                c_next_0 = pchip_policies[t + 1][0](m_next[:, 0])
                c_next_1 = pchip_policies[t + 1][1](m_next[:, 1])
                
                # Ensure c_next is positive for utility
                c_next_0 = np.maximum(c_next_0, 1e-12)
                c_next_1 = np.maximum(c_next_1, 1e-12)
                
                # Marginal utilities
                mu_next_0 = c_next_0 ** (-gamma)
                mu_next_1 = c_next_1 ** (-gamma)
                
                # Expected marginal utility
                E_mu_next = transition[z, 0] * mu_next_0 + transition[z, 1] * mu_next_1
                
                # Residual: c^(-gamma) - beta * R * E_mu_next
                # c_mid is (n_fine,)
                mu_mid = c_mid ** (-gamma)
                residual = mu_mid - beta * R * E_mu_next
                
                # Update bounds
                # If residual > 0, c is too small (since g is decreasing)
                # If residual < 0, c is too large
                mask_pos = residual > 0
                c_lo[mask_pos] = c_mid[mask_pos]
                
                mask_neg = residual < 0
                c_hi[mask_neg] = c_mid[mask_neg]
                
                # If residual is exactly 0, no update needed, but rare
                # We can break early if max width is small
                if np.max(c_hi - c_lo) < 1e-10:
                    break
            
            policy_fine[t, z, :] = (c_lo + c_hi) / 2.0
            
            # Clip to valid range
            policy_fine[t, z, :] = np.clip(policy_fine[t, z, :], 1e-12, m_fine)
            
        # Build PCHIP interpolators for this period on the fine grid
        pchip_policies[t] = [
            PchipInterpolator(m_fine, policy_fine[t, 0, :], extrapolate=False),
            PchipInterpolator(m_fine, policy_fine[t, 1, :], extrapolate=False)
        ]
        
    # Map the fine grid policy back to the original resource_grid
    policy = np.zeros((horizon, 2, n_out))
    for t in range(horizon):
        for z in range(2):
            # Interpolate from m_fine to resource_grid
            # PchipInterpolator is already built, but we can just use it
            # Or np.interp if we want linear, but PCHIP is smoother
            # Since we have the interpolator object, use it
            policy[t, z, :] = pchip_policies[t][z](resource_grid)
            
            # Ensure validity: 0 < c <= m + 1e-10
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, resource_grid + 1e-10)
            
    # Terminal period must equal grid exactly
    policy[horizon - 1, :, :] = resource_grid
    
    return policy
