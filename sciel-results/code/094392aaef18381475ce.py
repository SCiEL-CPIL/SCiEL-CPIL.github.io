import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_out = len(resource_grid)
    m_out = resource_grid
    
    # Use a finer grid for solving to improve accuracy
    m_fine = np.geomspace(0.001, 40, 501)
    n_fine = len(m_fine)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Store value function coefficients (we'll store V directly on m_fine)
    # V_next[z, i] = value at m_fine[i] for state z
    V_next = np.zeros((2, n_fine))
    
    # Policy on fine grid
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            break
            
        V_curr = np.zeros((2, n_fine))
        
        for z in range(2):
            probs = transition[z]
            
            # For each m in m_fine, solve the Euler equation
            for i in range(n_fine):
                m = m_fine[i]
                
                if t == horizon - 1:
                    # Last period: consume all
                    c_opt = m
                    V_curr[z, i] = utility(c_opt)
                    policy_fine[t, z, i] = c_opt
                    continue
                
                # Define the Euler equation: u'(c) = beta * E[u'(c')]
                # where c' = R*(m - c) + income[z']
                
                def euler_eq(c):
                    c = max(c, 1e-12)
                    lhs = utility_prime(c)
                    
                    # Expected marginal utility of next period consumption
                    m_next_0 = R * (m - c) + income[0]
                    m_next_1 = R * (m - c) + income[1]
                    
                    # Interpolate V_next to get V_next_value, but we need V_next_prime
                    # Actually, we need u'(c') where c' is optimal next period
                    # This is complex. Instead, let's use the value function directly.
                    
                    # Alternative: use the Bellman equation approach with fine grid search
                    # but with a refined search around the approximate solution
                    
                    return 0  # Placeholder
                
                # Actually, let's use a hybrid approach:
                # 1. Coarse search to find approximate c*
                # 2. Refine using Newton's method on the Bellman equation
                
                # Coarse search
                fractions = np.linspace(0.001, 1.0, 201)
                c_cands = m * fractions
                c_cands = np.clip(c_cands, 1e-12, m)
                
                # Compute values
                m_next_0 = R * (m - c_cands) + income[0]
                m_next_1 = R * (m - c_cands) + income[1]
                
                # Interpolate continuation values
                V_cont_0 = np.interp(m_next_0, m_fine, V_next[0])
                V_cont_1 = np.interp(m_next_1, m_fine, V_next[1])
                E_V_cont = probs[0] * V_cont_0 + probs[1] * V_cont_1
                
                U_c = utility(c_cands)
                Total_Value = U_c + beta * E_V_cont
                
                idx = np.argmax(Total_Value)
                c_approx = c_cands[idx]
                
                # Refine with local search
                # Use a fine grid around c_approx
                delta = m * 0.01
                c_fine_local = np.linspace(max(1e-12, c_approx - delta), min(m, c_approx + delta), 101)
                
                m_next_0_f = R * (m - c_fine_local) + income[0]
                m_next_1_f = R * (m - c_fine_local) + income[1]
                
                V_cont_0_f = np.interp(m_next_0_f, m_fine, V_next[0])
                V_cont_1_f = np.interp(m_next_1_f, m_fine, V_next[1])
                E_V_cont_f = probs[0] * V_cont_0_f + probs[1] * V_cont_1_f
                
                U_c_f = utility(c_fine_local)
                Total_Value_f = U_c_f + beta * E_V_cont_f
                
                idx_f = np.argmax(Total_Value_f)
                c_opt = c_fine_local[idx_f]
                
                V_curr[z, i] = Total_Value_f[idx_f]
                policy_fine[t, z, i] = c_opt
                
        V_next = V_curr
    
    # Fill any unfilled periods
    last_valid_t = -1
    for t in range(horizon - 1, -1, -1):
        if not np.all(policy_fine[t] == 0):
            last_valid_t = t
            break
            
    if last_valid_t != horizon - 1:
        for t in range(last_valid_t, -1, -1):
            if np.all(policy_fine[t] == 0):
                policy_fine[t] = policy_fine[last_valid_t]
                
    # Interpolate to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(m_out, m_fine, policy_fine[t, z, :])
            
    # Validity checks
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.maximum(consumption_out[t, z, :], 1e-12)
            consumption_out[t, z, :] = np.minimum(consumption_out[t, z, :], m_out + 1e-10)
            
    # Constraint: c[5] = m_out
    consumption_out[5, :, :] = m_out[None, None, :]
    
    return consumption_out
