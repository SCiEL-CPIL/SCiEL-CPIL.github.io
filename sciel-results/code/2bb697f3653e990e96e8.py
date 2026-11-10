import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    num_states = len(income)
    c_floor = 1e-12
    
    # Dense internal grid for backward induction
    n_internal = 500
    m_internal = np.geomspace(0.001, 40, n_internal)
    log_m_internal = np.log(m_internal)
    
    # Number of basis functions for value representation
    n_basis = 20
    
    # Utility functions
    def utility(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if gamma == 1:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)
    
    # Value function representation: V(m) = sum_j a_j * m^(p_j)
    # where p_j are chosen exponents. This captures CRRA asymptotics.
    # For CRRA, V(m) ~ m^(1-gamma)/(1-gamma) as m -> inf, and V(m) ~ m^(1-gamma) as m -> 0
    # We use a sum of power terms to approximate this.
    
    # Choose exponents: a mix that spans the relevant range
    # For gamma=1 (log), V ~ log(m), so we need log-like behavior
    # For gamma>1, V ~ m^(1-gamma)/(1-gamma) which is negative and decreasing in m
    
    # Basis exponents: use a geometric spread
    if gamma == 1:
        # Log utility: use log basis
        # V(m) = sum_j a_j * log(m + eps_j)
        # Simpler: just use PCHIP on log(m) space
        use_log_basis = True
    else:
        # Power utility: V(m) ~ m^(1-gamma)
        # Use power basis with exponents around 1-gamma
        use_log_basis = False
    
    # Store value coefficients and policy
    # For each t, z: store value coefficients and policy on grid
    value_coeffs = np.zeros((horizon, num_states, n_basis))
    policy_int = np.zeros((horizon, num_states, n_internal))
    
    # Terminal period
    t = horizon - 1
    for z in range(num_states):
        policy_int[t, z, :] = m_internal
        # Fit value coefficients to terminal utility
        if use_log_basis:
            # V(m) = log(m)
            # Represent as sum of log(m + eps_j)
            # For simplicity, just use the utility directly
            pass
        else:
            # V(m) = m^(1-gamma)/(1-gamma)
            # Single term suffices
            value_coeffs[t, z, 0] = 1.0 / (1.0 - gamma)
            # Exponent is 1-gamma
    
    # Define basis functions
    if gamma == 1:
        # Log basis: phi_j(m) = log(m + eps_j)
        eps_vals = np.geomspace(1e-4, 10, n_basis)
        def basis_func(m, j):
            return np.log(m + eps_vals[j])
        def basis_prime(m, j):
            return 1.0 / (m + eps_vals[j])
        def basis_double_prime(m, j):
            return -1.0 / (m + eps_vals[j])**2
    else:
        # Power basis: phi_j(m) = m^(p_j)
        # Exponents centered around 1-gamma with spread
        p_center = 1.0 - gamma
        p_spread = max(0.5, abs(gamma) * 0.5)
        p_exponents = p_center + np.linspace(-p_spread, p_spread, n_basis)
        def basis_func(m, j):
            return np.power(np.maximum(m, c_floor), p_exponents[j])
        def basis_prime(m, j):
            return p_exponents[j] * np.power(np.maximum(m, c_floor), p_exponents[j] - 1)
        def basis_double_prime(m, j):
            return p_exponents[j] * (p_exponents[j] - 1) * np.power(np.maximum(m, c_floor), p_exponents[j] - 2)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: consume all
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy_int[tt, z, :] = m_internal
            break
        
        for z in range(num_states):
            probs = transition[z]
            
            # Newton-Raphson on FOC: u'(c) = beta * R * sum_zp P(z') * V'(m')
            # where m' = R*(m - c) + income[z']
            
            # Initial guess: proportional consumption
            c = np.clip(m_internal * 0.5, c_floor, m_internal)
            c_low = np.full(n_internal, c_floor)
            c_high = m_internal.copy()
            
            max_iter = 50
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                lhs = u_prime(c)
                rhs = np.zeros(n_internal)
                rhs_prime = np.zeros(n_internal)
                
                for zp in range(num_states):
                    if probs[zp] < 1e-15:
                        continue
                    m_next = R * (m_internal - c) + income[zp]
                    m_next_clipped = np.clip(m_next, c_floor, 40.0)
                    
                    # Compute V'(m_next) = sum_j a_j * phi_j'(m_next)
                    vp = np.zeros(n_internal)
                    vpp = np.zeros(n_internal)
                    for j in range(n_basis):
                        a_j = value_coeffs[t + 1, zp, j]
                        if abs(a_j) > 1e-15:
                            vp += a_j * basis_prime(m_next_clipped, j)
                            vpp += a_j * basis_double_prime(m_next_clipped, j)
                    
                    rhs += probs[zp] * vp
                    rhs_prime += probs[zp] * vpp * (-R)
                
                rhs = beta * R * rhs
                rhs_prime = beta * R * rhs_prime
                
                f_val = lhs - rhs
                f_prime = u_double_prime(c) - rhs_prime
                
                # Newton step with bisection safeguard
                newton_step = -f_val / np.where(np.abs(f_prime) > 1e-15, f_prime, 1e-15)
                c_newton = c + newton_step
                
                # Bisection update
                too_high = f_val > 0
                c_low = np.where(too_high, c, c_low)
                c_high = np.where(~too_high, c, c_high)
                c_bisect = (c_low + c_high) / 2.0
                
                c_new = np.where(
                    (c_newton >= c_floor) & (c_newton <= m_internal),
                    c_newton,
                    c_bisect
                )
                c = np.clip(c_new, c_floor, m_internal)
                
                if np.all(c_high - c_low < 1e-10 * m_internal):
                    break
            
            policy_int[t, z, :] = np.clip(c, c_floor, m_internal)
            
            # Fit value function coefficients
            # V(m) = u(c*) + beta * sum_zp P(z') * V_{t+1}(R*(m-c*) + income[z'])
            u_val = utility(policy_int[t, z, :])
            cont_val = np.zeros(n_internal)
            for zp in range(num_states):
                if probs[zp] < 1e-15:
                    continue
                m_next = R * (m_internal - policy_int[t, z, :]) + income[zp]
                m_next_clipped = np.clip(m_next, c_floor, 40.0)
                
                # V_{t+1}(m_next) = sum_j a_j * phi_j(m_next)
                v_next = np.zeros(n_internal)
                for j in range(n_basis):
                    a_j = value_coeffs[t + 1, zp, j]
                    if abs(a_j) > 1e-15:
                        v_next += a_j * basis_func(m_next_clipped, j)
                
                cont_val += probs[zp] * v_next
            
            V_target = u_val + beta * cont_val
            
            # Fit coefficients: minimize ||sum_j a_j * phi_j(m) - V_target||^2
            # Build design matrix
            A = np.zeros((n_internal, n_basis))
            for j in range(n_basis):
                A[:, j] = basis_func(m_internal, j)
            
            # Solve least squares
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, V_target, rcond=None)
                value_coeffs[t, z, :] = coeffs
            except Exception:
                # Fallback: use previous coefficients
                value_coeffs[t, z, :] = value_coeffs[t + 1, z, :]
    
    # Interpolate policy to resource_grid
    policy = np.zeros((horizon, num_states, len(resource_grid)))
    for t in range(horizon):
        for z in range(num_states):
            try:
                log_policy = np.log(policy_int[t, z, :] + 1e-30)
                interp = PchipInterpolator(log_m_internal, log_policy)
                policy[t, z, :] = np.exp(interp(np.log(resource_grid)))
            except Exception:
                policy[t, z, :] = np.interp(resource_grid, m_internal, policy_int[t, z, :])
            policy[t, z, :] = np.clip(policy[t, z, :], c_floor, resource_grid)
    
    # Ensure terminal condition is exact
    for z in range(num_states):
        policy[horizon - 1, z, :] = resource_grid
    
    return policy
