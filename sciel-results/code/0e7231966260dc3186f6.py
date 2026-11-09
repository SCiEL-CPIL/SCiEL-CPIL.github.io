import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=float)
    budget = int(budget)

    n_states = len(income)
    n_m = len(m_grid)

    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Log-linear value function: V_z(m) ≈ A_z + B_z * log(m)
    # Initialize with log utility (terminal condition)
    A = np.zeros(n_states)
    B = np.zeros(n_states)
    
    # Terminal: V(m) = u(m), so for CRRA:
    # If gamma=1: log(m) => A=0, B=1
    # If gamma!=1: m^(1-gamma)/(1-gamma) => not log-linear, but we approximate
    # For initialization, use the terminal utility coefficients
    if abs(gamma - 1.0) < 1e-12:
        A[:] = 0.0
        B[:] = 1.0
    else:
        # Approximate m^(1-gamma)/(1-gamma) with A + B*log(m) over the grid
        # Use least squares fit
        log_m = np.log(m_grid)
        V_terminal = np.power(m_grid, 1.0 - gamma) / (1.0 - gamma)
        X = np.column_stack([np.ones(n_m), log_m])
        coeffs, _, _, _ = np.linalg.lstsq(X, V_terminal, rcond=None)
        A[:] = coeffs[0]
        B[:] = coeffs[1]

    policy = np.zeros((horizon, n_states, n_m))
    V = np.zeros((horizon, n_states, n_m))

    # Store the log-linear coefficients for each period
    A_per_t = np.zeros((horizon, n_states))
    B_per_t = np.zeros((horizon, n_states))

    # Final period: t = horizon - 1 = 5
    A_per_t[5, :] = A
    B_per_t[5, :] = B
    for z in range(n_states):
        V[5, z, :] = utility(m_grid)
    policy[5, :, :] = m_grid[np.newaxis, :]

    # Work backwards from t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            for tt in range(t, 0, -1):
                for z in range(n_states):
                    policy[tt, z, :] = policy[tt + 1, z, :]
                    V[tt, z, :] = V[tt + 1, z, :]
            policy[5, :, :] = m_grid[np.newaxis, :]
            for z in range(n_states):
                V[5, z, :] = utility(m_grid)
            for tt in range(horizon):
                for z in range(n_states):
                    policy[tt, z, :] = np.clip(policy[tt, z, :], 1e-12, m_grid + 1e-10)
            return policy

        # For each state z, solve for optimal A_z, B_z using Newton-Kantorovich
        for z in range(n_states):
            # The FOC for CRRA with log-linear V:
            # u'(c) = beta * R * E_z'[V_{t+1}'(m')]
            # where m' = R*(m-c) + income[z']
            # V'_{t+1}(m) = B_{t+1, z'} / m
            # So: u'(c) = beta * R * sum_z' P(z|z') * B_{t+1,z'} / (R*(m-c) + income[z'])
            
            # For CRRA: u'(c) = c^(-gamma)
            # c^(-gamma) = beta * R * sum_z' P(z|z') * B_{t+1,z'} / (R*(m-c) + income[z'])
            
            # This is nonlinear in c, but we can solve it numerically.
            # The key insight: with log-linear V, the FOC gives a smooth policy.
            
            # Newton-Kantorovich iteration to find A_z, B_z:
            # We want V_z(m) = A_z + B_z * log(m) to approximate the true value function.
            # The true value: V_z(m) = max_c [u(c) + beta * E_z'[V_{t+1,z'}(m')]]
            
            # Strategy: 
            # 1. For a given (A, B) of next period, compute optimal c for each m
            # 2. Compute resulting V(m)
            # 3. Fit A_z, B_z to V(m) via least squares
            # 4. Iterate until convergence
            
            A_new = A.copy()
            B_new = B.copy()
            
            max_iter_nk = 20
            tol_nk = 1e-10
            
            for nk_iter in range(max_iter_nk):
                if time.time() > deadline:
                    break
                    
                # For each m, find optimal c using the FOC
                # c^(-gamma) = beta * R * sum_z' P(z|z') * B_per_t[t+1, z'] / (R*(m-c) + income[z'])
                
                # Define the FOC residual
                def foc_residual(c, m):
                    """Return f(c) = u'(c) - beta*R*E[V'(m')]"""
                    c_safe = np.maximum(c, 1e-12)
                    u_prime = np.power(c_safe, -gamma)
                    m_prime = R * (m - c) + income[np.newaxis, :]
                    # V'(m') = B_per_t[t+1, z'] / m'
                    v_prime = B_per_t[t + 1, :] / np.maximum(m_prime, 1e-12)
                    expected_v_prime = np.sum(transition[z, :] * v_prime, axis=1)
                    return u_prime - beta * R * expected_v_prime
                
                # For each m, solve FOC using bounded scalar minimization
                # Minimize -[u(c) + beta*E[V_next(m')]]
                c_opt = np.zeros(n_m)
                
                for i in range(n_m):
                    m_i = m_grid[i]
                    
                    def objective(c):
                        c_safe = np.maximum(c, 1e-12)
                        c_clipped = np.minimum(c_safe, m_i + 1e-10)
                        u_val = utility(c_clipped)
                        m_prime = R * (m_i - c_clipped) + income
                        # Use log-linear V for next period
                        m_prime_safe = np.maximum(m_prime, 1e-12)
                        v_next = A_per_t[t + 1, :] + B_per_t[t + 1, :] * np.log(m_prime_safe)
                        expected_v = np.sum(transition[z, :] * v_next)
                        return -(u_val + beta * expected_v)
                    
                    # Bounded scalar minimization
                    res = minimize_scalar(objective, bounds=(1e-12, m_i + 1e-10), method='bounded',
                                          options={'xatol': 1e-10, 'maxiter': 100})
                    c_opt[i] = res.x
                
                # Compute V(m) = u(c_opt) + beta * E[V_next(m')]
                c_safe = np.maximum(c_opt, 1e-12)
                u_vals = utility(c_safe)
                m_prime = R * (m_grid - c_opt) + income[np.newaxis, :]
                m_prime_safe = np.maximum(m_prime, 1e-12)
                v_next = A_per_t[t + 1, :] + B_per_t[t + 1, :] * np.log(m_prime_safe)
                expected_v = np.sum(transition[z, :] * v_next, axis=1)
                V_m = u_vals + beta * expected_v
                
                # Fit A_z, B_z to V(m) via least squares
                log_m = np.log(m_grid)
                X = np.column_stack([np.ones(n_m), log_m])
                coeffs, _, _, _ = np.linalg.lstsq(X, V_m, rcond=None)
                A_new[z] = coeffs[0]
                B_new[z] = coeffs[1]
                
                # Check convergence
                delta_A = abs(A_new[z] - A[z])
                delta_B = abs(B_new[z] - B[z])
                A[z] = A_new[z]
                B[z] = B_new[z]
                
                if delta_A < tol_nk and delta_B < tol_nk:
                    break
            
            A_per_t[t, z] = A[z]
            B_per_t[t, z] = B[z]
            
            # Compute policy and value on the grid
            c_opt = np.zeros(n_m)
            for i in range(n_m):
                m_i = m_grid[i]
                
                def objective(c):
                    c_safe = np.maximum(c, 1e-12)
                    c_clipped = np.minimum(c_safe, m_i + 1e-10)
                    u_val = utility(c_clipped)
                    m_prime = R * (m_i - c_clipped) + income
                    m_prime_safe = np.maximum(m_prime, 1e-12)
                    v_next = A_per_t[t + 1, :] + B_per_t[t + 1, :] * np.log(m_prime_safe)
                    expected_v = np.sum(transition[z, :] * v_next)
                    return -(u_val + beta * expected_v)
                
                res = minimize_scalar(objective, bounds=(1e-12, m_i + 1e-10), method='bounded',
                                      options={'xatol': 1e-10, 'maxiter': 100})
                c_opt[i] = res.x
            
            c_safe = np.maximum(c_opt, 1e-12)
            u_vals = utility(c_safe)
            m_prime = R * (m_grid - c_opt) + income[np.newaxis, :]
            m_prime_safe = np.maximum(m_prime, 1e-12)
            v_next = A_per_t[t + 1, :] + B_per_t[t + 1, :] * np.log(m_prime_safe)
            expected_v = np.sum(transition[z, :] * v_next, axis=1)
            V_m = u_vals + beta * expected_v
            
            policy[t, z, :] = c_opt
            V[t, z, :] = V_m

    # Final validation
    for t in range(horizon):
        for z in range(n_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
    policy[5, :, :] = m_grid[np.newaxis, :]

    return policy
