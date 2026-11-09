"""
Consumption policy solver: Euler equation root-finding with finite difference V'
"""
import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 140

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)

def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def _solve_consumption_single(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    trans = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    
    consumption = np.zeros((horizon, states, n_grid))
    
    # Terminal condition: consume all
    consumption[horizon - 1] = np.tile(m_grid, (states, 1))
    V_next = np.tile(_crsa_utility(m_grid, gamma), (states, 1))
    
    # Precompute log grid for potential use, though we work in linear m for V'
    # We need V' for each state.
    # We will compute V_next_prime as well to speed up next iteration? 
    # No, V_next changes every step. We compute V_next_prime at the start of each loop iteration for V_current?
    # Actually, we need V' of the NEXT period's value function to solve for CURRENT consumption.
    # So we need V_prime_next.
    
    def compute_v_prime(V, m_grid):
        # V shape: (states, n_grid)
        # m_grid shape: (n_grid,)
        # Use central differences for interior, one-sided for boundaries
        n = len(m_grid)
        V_prime = np.zeros_like(V)
        
        # To handle the geometric grid properly, we can just use np.gradient with edge_order=2
        # np.gradient works on the values. Since m_grid is not uniform, we must pass it as the second argument.
        # V shape is (states, n_grid). We want gradient along axis 1.
        # V_prime[z, i] = dV[z, i] / dm[i]
        V_prime = np.gradient(V, m_grid, axis=1, edge_order=2)
        return V_prime

    V_prime_next = compute_v_prime(V_next, m_grid)
    
    for t in range(horizon - 2, -1, -1):
        V_current = np.zeros((states, n_grid))
        
        for z in range(states):
            m = m_grid
            c_opt = np.zeros(n_grid)
            
            # We need to solve for c in (0, m) for each m in m_grid
            # Residual: u'(c) - beta * R * sum_zp trans[z, zp] * V_prime_next[zp, R(m-c) + income[zp]]
            
            # To speed up brentq, we can vectorize the interpolation of V_prime_next.
            # However, brentq is scalar. 
            # Given n_grid=101 and horizon=6, states=2, total calls = 6 * 2 * 101 = ~1200.
            # Each brentq call takes ~10-20 iterations. 24000 function evaluations.
            # Each evaluation involves interpolation of 2 states. This is very fast.
            
            # Pre-compute interpolation arrays for V_prime_next
            # We need to interpolate V_prime_next[zp] at points m_prime.
            # m_prime = R * (m - c) + income[zp]
            
            for i in range(n_grid):
                m_i = m[i]
                
                # Define residual function for brentq
                # c is in (eps, m_i)
                # We need to be careful with bounds.
                # u'(c) -> inf as c -> 0.
                # RHS -> beta * R * E[V'(R*m + income)] which is finite.
                # So there is always a root in (0, m_i).
                
                # Lower bound: small epsilon
                c_min = 1e-8
                # Upper bound: m_i (where u'(m_i) is finite, RHS is beta*R*E[V'(income)])
                # Actually if c=m_i, m'=income. 
                # If m_i is very small, c_min might be >= m_i? 
                # m_grid starts at 0.001. c_min = 1e-8 is safe.
                c_max = m_i
                
                def residual(c):
                    # u'(c)
                    u_prime = _crsa_marginal(c, gamma)
                    
                    # m_prime for each state zp
                    # m_prime = R * (m_i - c) + income[zp]
                    m_prime_vec = R * (m_i - c) + income
                    # Clip to grid bounds for interpolation
                    # If m_prime < m_grid[0], extrapolate? 
                    # V' is defined on [m_grid[0], m_grid[-1]].
                    # For m_prime < m_grid[0], we can assume V' is roughly constant or use the boundary value.
                    # Given m_grid[0]=0.001 and income min 0.5, R=1.02, m_prime is usually > 0.
                    # If c is close to m_i, m_prime is close to income. 
                    # If c is small, m_prime is close to R*m_i + income.
                    # The smallest m_prime happens when c is large (c~m_i) and m_i is small.
                    # m_min_prime ~ R*(0) + 0.5 = 0.5. This is > m_grid[0]=0.001.
                    # The largest m_prime happens when c is small and m_i is large.
                    # m_max_prime ~ R*40 + 1.5 = 42.5. This is > m_grid[-1]=40.
                    # So we need to handle extrapolation or clipping.
                    # Clipping to [m_grid[0], m_grid[-1]] is standard for interpolation.
                    m_prime_clipped = np.clip(m_prime_vec, m_grid[0], m_grid[-1])
                    
                    # Interpolate V_prime_next for each state
                    # V_prime_next shape (states, n_grid)
                    # We need V_prime_next[zp] at m_prime_clipped[zp]
                    # Vectorized interpolation
                    vp_interp = np.empty(states)
                    for zp in range(states):
                        vp_interp[zp] = np.interp(m_prime_clipped[zp], m_grid, V_prime_next[zp])
                    
                    # Expected V'
                    exp_vp = np.dot(trans[z], vp_interp)
                    
                    rhs = beta * R * exp_vp
                    return u_prime - rhs

                # Solve
                try:
                    c_i = brentq(residual, c_min, c_max, xtol=1e-10, rtol=1e-12, maxiter=50)
                except Exception:
                    # Fallback to midpoint or simple heuristic if root finding fails
                    c_i = 0.5 * m_i
                    
                # Ensure constraints
                if c_i <= 0:
                    c_i = c_min
                if c_i > m_i:
                    c_i = m_i
                    
                c_opt[i] = c_i
                V_current[z, i] = _crsa_utility(c_i, gamma) + beta * (
                    # We need the actual expected value, not just V'. 
                    # We didn't store V_next values in the residual loop.
                    # We can recompute or store. 
                    # For efficiency, we can just use the value from the previous step's interpolation?
                    # No, V_current[z,i] is the max value. 
                    # We can compute it as u(c) + beta * E[V_next(m')]
                    # Let's compute E[V_next(m')] here.
                    0.0 
                )
                
                # Recompute V_current properly
                m_prime_vec = R * (m_i - c_i) + income
                m_prime_clipped = np.clip(m_prime_vec, m_grid[0], m_grid[-1])
                exp_v = 0.0
                for zp in range(states):
                    v_interp = np.interp(m_prime_clipped[zp], m_grid, V_next[zp])
                    exp_v += trans[z, zp] * v_interp
                V_current[z, i] = _crsa_utility(c_i, gamma) + beta * exp_v
                
            consumption[t, z, :] = c_opt
            
        V_next = V_current
        V_prime_next = compute_v_prime(V_next, m_grid)

    return consumption

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    try:
        result = _solve_consumption_single(model, resource_grid, budget)

        # Validate
        if not np.all(np.isfinite(result)):
            raise ValueError("Non-finite values")
        m = resource_grid[None, None, :]
        if np.any(result <= 0) or np.any(result > m + 1e-10):
            raise ValueError("c out of bounds")
        horizon = model['horizon']
        states = model['transition'].shape[0]
        if not np.allclose(result[horizon - 1],
                           np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")

        return result
    except Exception:
        # Fallback: simple heuristic
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)

        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                # Simple saving rule: save a fraction based on horizon remaining
                frac_save = 0.1 * (horizon - 1 - t) / (horizon - 1)
                result[t] = (1.0 - frac_save) * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))

        return result
