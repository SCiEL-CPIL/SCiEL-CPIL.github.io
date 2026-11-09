"""Economics: consumption-saving policy accuracy solver."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    n_states = len(income)
    n_grid = len(resource_grid)

    # CRRA utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Utility for scalar
    def u_scalar(c):
        c = max(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    # Final period: consume everything
    # consumption[t, z, i] for t in 0..horizon-1
    # At t=horizon (last), c = m (consume all)
    # We need consumption[t, z, i] for t=0..horizon-1, shape (horizon, n_states, n_grid)
    # But the task says shape (6, 2, 101) and c[5] equal to grid.
    # So t=5 is the last period where c[5] = m (the grid value).
    # Actually, horizon=6 means periods 0,1,2,3,4,5. At t=5 (last), consume all.
    # So we compute value function backward from t=5.

    # Value function V[t, z, m] = max_c { u(c) + beta * E[V[t+1, z', m']] }
    # m' = R*(m - c) + income[z']
    # z' drawn from transition[z]

    # We'll use a fine grid for m to get accurate interpolation.
    # The scoring interpolates onto np.geomspace(0.001, 40, 251).
    # Let's use a grid that's at least as fine as 251 points, maybe more.
    # But we also need to be fast. Let's use 251 or 301 points for the value function grid.
    # Actually, the scoring evaluates at 251 points and the best is over 301 consumption fractions.
    # Let's use a grid of m values for the value function.

    # Strategy: 
    # 1. Define a fine grid for m: m_fine = np.geomspace(0.001, 40, 251)
    # 2. Compute value function V[t, z, :] on this grid via backward induction.
    # 3. For each t, z, m on the fine grid, find optimal c.
    # 4. Return consumption on the original resource_grid (101 points) by interpolation.

    # But wait, the return must be on resource_grid (101 points). The scoring interpolates
    # our 101-point policy onto 251 points. So we should make our 101-point policy as accurate
    # as possible.

    # Let's use a finer internal grid for computation, then interpolate the optimal consumption
    # back to the 101-point grid.

    # Actually, a better approach: compute the optimal consumption function c*(m, z, t) 
    # on a fine grid, then interpolate to the 101-point resource_grid.

    # Let's use m_compute = np.geomspace(0.001, 40, 501) for internal computation.
    # This gives better accuracy when we interpolate to 101 points.

    # Time check: 6 periods * 2 states * 501 m-points * (search over c) 
    # If we use a coarse c-search first, then refine, it should be fast.

    # Alternative: use the fact that for CRRA, the optimal consumption has a known form
    # in the infinite horizon case, but here it's finite horizon with Markov income.

    # Let's do backward induction with a fine m-grid and a c-search.

    m_compute = np.geomspace(0.001, 40, 501)
    n_m = len(m_compute)

    # V[t, z, i] for t=0..horizon-1, z=0..n_states-1, i=0..n_m-1
    # V[horizon-1, z, i] = u(m_compute[i])  (consume all at last period)

    # Actually, at the last period t=horizon-1, the agent consumes all: c = m.
    # So V[horizon-1, z, i] = u(m_compute[i])

    # For t < horizon-1:
    # V[t, z, i] = max_c { u(c) + beta * sum_{z'} transition[z, z'] * V[t+1, z', m'] }
    # where m' = R*(m - c) + income[z']
    # and c in [0, m] (with c >= 1e-12 for utility)

    # We'll store the optimal consumption as well.
    # c_opt[t, z, i] = optimal consumption at state (t, z, m_compute[i])

    V = np.zeros((horizon, n_states, n_m))
    c_opt = np.zeros((horizon, n_states, n_m))

    # Last period
    V[horizon-1, :, :] = u(m_compute)
    c_opt[horizon-1, :, :] = m_compute  # consume all

    # Backward induction
    for t in range(horizon-2, -1, -1):
        if time.time() > deadline - 10:
            # Not enough time, break and use what we have
            break
        for z in range(n_states):
            # For each m in m_compute, find optimal c
            # c in [1e-12, m]
            # Let's use a log-spaced or linear-spaced grid for c as a fraction of m.
            # c = m * frac, frac in [1e-4, 1]
            # Use 101 fractions for speed, then refine.
            
            # First pass: coarse grid
            fracs = np.linspace(0.001, 1.0, 101)
            c_grid = m_compute[None, :] * fracs[:, None]  # (101, n_m)
            # But c can't exceed m, and c >= 1e-12
            c_grid = np.maximum(c_grid, 1e-12)
            
            # For each c, compute m' for each z'
            # m' = R*(m - c) + income[z']
            # shape: (101, n_m, n_states) for m'
            m_minus_c = m_compute[None, None, :] - c_grid[:, :, None]  # (101, n_m, 1)
            m_prime = R * m_minus_c + income[None, None, :]  # (101, n_m, n_states)
            
            # Clip m_prime to be within our grid range for interpolation
            m_prime_clipped = np.clip(m_prime, m_compute[0], m_compute[-1])
            
            # Continuation value: V[t+1, z', m']
            # For each z', interpolate V[t+1, z', :] at m_prime
            # V_next[z', :] has shape (n_m,)
            # We need to interpolate for each (c_idx, m_idx, z')
            
            # Vectorized interpolation
            # V_interp[c_idx, m_idx, z'] = interp(m_prime[c_idx, m_idx, z'], m_compute, V[t+1, z', :])
            
            V_next = V[t+1, :, :]  # (n_states, n_m)
            
            # For each z', we have V_next[z', :] 
            # m_prime has shape (101, n_m, n_states)
            # We want V_interp with shape (101, n_m, n_states)
            
            # Use np.interp along the m dimension
            V_interp = np.zeros_like(m_prime)
            for zp in range(n_states):
                V_interp[:, :, zp] = np.interp(m_prime[:, :, zp].ravel(), m_compute, V_next[zp, :]).reshape(m_prime[:, :, zp].shape)
            
            # Expected continuation value: sum over z' of transition[z, z'] * V_interp[:, :, z']
            # transition[z, :] has shape (n_states,)
            exp_cont = np.sum(transition[z, None, None, :] * V_interp[:, :, None, :], axis=-1)
            # Wait, transition[z] is shape (n_states,). 
            # exp_cont[c_idx, m_idx] = sum_{z'} transition[z, z'] * V_interp[c_idx, m_idx, z']
            exp_cont = np.tensordot(transition[z], V_interp, axes=([0], [2]))
            # shape: (101, n_m)
            
            # Total value: u(c) + beta * exp_cont
            u_vals = u(c_grid)  # (101, n_m)
            total_val = u_vals + beta * exp_cont  # (101, n_m)
            
            # Find argmax over c for each m
            best_idx = np.argmax(total_val, axis=0)  # (n_m,)
            c_opt[t, z, :] = c_grid[best_idx, np.arange(n_m)]
            V[t, z, :] = total_val[best_idx, np.arange(n_m)]
        
        # Time check
        if time.time() > deadline - 5:
            break
    
    # Now we have c_opt[t, z, :] on m_compute (501 points)
    # We need to return consumption on resource_grid (101 points)
    # consumption[t, z, i] = c_opt[t, z, :] interpolated to resource_grid[i]
    
    consumption = np.zeros((horizon, n_states, n_grid))
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.interp(resource_grid, m_compute, c_opt[t, z, :])
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z, :] = np.clip(consumption[t, z, :], 1e-12, resource_grid + 1e-10)
    
    # c[5] must equal the grid within 1e-9
    consumption[5, :, :] = resource_grid[None, :]
    
    # Ensure finite
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40.0, neginf=1e-12)
    
    return consumption
# EVOLVE-BLOCK-END
