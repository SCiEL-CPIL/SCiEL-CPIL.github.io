import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precompute cost vector
    # cost_vec[i] is the number of cut edges for the bitstring i
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    
    # Efficiently compute cost vector
    indices = np.arange(num_states)
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1

    # 2. Fast simulation using tensor operations
    # State is a tensor of shape (2, 2, ..., 2) with n dimensions
    # We will use a flattened array for cost application and reshape for RX.
    
    def simulate(g1, b1, g2, b2):
        # Initial state: |+>^n -> all entries 1/sqrt(2^n)
        # As a tensor:
        state = np.ones([2]*n, dtype=np.complex128) / np.sqrt(num_states)
        
        # Helper to apply global phase (cost)
        # Cost is diagonal in computational basis.
        # phase_vec = exp(-i * gamma * cost_vec)
        # state_flat = state.flatten()
        # state_flat *= phase_vec
        # state = state_flat.reshape([2]*n)
        
        # Helper to apply RX(beta) to all qubits
        # RX = [[c, -i s], [-i s, c]]
        # We can do this efficiently by iterating over axes.
        
        def apply_rx(state, beta):
            c = np.cos(beta / 2.0)
            s = np.sin(beta / 2.0)
            # For each axis q:
            #   new_state[..., 0, ...] = c * old_state[..., 0, ...] - 1j*s * old_state[..., 1, ...]
            #   new_state[..., 1, ...] = -1j*s * old_state[..., 0, ...] + c * old_state[..., 1, ...]
            # Since the operation is identical on all axes, we can loop.
            
            # Optimization: Pre-compute slices? Or just use numpy indexing.
            # Using np.take or slicing might be faster.
            
            for q in range(n):
                # Move axis q to front
                state = np.moveaxis(state, q, 0)
                # Reshape to (2, -1)
                orig_shape = state.shape
                flat = state.reshape(2, -1)
                
                v0 = flat[0, :]
                v1 = flat[1, :]
                
                new_v0 = c * v0 - 1j * s * v1
                new_v1 = -1j * s * v0 + c * v1
                
                flat[0, :] = new_v0
                flat[1, :] = new_v1
                
                state = flat.reshape(orig_shape)
                state = np.moveaxis(state, 0, q)
                
            return state

        def apply_cost(state, gamma):
            phase = np.exp(-1j * gamma * cost_vec)
            flat = state.flatten()
            flat *= phase
            state = flat.reshape([2]*n)
            return state

        # Layer 1
        state = apply_cost(state, g1)
        state = apply_rx(state, b1)
        # Layer 2
        state = apply_cost(state, g2)
        state = apply_rx(state, b2)
        
        # Calculate expected cut
        probs = np.abs(state)**2
        exp_cut = np.dot(probs.flatten(), cost_vec)
        
        return exp_cut

    # 3. Objective function for minimization (maximize cut)
    def objective(params):
        if time.time() > deadline:
            return 1e10
        g1, b1, g2, b2 = params
        # Wrap parameters to [0, pi] to help optimizer? 
        # Actually, the landscape is periodic. 
        # Let's just compute.
        val = simulate(g1, b1, g2, b2)
        return -val

    # 4. Optimization Strategy
    # Strategy: Grid search over a coarse grid, then refine the top few points.
    # The parameter space is [0, pi] x [0, pi] x [0, pi] x [0, pi].
    # Grid size: 10x10x10x10 = 10,000 points.
    # Each simulation takes ~1ms? 10,000 * 1ms = 10s. Feasible.
    # Let's try a slightly finer grid or adaptive.
    # 12 points per dimension -> 20,736 points. ~20s.
    
    # Let's start with a moderate grid.
    # We can also exploit symmetry: gamma in [0, pi], beta in [0, pi].
    
    grid_size = 12
    grid_points = np.linspace(0, np.pi, grid_size)
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Grid search
    # We can vectorize the grid search? No, simulation is complex.
    # But we can iterate.
    
    # To speed up, we can pre-compute the cost phases for each grid point?
    # No, gamma is in the exponent.
    
    # Let's just loop.
    # 12^4 = 20736 iterations.
    # If each takes 0.5ms, that's 10s.
    
    # We will collect the top K points from the grid to refine.
    # Store (value, params)
    
    top_results = []
    
    # Use a heap or just a list and sort at the end?
    # List and sort is fine for 20k items.
    
    for g1 in grid_points:
        if time.time() > deadline - 20: # Reserve time for refinement
            break
        for b1 in grid_points:
            if time.time() > deadline - 20:
                break
            for g2 in grid_points:
                if time.time() > deadline - 20:
                    break
                for b2 in grid_points:
                    if time.time() > deadline - 20:
                        break
                    val = simulate(g1, b1, g2, b2)
                    top_results.append((val, (g1, b1, g2, b2)))
    
    # Sort by value (descending)
    top_results.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 points for refinement
    num_refine = 5
    if top_results:
        # Check if best grid point is better than current best
        if top_results[0][0] > best_val:
            best_val = top_results[0][0]
            best_params = np.array(top_results[0][1])
            
    # Refine top candidates
    for i in range(min(num_refine, len(top_results))):
        if time.time() > deadline - 10:
            break
        start_params = np.array(top_results[i][1])
        
        # Use Nelder-Mead for refinement
        # Set initial simplex size small
        try:
            res = minimize(objective, start_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_val: # Note: objective is -cut, so lower is better
                # Wait, best_val is stored as positive cut.
                # res.fun is negative cut.
                # So we compare -res.fun > best_val
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_params = res.x
        except Exception:
            pass

    # Return in required shape (p, 2)
    # p=2, so we have 2 gammas and 2 betas
    # best_params is [g1, b1, g2, b2]
    return best_params.reshape(p, 2)
