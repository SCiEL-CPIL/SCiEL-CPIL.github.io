"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def construct():
    """
    Minimize logarithmic energy of N points on the unit sphere using
    Simulated Annealing.
    """
    
    def compute_energy_full(x):
        """Compute the full logarithmic energy for a configuration x (N, 3)."""
        # x is assumed to be on the sphere
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Energy = sum_{i<j} log(1/dist) = -0.5 * sum_{i!=j} log(dist)
        # Using log(dist_sq) = 2*log(dist), so sum log(dist) = 0.5 * sum log(dist_sq)
        # We need sum_{i<j} log(1/dist) = - sum_{i<j} log(dist)
        # sum_{i<j} log(dist) = 0.5 * sum_{i!=j} log(dist)
        # sum_{i!=j} log(dist) = 0.5 * sum_{i!=j} log(dist_sq)
        # So sum_{i<j} log(dist) = 0.25 * sum_{i!=j} log(dist_sq)
        # Energy = -0.25 * sum_{i!=j} log(dist_sq)
        
        # Note: dist_sq on diagonal is 1, log(1)=0, so we can just sum all off-diagonal
        log_dist_sq = np.log(dist_sq)
        # Sum all off-diagonal elements
        total_log_dist_sq = np.sum(log_dist_sq)
        energy = -0.25 * total_log_dist_sq
        return energy

    def compute_energy_change(x, i, x_new_i):
        """
        Compute the change in energy if point i is moved to x_new_i.
        x: current configuration (N, 3)
        i: index of the point to move
        x_new_i: new position of point i (3,)
        """
        # Energy contribution of point i to others in current state
        # E_i_old = sum_{j!=i} log(1 / |x_i - x_j|)
        # E_i_new = sum_{j!=i} log(1 / |x_new_i - x_j|)
        
        # Vectorized calculation
        # Distances from x_i to all x_j
        diff_old = x[i] - x
        dist_sq_old = np.sum(diff_old * diff_old, axis=1)
        dist_sq_old[i] = 1.0 # Ignore self
        
        diff_new = x_new_i - x
        dist_sq_new = np.sum(diff_new * diff_new, axis=1)
        dist_sq_new[i] = 1.0 # Ignore self
        
        # log(1/d) = -log(d) = -0.5*log(d^2)
        # Delta E = sum_{j!=i} (-0.5*log(dist_sq_new[j]) - (-0.5*log(dist_sq_old[j])))
        # Delta E = -0.5 * sum_{j!=i} (log(dist_sq_new[j]) - log(dist_sq_old[j]))
        
        # Avoid log(0) if points coincide (should not happen with normalization and random moves, but safe)
        # dist_sq is always > 0 for distinct points on sphere
        delta_e = -0.5 * (np.sum(np.log(dist_sq_new)) - np.sum(np.log(dist_sq_old)))
        
        return delta_e

    def reproject(x_vec):
        """Normalize a vector to the unit sphere."""
        norm = np.linalg.norm(x_vec)
        if norm < 1e-15:
             return np.array([1.0, 0.0, 0.0])
        return x_vec / norm

    # Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = compute_energy_full(x)
    best_x = x.copy()
    best_energy = current_energy
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    alpha = 0.9999 # Cooling rate
    min_T = 1e-6
    
    start_time = time.time()
    time_budget = 110
    
    rng = np.random.default_rng(42)
    
    # We will run multiple restarts if time permits, or just one long run
    # Given the complexity, a single long run with good cooling is often best.
    # Let's try a few restarts with shorter runs to explore different basins.
    
    num_restarts = 5
    time_per_restart = time_budget / num_restarts
    
    for restart in range(num_restarts):
        # If this is not the first restart, perturb the best solution found so far
        if restart > 0:
            # Create a perturbed version of the best solution
            x = best_x.copy()
            # Add some random noise
            noise = rng.normal(0, 0.1, x.shape)
            x = x + noise
            # Renormalize
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / norms
            current_energy = compute_energy_full(x)
            
            # Reset temperature for this restart, maybe start higher or lower?
            # Starting high allows escaping the previous basin.
            T = 1.0 
        else:
            T = 1.0
            current_energy = compute_energy_full(x)

        while T > min_T and (time.time() - start_time) < time_budget - 5:
            # Pick a random point to move
            idx = rng.integers(0, N)
            
            # Generate a random move
            # Option 1: Small random step in Euclidean space
            # Option 2: Random rotation around a random axis
            # Let's use a random step: x_new = normalize(x_old + step * random_dir)
            # The magnitude of the step should be related to T?
            # Or keep step size constant and let T control acceptance?
            # Usually, step size is fixed or slowly decreasing.
            
            step_size = 0.1
            random_dir = rng.normal(size=3)
            random_dir = random_dir / np.linalg.norm(random_dir)
            
            x_new_candidate = x[idx] + step_size * random_dir
            x_new_candidate = reproject(x_new_candidate)
            
            # Calculate energy change
            delta_e = compute_energy_change(x, idx, x_new_candidate)
            
            # Acceptance criterion
            if delta_e < 0:
                # Improvement
                x[idx] = x_new_candidate
                current_energy += delta_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
            else:
                # Accept with probability exp(-delta_e / T)
                if rng.random() < math.exp(-delta_e / T):
                    x[idx] = x_new_candidate
                    current_energy += delta_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_x = x.copy()
            
            # Cool down
            T *= alpha
            
            # Optional: Reheat if stuck? 
            # For now, simple cooling.
            
    # Final refinement: Gradient Descent from the best SA result
    # This helps polish the solution to high precision.
    
    def compute_gradient(x):
        """
        Compute the Riemannian gradient of the energy on the sphere.
        """
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        inv_dist_sq = 1.0 / dist_sq
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        grad_eucl = np.einsum('ijk,ij->ik', diff, inv_dist_sq)
        
        # Project to tangent space
        grad_dot_x = np.sum(grad_eucl * x, axis=1, keepdims=True)
        grad_riemann = grad_eucl - grad_dot_x * x
        
        return grad_riemann

    def retraction(x, grad, step_size):
        x_new = x + step_size * grad
        norms = np.linalg.norm(x_new, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        return x_new / norms

    x_ref = best_x.copy()
    e_ref = best_energy
    step_size = 0.01
    min_step = 1e-10
    
    for it in range(10000):
        if time.time() - start_time > time_budget:
            break
        
        grad = compute_gradient(x_ref)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < 1e-10:
            break
        
        x_cand = retraction(x_ref, grad, step_size)
        e_cand = compute_energy_full(x_cand)
        
        if e_cand < e_ref:
            x_ref = x_cand
            e_ref = e_cand
            if e_ref < best_energy:
                best_energy = e_ref
                best_x = x_ref.copy()
            # Increase step if good
            step_size = min(step_size * 1.2, 0.1)
        else:
            step_size *= 0.5
            if step_size < min_step:
                break
                
    # Ensure output validity
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / np.maximum(norms, 1e-15)
    
    # Check for close points
    diff = best_x[:, np.newaxis, :] - best_x[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        rng_fix = np.random.default_rng(12345)
        best_x = best_x + rng_fix.normal(0, 1e-6, best_x.shape)
        norms = np.linalg.norm(best_x, axis=1, keepdims=True)
        best_x = best_x / np.maximum(norms, 1e-15)
        
    return best_x
# EVOLVE-BLOCK-END
