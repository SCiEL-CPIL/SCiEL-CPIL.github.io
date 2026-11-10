import numpy as np
import time
import random

N = 50

def compute_radius(centers):
    """
    Compute the radius of the largest equal circles that can be placed 
    at the given centers in the square [-0.5, 0.5]^2.
    """
    # centers shape (N, 2)
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Check boundaries
    # If any center is outside, return 0 or negative? 
    # The task says "inside the square". 
    # min_i min_k (1/2 - |c_ik|)
    # If c_ik > 0.5, this is negative.
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    if min_boundary_dist < 0:
        # Strictly outside
        return min_boundary_dist # Or 0? The score formula gives negative if outside? 
        # "Score: ... min_i min_k (1/2 - |c_ik|)"
        # If c is at 0.6, 0.5 - 0.6 = -0.1.
        # But usually "inside" implies validity. 
        # Let's assume the evaluator handles validity, but we want to maximize.
        # If we return negative, it's worse than 0.
        
    # Pairwise distances
    # Use broadcasting for speed, N=50 is small
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask diagonal
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    return r

def _feasibility_energy(centers, r):
    """
    Compute the sum of squared violations for a fixed radius r.
    Violations:
    1. Pairwise distance < 2r
    2. Distance to boundary < r
    """
    energy = 0.0
    
    # 1. Boundary violations
    # d_left = x + 0.5, need >= r -> x + 0.5 - r >= 0
    # d_right = 0.5 - x, need >= r -> 0.5 - x - r >= 0
    # etc.
    
    # Vectorized boundary check
    # x violations
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Left: x - (-0.5 + r) < 0
    viol_l = np.clip((-0.5 + r) - x, 0, None)
    # Right: (0.5 - r) - x < 0
    viol_r = np.clip(x - (0.5 - r), 0, None)
    # Bottom: y - (-0.5 + r) < 0
    viol_b = np.clip((-0.5 + r) - y, 0, None)
    # Top: (0.5 - r) - y < 0
    viol_t = np.clip(y - (0.5 - r), 0, None)
    
    energy += np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2)
    
    # 2. Pairwise violations
    # dist_ij < 2r
    # We can compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Only upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    
    # Violation amount: (2r - d_ij) if d_ij < 2r else 0
    viol_pairs = np.clip(2.0 * r - d_ij, 0, None)
    energy += np.sum(viol_pairs**2)
    
    return energy

def _simulated_annealing_feasibility(centers, r, start_time, deadline, initial_temp=0.01, cooling=0.999, max_iters=5000):
    """
    Try to find a configuration with energy 0 for radius r.
    Returns the best centers found.
    """
    best_centers = centers.copy()
    best_energy = _feasibility_energy(best_centers, r)
    
    current_centers = centers.copy()
    current_energy = best_energy
    
    temp = initial_temp
    
    rng = np.random.default_rng()
    
    for i in range(max_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Propose a move: perturb one random center
        idx = rng.integers(0, N)
        old_pos = current_centers[idx].copy()
        
        # Random step
        step_size = 0.02 * temp # Scale step with temperature? Or fixed?
        # Fixed step is often better for packing. Let's use a fixed step that decays slightly.
        # Actually, let's use a step size that depends on the current "tightness".
        # If energy is high, we are far from feasible, take bigger steps?
        # Standard SA: fixed step.
        step = 0.05
        new_pos = old_pos + rng.normal(0, step, size=2)
        
        # Keep inside square [-0.5, 0.5] to avoid huge boundary violations
        new_pos[0] = np.clip(new_pos[0], -0.49, 0.49)
        new_pos[1] = np.clip(new_pos[1], -0.49, 0.49)
        
        current_centers[idx] = new_pos
        
        new_energy = _feasibility_energy(current_centers, r)
        
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_centers = current_centers.copy()
                # If we found a perfect solution, we can stop early or keep going to polish?
                # If energy is 0, we are feasible.
                if best_energy < 1e-8:
                    # We have a feasible packing. We can stop this SA run.
                    # But maybe we can find a "better" feasible one (more centered)?
                    # For the purpose of finding a valid r, 0 is enough.
                    break
        else:
            # Revert
            current_centers[idx] = old_pos
            
        # Cool down
        temp *= cooling
        
    return best_centers, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initial guess
    # Use a grid to start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    initial_centers = grid[:N].copy()
    
    best_radius = compute_radius(initial_centers)
    best_centers = initial_centers.copy()
    
    # 2. Iterative Radius Increase
    # Start from a radius slightly smaller than current best to ensure feasibility
    current_r = best_radius * 0.95
    
    # Try to increase radius
    # Step size for radius increase
    r_step = 0.0005
    
    # Time budget for each radius attempt
    # We have ~140s.
    # Let's allocate time based on number of attempts.
    # If we do 50 attempts, 2.8s each.
    
    # Let's do a binary search style or linear increase?
    # Linear increase is safer if we can check feasibility quickly.
    # SA is stochastic.
    
    # Strategy:
    # While time permits:
    #   1. Pick a target r = current_r + r_step
    #   2. Run SA to find feasible packing for r
    #   3. If found (energy ~ 0), update current_r, best_radius, best_centers
    #   4. If not found, maybe reduce r_step or try a different starting point?
    #      Or just stop if we fail multiple times?
    
    # To make it robust, we should keep the best feasible centers found for the *current* r.
    # If we fail to find a feasible packing for r_new, we revert to the last known feasible packing for r_old.
    
    last_feasible_centers = initial_centers.copy()
    last_feasible_r = best_radius
    
    # We will try to push the radius up.
    # We need a good starting point for SA. The last feasible configuration is the best start.
    
    # Parameters for SA
    sa_time_budget = 2.0 # seconds per radius attempt
    
    while time.time() < deadline - 10.0:
        target_r = last_feasible_r + r_step
        
        # Run SA
        sa_start = time.time()
        # We pass the last feasible centers as the start
        new_centers, energy = _simulated_annealing_feasibility(
            last_feasible_centers, 
            target_r, 
            start_time, 
            deadline, 
            initial_temp=0.01, 
            cooling=0.9995, 
            max_iters=2000 # Cap iterations, time check inside
        )
        
        # Check if we found a feasible solution
        # Energy should be very small
        if energy < 1e-6:
            # It's feasible!
            # Verify with exact radius calculation
            actual_r = compute_radius(new_centers)
            if actual_r >= target_r - 1e-8:
                last_feasible_r = actual_r
                last_feasible_centers = new_centers.copy()
                
                if last_feasible_r > best_radius:
                    best_radius = last_feasible_r
                    best_centers = last_feasible_centers.copy()
                    
                # Maybe increase step size slightly if we are doing well?
                # Or keep it constant.
                # Let's keep it constant for stability.
            else:
                # Numerical issues? Treat as failure
                pass
        else:
            # Failed to find feasible packing for target_r
            # We can try a smaller step next time, or just stop.
            # To avoid getting stuck, let's reduce the step size.
            r_step *= 0.8
            if r_step < 1e-6:
                break
            
            # Also, it might be good to "relax" the current centers a bit before trying again
            # But we are already using the last feasible ones.
            
            # If we fail too many times in a row, we might be at a local optimum.
            # Let's try a random restart from a perturbed version of the best known packing?
            # Or just accept that we've hit a wall.
            
            # Let's try one random restart if we have time
            if time.time() < deadline - 15.0:
                # Perturb best centers
                perturbed = best_centers + np.random.normal(0, 0.02, size=(N, 2))
                # Clip to box
                perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
                perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
                
                # Try to find feasible packing for a slightly smaller r? 
                # No, we want to increase. 
                # If we can't increase from the current config, maybe a different config allows it.
                # Let's try to find a feasible packing for the *current* last_feasible_r using the perturbed start.
                # If we find a *different* feasible packing for the same r, we might be able to push further.
                
                alt_centers, alt_energy = _simulated_annealing_feasibility(
                    perturbed, 
                    last_feasible_r, 
                    start_time, 
                    deadline,
                    initial_temp=0.05, # Higher temp for exploration
                    cooling=0.999,
                    max_iters=1000
                )
                
                if alt_energy < 1e-6:
                    # Found a different feasible packing for the same radius
                    # Use it as the new starting point
                    last_feasible_centers = alt_centers
                    # Don't update best_radius, it's the same.
                    # But we have a new "base" to push from.
        
        # Check time
        if time.time() > deadline - 5.0:
            break

    return best_centers

# Test locally if needed
if __name__ == "__main__":
    c = construct()
    r = compute_radius(c)
    print(f"Radius: {r}")
    print(f"Centers: {c[:5]}")
