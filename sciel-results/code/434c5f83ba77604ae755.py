import numpy as np
from scipy.optimize import differential_evolution
import time

N = 50

def compute_radius(centers):
    """
    Compute the radius of the largest equal circles that can be placed 
    at the given centers in the square [-0.5, 0.5]^2.
    """
    # centers shape (N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Mask the diagonal and lower triangle
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    # Distance to boundaries
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    r = 0.5 * min(min_pair_dist, min_boundary_dist)
    
    return r

def _energy(x_flat, r_target):
    """
    Objective function for optimization.
    We want to maximize the minimum distance d_min = min(pair_dists, boundary_dists).
    Let D = d_min.
    We want D >= 2 * r_target.
    
    Penalty function:
    If D >= 2 * r_target:
        Energy = (2 * r_target - D)^2  (This is 0 if feasible and strictly increasing as D drops below target)
        Actually, we want to maximize D.
        Let's define the objective to minimize.
        If D >= 2*r_target:
            We are feasible. To push for larger D, we can minimize (2*r_target - D)^2? 
            No, that would be 0 for any D >= 2*r_target.
            We want to encourage D > 2*r_target.
            So if D >= 2*r_target, Energy = (2*r_target - D)^2 is not enough to distinguish D=2r from D=10r.
            
            Better:
            If D >= 2*r_target:
                Energy = -(D - 2*r_target)  # Minimize this -> Maximize D
            Else:
                Energy = (2*r_target - D)^2  # Penalize infeasibility quadratically
            
            This creates a "barrier" at D=2r_target.
    """
    centers = x_flat.reshape(N, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    min_pair_dist = np.min(dist[iu])
    
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary_dist = np.min(np.concatenate([d_left, d_right, d_bottom, d_top]))
    
    D = min(min_pair_dist, min_boundary_dist)
    
    target = 2.0 * r_target
    
    if D >= target:
        return -(D - target)
    else:
        return (target - D) ** 2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Initial guess for radius
    # Boll et al. radius is ~0.07137. Let's start a bit lower to ensure feasibility is found easily.
    # Area argument: 50 * pi * r^2 <= 1 => r <= 1/sqrt(50*pi) ~ 0.0798.
    # Let's start at 0.07.
    r_current = 0.070
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    rng = np.random.default_rng(42)
    
    # Strategy:
    # 1. Try to find a feasible packing for r_current.
    # 2. If successful, increment r_current and try again.
    # 3. If failed, decrement r_current slightly and retry with different seeds.
    
    # We will use Differential Evolution (DE) as the optimizer.
    # DE is robust and good for global search.
    # We need to limit the number of evaluations to fit in the time budget.
    
    # Pre-compute number of restarts allowed based on time
    # DE with popsize=20, maxiter=100 does 2000 evals.
    # Each eval is O(N^2) ~ 2500 ops. Very fast.
    # So we can do many restarts.
    
    max_restarts_per_radius = 3
    max_radius_steps = 50
    
    r_step = 0.0005 # Small step to ensure we don't jump over the optimum
    
    for r_step_idx in range(max_radius_steps):
        if time.time() > deadline - 10.0:
            break
            
        r_target = 0.070 + r_step_idx * r_step
        
        found_feasible = False
        
        for restart in range(max_restarts_per_radius):
            if time.time() > deadline - 5.0:
                break
                
            # Generate initial population for DE
            # We can use random init or a jittered grid.
            # DE handles the population internally, but we can provide 'init' as a list of points or 'random'.
            # Using 'random' is fine.
            
            try:
                # maxiter: 50 is small, but DE is powerful.
                # popsize: 15
                # tol: 1e-6
                res = differential_evolution(
                    _energy, 
                    bounds, 
                    args=(r_target,),
                    seed=42 + r_step_idx * 100 + restart,
                    maxiter=30,
                    popsize=15,
                    tol=1e-6,
                    init='sobol', # Use Sobol for better coverage
                    polish=False # We don't need gradient polish, DE is enough
                )
                
                # Check if the best solution is feasible
                # Feasible means energy is close to 0 or negative (if we pushed past target)
                # Actually, if D >= target, energy is -(D-target). 
                # If the optimizer found a solution with D >= target, energy <= 0.
                # If D < target, energy > 0.
                
                if res.fun <= 1e-6: # Allow small numerical error
                    # It's feasible!
                    # Let's calculate the actual radius of this configuration
                    c = res.x.reshape(N, 2)
                    actual_r = compute_radius(c)
                    
                    if actual_r > best_radius:
                        best_radius = actual_r
                        best_centers = c
                        
                    # If we found a feasible solution, we can try a slightly larger radius in the next iteration
                    # But we must ensure the current one is stored.
                    # To be safe, we only increment r_target if we are confident.
                    # Since we are stepping up, if this r_target worked, the next one (r_target + step) might also work if we start from this solution.
                    # However, DE restarts from scratch.
                    # So we just continue the loop to try r_target + step.
                    found_feasible = True
                    break # Stop restarting for this r_target, move to next larger r_target
                    
            except Exception:
                continue
        
        # If we didn't find a feasible solution for this r_target, 
        # we might have jumped too high.
        # In a real implementation, we'd back off. 
        # Here, we just keep trying the next step, hoping the random init hits a good spot.
        # Or we could decrease r_target. 
        # Given the time limit, simple forward stepping with multiple restarts is okay.
        # If we miss the optimum, we might end up with a suboptimal result.
        # To mitigate, we can keep the best result found so far.
        
    # Final check: if we have a best_centers, return it.
    # If not, return a grid (fallback).
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N].copy()
        
    return best_centers
