"""
Optimized packing of 50 spheres in a unit cube.
Strategy:
1. Initialize with a perturbed grid.
2. Run a damped molecular dynamics simulation (Repulsive Force) to escape local minima.
3. Polish the result using L-BFGS-B with a soft-min objective for high precision.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5
EPS = 1e-12

def _compute_score(centers):
    """Compute the packing radius for a set of centers."""
    # Clip to valid range to ensure validity
    centers = np.clip(centers, -CUBE_HALF + EPS, CUBE_HALF - EPS)
    
    # Wall distance
    wall_dists = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall_dists)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_wall, min_pair)

def _soft_min_obj(params, beta, centers_shape=(N, 3)):
    """
    Objective: minimize -soft_min(distances)
    """
    centers = params.reshape(centers_shape)
    centers = np.clip(centers, -CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)
    
    # Wall distances
    wall_dists = CUBE_HALF - np.abs(centers)
    
    # Pairwise distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Collect all distances
    # We only need upper triangle for pairwise to save memory/time, but full is fine for N=50
    all_dists = np.concatenate([wall_dists.flatten(), dists[np.triu_indices(N, k=1)]])
    
    # Soft minimum: -(1/beta) * log(sum(exp(-beta * d)))
    min_d = np.min(all_dists)
    exp_vals = np.exp(-beta * (all_dists - min_d))
    soft_min_val = min_d - np.log(np.sum(exp_vals)) / beta
    
    return -soft_min_val

def _dynamics_phase(centers, time_budget):
    """
    Run a damped molecular dynamics simulation to maximize the minimum distance.
    Force law: F = k / r^2 (repulsive) for pairs and walls.
    """
    positions = centers.copy().astype(np.float64)
    velocities = np.zeros_like(positions)
    
    # Parameters
    k_force = 1.0
    damping = 0.95
    dt = 0.01
    max_steps = int(time_budget / (N * 0.0001)) # Rough estimate, we check time explicitly
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(positions)
        
        # Pairwise forces
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0) 
        # Force magnitude: k / r^2
        # Direction: diff / r
        # F_ij = k * diff_ij / r_ij^3
        inv_r3 = 1.0 / (dists**3)
        np.fill_diagonal(inv_r3, 0.0)
        
        force_mat = k_force * diff * inv_r3[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        
        # Wall forces
        # If x < -0.5 + r, force is positive (pushing right)
        # Here we treat "r" as a virtual radius that is effectively the current min distance?
        # Better: just push hard if close to wall.
        # Let's use a similar 1/r^2 law for walls.
        # Distance to wall
        d_left = positions[:, 0] - (-CUBE_HALF)
        d_right = CUBE_HALF - positions[:, 0]
        d_bottom = positions[:, 1] - (-CUBE_HALF)
        d_top = CUBE_HALF - positions[:, 1]
        d_back = positions[:, 2] - (-CUBE_HALF)
        d_front = CUBE_HALF - positions[:, 2]
        
        # Clamp distances to avoid singularity
        d_left = np.clip(d_left, 1e-6, None)
        d_right = np.clip(d_right, 1e-6, None)
        d_bottom = np.clip(d_bottom, 1e-6, None)
        d_top = np.clip(d_top, 1e-6, None)
        d_back = np.clip(d_back, 1e-6, None)
        d_front = np.clip(d_front, 1e-6, None)
        
        f_left = k_force / (d_left**2)
        f_right = -k_force / (d_right**2)
        f_bottom = k_force / (d_bottom**2)
        f_top = -k_force / (d_top**2)
        f_back = k_force / (d_back**2)
        f_front = -k_force / (d_front**2)
        
        forces[:, 0] += f_left + f_right
        forces[:, 1] += f_bottom + f_top
        forces[:, 2] += f_back + f_front
        
        # Update velocities and positions (Velocity Verlet simplified with damping)
        velocities = velocities * damping + forces * dt
        positions = positions + velocities * dt
        
        # Hard clip to keep inside (safety)
        positions = np.clip(positions, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
    return positions

def _build_initial_grid():
    """
    Build a good initial configuration.
    """
    # 4x4x4 grid gives 64 spots. We take 50.
    # To be more symmetric, maybe 5x4x3 = 60? Or just 4x4x4.
    # Let's use a 4x4x4 grid, remove corners if possible or just take first 50.
    # Actually, a jittered 5x4x3 might be better distributed?
    # 5*4*3 = 60. 
    # Let's stick to 4x4x4 = 64, remove 14.
    
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    perturbation = 0.05 * np.random.randn(len(grid), 3)
    grid = grid + perturbation
    grid = np.clip(grid, -CUBE_HALF + 0.01, CUBE_HALF - 0.01)
    
    return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initial Configuration
    centers = _build_initial_grid()
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # 2. Dynamics Phase
    # Allocate 80 seconds to dynamics
    dyn_time_budget = min(80.0, deadline - time.time() - 20.0)
    if dyn_time_budget > 10.0:
        dyn_centers = _dynamics_phase(centers, dyn_time_budget)
        dyn_score = _compute_score(dyn_centers)
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dyn_centers.copy()
            centers = dyn_centers
        else:
            # Keep original if dynamics made it worse (unlikely but possible with bad params)
            centers = best_centers
    
    # 3. Polish Phase
    # Use L-BFGS-B with increasing beta
    # Allocate remaining time (approx 40-50s) to polishing
    polish_time_budget = deadline - time.time() - 5.0
    
    if polish_time_budget > 5.0:
        # Schedule: start with moderate beta, increase
        beta_schedule = [500, 2000, 5000, 10000]
        
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
            
            try:
                # Use current best as start
                start_params = centers.flatten()
                
                result = minimize(
                    _soft_min_obj,
                    start_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-CUBE_HALF + 1e-10, CUBE_HALF - 1e-10)] * (N * 3),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                new_centers = result.x.reshape(N, 3)
                new_centers = np.clip(new_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    centers = new_centers
            except Exception:
                pass

    # Final validity check
    best_centers = np.clip(best_centers, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
    
    return best_centers

# EVOLVE-BLOCK-START
# The main construct() function is defined above
# EVOLVE-BLOCK-END
