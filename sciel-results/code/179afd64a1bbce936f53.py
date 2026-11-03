"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Calculate the packing radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
    
    # Compute all pairwise distances
    # Efficiently using broadcasting
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    # Centers must be in [-0.5, 0.5]
    # Distance to wall for coordinate k is 0.5 - abs(c_ik)
    dists_to_wall = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_wall)
    
    # Radius is half the minimum distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    # Ensure non-negative and valid
    if r <= 0:
        return 0.0
        
    return r

def _lj_energy_and_grad(centers, sigma=0.2, epsilon=1.0, wall_k=100.0):
    """
    Calculate Lennard-Jones-like energy and gradient.
    We use a smooth repulsion potential to avoid singularities.
    E = sum_{i<j} V(r_ij) + sum_i V_wall(c_i)
    V(r) = epsilon * (sigma/r)^12 - epsilon * (sigma/r)^6 is standard LJ, but singular at 0.
    For packing, we mostly care about repulsion. A simple inverse power law V(r) = 1/r^p is hard to differentiate near 0 if p is large.
    Let's use a softened potential: V(r) = epsilon * (sigma^2 / (r^2 + delta))^6 
    This is smooth everywhere.
    """
    n = centers.shape[0]
    # Pairwise distances
    # To save memory and time, we might do it in chunks or just full matrix for N=100 (100x100 is fine)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    r_sq = np.sum(diff**2, axis=-1)
    
    # Regularization to avoid division by zero
    delta_sq = 1e-8 
    r_sq_reg = r_sq + delta_sq
    
    # Potential: V = epsilon * (sigma^2 / r_sq_reg)^6
    # This decays as 1/r^12 for large r, and is smooth at 0.
    # Gradient wrt positions:
    # dV/dr_sq = -6 * epsilon * sigma^12 * r_sq_reg^-7
    # d(r_sq)/d(c_i) = 2 * (c_i - c_j)
    # So dV/d(c_i) = -12 * epsilon * sigma^12 * r_sq_reg^-7 * (c_i - c_j) for all j != i
    
    # Let's compute the force (negative gradient) directly, then negate for gradient? 
    # No, SGLD uses gradient. Let's compute gradient.
    
    # Term: (sigma^2 / r_sq_reg)^6 = sigma^12 * r_sq_reg^-6
    # d/d(c_i) sum_j V_ij = sum_j 6 * epsilon * sigma^12 * (-7) * r_sq_reg_ij^-7 * d(r_sq_ij)/d(c_i)
    # d(r_sq_ij)/d(c_i) = 2 * (c_i - c_j)
    # So grad_i = sum_j -84 * epsilon * sigma^12 * r_sq_reg_ij^-7 * (c_i - c_j)
    
    # Vectorized computation:
    # r_sq_reg is (N, N)
    # We need sum over j for each i of w_ij * (c_i - c_j)
    
    # Let's use a simpler potential for stability and speed: V(r) = 1 / r^2 + small_const? 
    # Actually, the standard approach in these tasks is often just maximizing the minimum distance.
    # But SGLD needs a smooth energy. 
    # Let's stick to V(r) = epsilon * (sigma^2 / (r^2 + delta_sq))^6.
    
    sigma2 = sigma * sigma
    # r_sq_reg^-7
    inv_r_sq_7 = r_sq_reg ** (-7.0)
    # Weights w_ij = -84 * epsilon * sigma^12 * inv_r_sq_7
    # Note: diagonal terms are included but (c_i - c_i) is 0, so they vanish.
    weights = -84.0 * epsilon * (sigma2 ** 6) * inv_r_sq_7
    
    # Gradient contribution from pairs:
    # grad[i] = sum_j weights[i,j] * (c_i - c_j)
    # c_i - c_j is diff[i,j]
    # grad_pair = weights @ diff ? No.
    # grad[i] = sum_j w[i,j] * diff[i,j]
    # This is a matrix multiplication if we treat diff as vectors?
    # diff is (N, N, 3). weights is (N, N).
    # We want sum over axis 1 (j) of weights[:, :, None] * diff.
    grad_pair = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Wall potential
    # V_wall(x) = wall_k * sum_k (0.5 - |x_k|)^2 ? No, this is 0 at boundary, max inside.
    # We want repulsion from walls. 
    # Let V_wall = wall_k * sum_k (0.5 - x_k)^2 if x_k > 0? No.
    # Standard soft wall: V(x) = wall_k * (0.5 - |x|)^2 is a minimum at 0.5? No.
    # We want to push particles AWAY from walls.
    # So potential should increase as particle gets close to wall.
    # Let V_wall(x) = wall_k * sum_k (0.5 - |x_k|)^2 is low near walls? No, 0.5-|x| is small near walls.
    # So this potential is LOW near walls. We want HIGH near walls.
    # So V_wall(x) = wall_k * sum_k ( (0.5 - |x_k|) )^(-2) ? Singular.
    # Let's use V_wall(x) = wall_k * exp(-beta * (0.5 - |x|)). This is high inside, low at wall? No.
    # We want to keep particles inside. 
    # If we minimize E, we want E to be HIGH if particles go out or touch walls?
    # Usually, we maximize radius. 
    # In SGLD, we minimize Energy. 
    # So Energy should be HIGH when spheres overlap or touch walls.
    # Pair energy V(r) is HIGH when r is small. Good.
    # Wall energy should be HIGH when distance to wall is small.
    # Let d_w = min_k (0.5 - |c_ik|). 
    # V_wall = wall_k * exp(-gamma * d_w)? This is high when d_w is small.
    # Derivative is messy with min.
    
    # Alternative: Use a smooth box potential.
    # V_wall(x_i) = wall_k * sum_k (0.5 - x_ik)^2 for x_ik > 0? No.
    # Let's use a simple quadratic penalty that is active when close to boundary.
    # Actually, since we clip positions to the box, the hard constraint is satisfied.
    # The gradient from the clip is 0 in the interior. 
    # To encourage moving away from walls, we can add a potential that increases towards the boundary.
    # V_wall(c) = wall_k * sum_i sum_k (c_ik^2). This pushes everything to the center. 
    # This is not what we want; we want to fill the cube.
    # The pair repulsion will push them apart. The walls confine them.
    # If we just use pair repulsion, they will cluster in the center? No, repulsion pushes them apart.
    # With no external force, they expand until they hit the walls.
    # Once they hit the walls (clipped), the gradient is 0 (if we just clip).
    # To make them "feel" the wall, we need a potential gradient near the wall.
    
    # Let's add a potential that rises sharply near the walls.
    # V_wall(x) = wall_k * sum_k (0.5 - |x_k|)^(-p) is bad.
    # Let's use V_wall(x) = wall_k * sum_k exp(-lambda * (0.5 - |x_k|)).
    # This is large when x is close to 0.5 (wall). Small when x is near 0.
    # This encourages particles to stay away from walls? No, it encourages them to be near center.
    # We want them to be near the walls to maximize the sphere size?
    # Actually, the optimal packing fills the cube. 
    # If we minimize repulsion, we want to maximize distances.
    # If we have a fixed volume, maximizing min distance is the goal.
    # The "pressure" of the repulsion pushes them to the boundaries.
    # If we don't have a wall potential, the gradient is 0 at the boundary (after clipping).
    # This might cause them to stick to the boundary and not optimize the arrangement there.
    # A small wall potential that penalizes being "too close" to the wall might help?
    # No, we WANT them to be as close to the wall as possible (distance 0.5 - r).
    # So we don't want a repulsive wall potential. We want an ATTRACTIVE wall potential?
    # Or just let the pair repulsion push them out.
    # The issue is that at the boundary, the gradient from other particles might point inwards,
    # and the clipping just stops the motion. 
    # SGLD noise will push them back and forth.
    # Let's rely on the pair repulsion and clipping. 
    # However, to help "pack" them densely against the walls, a slight attraction to the walls
    # or just neutral walls might be best. 
    # If we add a repulsive wall potential, they will move inwards, reducing the effective volume.
    # So, NO wall potential in the energy function is likely better for maximizing radius,
    # relying on the repulsion to push them to the boundary.
    # The gradient of the pair potential will handle the internal arrangement.
    
    # So, grad = grad_pair.
    
    return np.sum(grad_pair, axis=1) * 0 + grad_pair # Just returning grad_pair. The sum was redundant in my thought process above, let's re-verify.
    # grad_pair[i] = sum_j w[i,j] * diff[i,j]. This is correct.

def construct():
    start_time = time.time()
    timeout = 115.0 # Safety margin
    
    # Initial positions: Random in the cube
    # Slightly inside to avoid initial gradient issues if any
    centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    best_centers = centers.copy()
    best_score = _score(best_centers)
    
    # SGLD Parameters
    # Step size eta
    eta = 0.01
    # Initial temperature (noise amplitude)
    T = 0.05
    # Cooling rate
    cooling_rate = 0.99
    min_T = 1e-4
    
    # Precompute constants for energy/grad
    sigma = 0.2
    epsilon = 1.0
    
    # Iterate
    while True:
        # Check time
        if time.time() - start_time > timeout:
            break
            
        # Calculate gradient
        grad = _lj_energy_and_grad(centers, sigma, epsilon)
        
        # Langevin update
        # x_{t+1} = x_t + eta * grad + sqrt(2 * eta * T) * noise
        # Note: grad is the gradient of the Energy. We want to MINIMIZE Energy.
        # So we move in direction of -grad.
        # Wait. 
        # If E is high when close, grad points away from close neighbors?
        # V(r) ~ 1/r^12. As r decreases, V increases. 
        # dV/dr is negative? No. 
        # Let r decrease. V increases. 
        # Gradient points in direction of INCREASING V.
        # So grad points towards the neighbor (increasing overlap).
        # We want to DECREASE V. So we move in direction of -grad.
        
        noise = np.random.normal(0, 1, size=centers.shape)
        noise_amp = np.sqrt(2 * eta * T)
        
        centers = centers - eta * grad + noise_amp * noise
        
        # Clip to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        # Evaluate score
        current_score = _score(centers)
        
        if current_score > best_score:
            best_score = current_score
            best_centers = centers.copy()
            
        # Anneal temperature
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
        # Adaptive step size? 
        # If score is improving, keep eta. If stuck, maybe increase noise?
        # For simplicity, keep fixed annealing schedule.
        
        # Periodic restart if stuck?
        # If no improvement for a long time, we could randomize slightly.
        # But the deadline guard is primary.
        
    return best_centers
# EVOLVE-BLOCK-END
