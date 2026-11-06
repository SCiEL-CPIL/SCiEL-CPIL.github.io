"""
Program for packing 50 equal circles in a unit circle.
Strategy: Physics-based simulation (Molecular Dynamics) with Verlet integration.
This avoids the slow convergence and local minima traps of gradient-based methods
on non-smooth objectives by using forces to push circles apart.
"""
import numpy as np
import time
import random

N = 50
DEADLINE = time.time() + 140.0  # Safety margin out of 150s

def compute_radius(points):
    """Compute the packing radius for given centres."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def fast_radius(points):
    """Fast radius computation."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    return min(boundary_dists.min(), min_pair_dist / 2.0)

def simulate_packing(initial_points, n_steps=2000, dt=0.001, damping=0.95):
    """
    Perform molecular dynamics simulation to maximize minimum distance.
    Uses repulsive forces between circles and a soft boundary constraint.
    """
    # Convert to float64 for precision
    x = initial_points.copy().astype(np.float64)
    v = np.zeros_like(x)
    
    # Parameters
    k_rep = 1000.0  # Repulsion strength
    k_bound = 5000.0 # Boundary stiffness
    softness = 0.1   # Softness of repulsion (avoid infinite force at overlap)
    
    best_x = x.copy()
    best_r = fast_radius(x)
    
    # Pre-allocate
    forces = np.zeros_like(x)
    
    for step in range(n_steps):
        if time.time() > DEADLINE - 5.0:
            break
            
        # Compute forces
        # 1. Pairwise repulsion
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-6, 1e-6, dists)
        
        # Force magnitude: k * (1/d^2) is common, but k*(1/d - 1) or similar.
        # Let's use a strong repulsive force that decays quickly.
        # F = k * (1 / d^2) * unit_vector
        # To keep it stable, we cap the force.
        inv_d2 = 1.0 / (dists_safe * dists_safe)
        # Mask diagonal
        np.fill_diagonal(inv_d2, 0.0)
        
        # Unit vectors
        unit_vecs = diff / dists_safe[:, :, np.newaxis]
        np.fill_diagonal(unit_vecs[:, :, 0], 0.0) # Placeholder, we mask below
        # Actually, simpler:
        # F_ij = k * (1/d^2) * (x_i - x_j)/d
        # Sum over j for each i
        
        # Vectorized force calculation
        # F_i = sum_{j != i} k * (x_i - x_j) / |x_i - x_j|^3
        # dists_safe^3
        dists_cubed = dists_safe ** 3
        np.fill_diagonal(dists_cubed, 1.0) # Avoid div by 0 in division
        
        # Factor for each pair
        pair_factor = k_rep / dists_cubed
        np.fill_diagonal(pair_factor, 0.0)
        
        # Force on i from j: pair_factor[i,j] * (x_i - x_j)
        # F_i = sum_j pair_factor[i,j] * (x_i - x_j)
        # = x_i * sum_j pair_factor[i,j] - sum_j pair_factor[i,j] * x_j
        
        sum_factors = np.sum(pair_factor, axis=1)
        # Weighted sum of positions
        # pair_factor is (N, N), x is (N, 2)
        weighted_sum = np.sum(pair_factor[:, :, np.newaxis] * x[np.newaxis, :, :], axis=1)
        
        pair_forces = x * sum_factors[:, np.newaxis] - weighted_sum
        
        # 2. Boundary force
        # Push inwards if close to boundary
        norms = np.linalg.norm(x, axis=1)
        # Soft boundary: if norm > 1 - softness, apply force
        # F_bound = -k_bound * (norm - (1 - softness)) * unit_vector if norm > 1-softness
        # Or simpler: always apply a spring to center? No, we want them to push to boundary.
        # We want to maximize min distance. The boundary acts as a hard wall.
        # If a point is outside, push it in. If inside, no force?
        # Actually, to pack tightly, we want them to be pushed against each other and the boundary.
        # A common trick: add a small attraction to the boundary or just a hard constraint.
        # Let's use a potential that penalizes being too close to the boundary? No, we want to use the space.
        # The repulsion handles the internal packing. The boundary just needs to contain them.
        # If we just use repulsion, they will fly out.
        # So we need a boundary force that keeps them inside.
        # F_bound = -k_bound * (norm - R_max) * (x / norm) for norm > R_max
        # But we want them to be able to reach the boundary.
        # So, let's use a "soft wall" that only activates if they cross R=1.
        # And maybe a small "gravity" towards the center to prevent clustering?
        # Actually, pure repulsion + hard boundary is enough if we damp velocity.
        
        boundary_forces = np.zeros_like(x)
        for i in range(N):
            if norms[i] > 0.99: # Start pushing if close
                # Push inwards
                push = (norms[i] - 0.99) * k_bound
                boundary_forces[i] = -push * (x[i] / norms[i])
        
        total_forces = pair_forces + boundary_forces
        
        # Verlet Integration
        v = v * damping + total_forces * dt
        x = x + v * dt
        
        # Hard constraint: clip to unit circle
        new_norms = np.linalg.norm(x, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            x[mask] = x[mask] * (1.0 / new_norms[mask])[:, np.newaxis]
            # Kill velocity component normal to boundary
            for i in np.where(mask)[0]:
                n = x[i] / np.linalg.norm(x[i])
                v_dot_n = np.dot(v[i], n)
                v[i] -= v_dot_n * n
        
        # Check improvement
        current_r = fast_radius(x)
        if current_r > best_r:
            best_r = current_r
            best_x = x.copy()
            
    return best_x, best_r

def local_polish(points, n_steps=200, step_size=0.005):
    """
    Simple coordinate descent polish to fix minor overlaps or boundary violations
    and squeeze out the last bit of radius.
    """
    x = points.copy()
    current_r = fast_radius(x)
    
    for _ in range(n_steps):
        if time.time() > DEADLINE - 2.0:
            break
        improved = False
        # Random order to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
            # Try small moves
            best_x_i = x[i].copy()
            best_r_i = current_r
            
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_x_i = x[i] + np.array([dx, dy])
                    if np.linalg.norm(new_x_i) >= 1.0:
                        continue
                    x[i] = new_x_i
                    new_r = fast_radius(x)
                    if new_r > best_r_i:
                        best_r_i = new_r
                        best_x_i = new_x_i.copy()
                    else:
                        x[i] = best_x_i # revert
            if best_r_i > current_r:
                x[i] = best_x_i
                current_r = best_r_i
                improved = True
        if not improved:
            # Reduce step size if stuck
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return x, current_r

def construct():
    global DEADLINE
    DEADLINE = time.time() + 140.0
    
    np.random.seed(42)
    random.seed(42)
    
    best_points = None
    best_radius = 0.0
    
    # Strategy: Run multiple simulations from different random starts
    n_restarts = 10
    time_per_restart = (DEADLINE - time.time()) / n_restarts
    
    for i in range(n_restarts):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Generate random seed
        # Use a hexagonal-like jitter or pure random
        if i % 2 == 0:
            # Hexagonal seed with jitter
            spacing = 0.2
            points_list = []
            for j in range(-5, 6):
                for k in range(-5, 6):
                    px = k * spacing + (0.5 * spacing if j % 2 else 0)
                    py = j * spacing * np.sqrt(3) / 2
                    if np.sqrt(px * px + py * py) < 0.95:
                        points_list.append([px + np.random.uniform(-0.02, 0.02), 
                                           py + np.random.uniform(-0.02, 0.02)])
            if len(points_list) >= N:
                # Select N points
                # Sort by distance to center to ensure coverage
                dists = np.linalg.norm(np.array(points_list), axis=1)
                indices = np.argsort(dists)
                seed_points = np.array(points_list)[indices[:N]]
            else:
                seed_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
        else:
            # Pure random
            seed_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
            
        # Run simulation
        # Adjust steps based on remaining time
        est_time_left = DEADLINE - time.time() - 10.0
        steps = int(min(5000, max(500, est_time_left * 50))) # Rough estimate
        
        sim_points, sim_r = simulate_packing(seed_points, n_steps=steps)
        
        if sim_r > best_radius:
            best_radius = sim_r
            best_points = sim_points.copy()
            
    # Final polish
    if best_points is not None and time.time() < DEADLINE - 5.0:
        polished_points, polished_r = local_polish(best_points)
        if polished_r > best_radius:
            best_points = polished_points
            best_radius = polished_r
            
    # Fallback if simulation failed or returned garbage
    if best_points is None or not np.all(np.isfinite(best_points)):
        # Fallback to simple hexagonal
        spacing = 0.2
        points_list = []
        for j in range(-5, 6):
            for k in range(-5, 6):
                px = k * spacing + (0.5 * spacing if j % 2 else 0)
                py = j * spacing * np.sqrt(3) / 2
                if np.sqrt(px * px + py * py) < 0.95:
                    points_list.append([px, py])
        if len(points_list) >= N:
            dists = np.linalg.norm(np.array(points_list), axis=1)
            indices = np.argsort(dists)
            best_points = np.array(points_list)[indices[:N]]
        else:
            best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
            
    # Final validation and clipping
    best_points = np.asarray(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if norms[i] >= 1.0:
            best_points[i] *= 0.999 / norms[i]
            
    return best_points

if __name__ == "__main__":
    points = construct()
    print("Final radius:", compute_radius(points))
