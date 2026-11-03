"""Improved program for the 200-circle packing task using Delaunay-based force optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    # Using broadcasting for speed
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _delaunay_force_optimization(centres, max_iterations=1000, initial_step=0.01, cooling=0.999):
    """
    Optimizes circle positions using Delaunay triangulation to identify
    close neighbors and apply repulsive forces.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    step_size = initial_step
    n = len(centres)
    
    for iteration in range(max_iterations):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except:
            # If Delaunay fails (e.g., duplicate points), perturb slightly
            centres += np.random.randn(n, 2) * 1e-5
            continue
        
        # Get unique edges from triangulation
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(edge)
        
        # Initialize forces
        forces = np.zeros_like(centres)
        
        # Apply repulsive forces along Delaunay edges
        # We want to push apart any pair that is closer than 2 * best_radius (or current min dist)
        # To be more robust, we push apart all Delaunay edges, weighted by how "tight" they are.
        for i, j in edges:
            d = centres[i] - centres[j]
            dist = np.linalg.norm(d)
            if dist < 1e-8:
                d = np.random.randn(2)
                dist = np.linalg.norm(d)
            
            # Force magnitude: stronger if distance is smaller
            # Use a simple inverse distance squared or linear repulsion
            # Let's use a force that is zero if dist > 2*best_radius, and grows as dist decreases
            # Actually, simpler: just repel all Delaunay neighbors to maintain a minimum distance.
            # The "target" distance is roughly 2 * best_radius.
            # If dist < 2 * best_radius, push apart.
            # But best_radius is changing. Let's just use a constant repulsion strength scaled by 1/dist.
            force_mag = 1.0 / (dist * dist)
            force_vec = (d / dist) * force_mag
            
            forces[i] += force_vec
            forces[j] -= force_vec
            
        # Apply boundary forces
        # If a circle is too close to the boundary, push it inward
        # Distance to boundary: 0.5 - abs(x) and 0.5 - abs(y)
        # We want to maximize the minimum of these.
        # So if 0.5 - abs(x) < best_radius, push towards center.
        
        # Identify active boundaries
        dist_to_border_x = 0.5 - np.abs(centres[:, 0])
        dist_to_border_y = 0.5 - np.abs(centres[:, 1])
        
        # For each point, if it's close to a wall, add force towards center
        # Threshold: best_radius
        for i in range(n):
            # X-axis
            if dist_to_border_x[i] < best_radius:
                # Push towards center (x=0)
                # Direction is -sign(x)
                dir_x = -np.sign(centres[i, 0])
                if dir_x == 0:
                    dir_x = 1.0
                # Force magnitude proportional to violation
                violation = best_radius - dist_to_border_x[i]
                forces[i, 0] += dir_x * violation * 10.0 # Scale factor
                
            # Y-axis
            if dist_to_border_y[i] < best_radius:
                dir_y = -np.sign(centres[i, 1])
                if dir_y == 0:
                    dir_y = 1.0
                violation = best_radius - dist_to_border_y[i]
                forces[i, 1] += dir_y * violation * 10.0

        # Update positions
        # Normalize forces to prevent huge jumps
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            forces = forces / max_force
        
        new_centres = centres + forces * step_size
        
        # Clip to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Calculate new radius
        new_radius = _compute_radius(new_centres)
        
        # Accept move if it improves or is equal (with some probability to escape minima? 
        # No, strict improvement is safer for packing, but might get stuck.
        # Let's accept if new_radius >= best_radius - 1e-9 (allow slight decrease for exploration? No, keep strict).
        # Actually, if we are stuck, we need to perturb.
        
        if new_radius > best_radius:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres.copy()
        else:
            # If no improvement, reduce step size and add noise
            step_size *= cooling
            if step_size < 1e-7:
                break
            
            # Add small random noise to escape local minima
            if iteration % 10 == 0:
                noise = np.random.randn(n, 2) * step_size * 0.5
                new_centres = centres + noise
                new_centres = np.clip(new_centres, -0.5, 0.5)
                r_noisy = _compute_radius(new_centres)
                if r_noisy > best_radius:
                    best_radius = r_noisy
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
                else:
                    # If noise didn't help, just keep current centres but maybe randomize one point
                    idx = np.random.randint(n)
                    new_centres[idx] += np.random.randn(2) * step_size
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    r_noisy = _compute_radius(new_centres)
                    if r_noisy > best_radius:
                        best_radius = r_noisy
                        best_centres = new_centres.copy()
                        centres = new_centres.copy()

    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run multiple restarts
    # 1. Optimized Grid
    # 2. Random Start
    # 3. Jittered Grid
    
    starts = []
    
    # Start 1: Grid
    starts.append(centres.copy())
    
    # Start 2: Random
    np.random.seed(42)
    rand_centres = np.random.uniform(-0.45, 0.45, (N, 2))
    starts.append(rand_centres)
    
    # Start 3: Jittered Grid
    jittered = centres + np.random.uniform(-0.01, 0.01, (N, 2))
    jittered = np.clip(jittered, -0.5, 0.5)
    starts.append(jittered)
    
    for start_centres in starts:
        optimised, radius = _delaunay_force_optimization(start_centres, max_iterations=500, initial_step=0.005)
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
            
    # Final polishing pass with smaller steps
    final_optimised, final_radius = _delaunay_force_optimization(best_centres, max_iterations=1000, initial_step=0.001)
    if final_radius > best_radius:
        best_centres = final_optimised.copy()
        best_radius = final_radius
        
    return best_centres
# EVOLVE-BLOCK-END
