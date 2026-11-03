"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import time

def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))

def _get_delaunay_neighbors(tri):
    """Get a list of unique neighbor pairs from Delaunay triangulation."""
    neighbors = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                neighbors.add((min(simplex[i], simplex[j]), max(simplex[i], simplex[j])))
    return list(neighbors)

def _optimize_delaunay(centres, max_iter=2000, step_size=0.005, tol=1e-8):
    """
    Optimize centres using Delaunay triangulation to identify neighbors.
    Uses a force-directed approach where points repel each other if they are too close,
    and are attracted to the center if they are too close to the boundary.
    """
    centres = centres.copy()
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Precompute boundary constraints
    # We will iterate and update
    
    for iter in range(max_iter):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
            neighbors = _get_delaunay_neighbors(tri)
        except Exception:
            # If Delaunay fails (e.g., coincident points), add small noise
            centres += np.random.randn(N, 2) * 1e-6
            centres = np.clip(centres, -0.5, 0.5)
            continue
        
        # Compute forces
        forces = np.zeros_like(centres)
        
        # Pairwise repulsion from Delaunay neighbors
        # We want to maximize the minimum distance, so we push apart points that are close.
        # A simple model: Force proportional to 1/d^2 for d < threshold, or just 1/d^2 always.
        # To focus on the minimum, we can use a sharper potential.
        # Let's use: if d < 2*best_radius + margin, push apart.
        
        # Actually, a global repulsion that is strong for short distances works well.
        # F = k / d^2 * direction
        # To avoid numerical issues, clamp d.
        
        for i, j in neighbors:
            d_vec = centres[i] - centres[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-8:
                # Coincident points, push randomly
                d_vec = np.random.randn(2)
                d = np.linalg.norm(d_vec)
            
            # Force magnitude
            # We want to equalize distances. A spring-like force that pushes apart if d is small.
            # Let's use a potential that penalizes d < r_target.
            # Here we don't know r_target exactly, so we use a generic repulsion.
            # Stronger repulsion for shorter distances.
            force_mag = 1.0 / (d * d + 1e-6)
            
            # Normalize direction
            d_dir = d_vec / d
            forces[i] += force_mag * d_dir
            forces[j] -= force_mag * d_dir
        
        # Boundary repulsion
        # Points close to the boundary should be pushed inward.
        # Distance to border: 0.5 - |c|
        # If dist < r_target, push in.
        # We can use a similar force model.
        dist_to_border = 0.5 - np.abs(centres)
        min_dist_to_border = np.min(dist_to_border)
        
        # For each point, calculate inward force
        # If a point is close to a specific wall, push it away from that wall.
        for i in range(N):
            x, y = centres[i]
            # Distance to each wall
            dx = 0.5 - abs(x)
            dy = 0.5 - abs(y)
            
            # Force from x-walls
            if dx < 0.1: # Only apply if somewhat close
                # Push towards center in x
                force_x = 1.0 / (dx * dx + 1e-6) * (-1 if x > 0 else 1)
                forces[i][0] += force_x
            
            # Force from y-walls
            if dy < 0.1:
                force_y = 1.0 / (dy * dy + 1e-6) * (-1 if y > 0 else 1)
                forces[i][1] += force_y
        
        # Update positions
        # Use a step size that decays or is adaptive
        # We want to move in the direction of the force
        # But we must ensure we don't jump over the minimum.
        # A simple gradient step: c += step * force
        
        # To prevent oscillation, we can use a smaller step size.
        # Also, we should check if the move improves the radius.
        
        # Calculate new centres
        new_centres = centres + step_size * forces
        
        # Clip to square
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Evaluate new radius
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + tol:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres
            # Reduce step size slightly if we improved, to settle down
            step_size *= 0.995
        else:
            # If we didn't improve, try a smaller step or random jitter
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                # Try random restarts or jitter
                if iter > max_iter // 2:
                    # Add small noise to escape local minima
                    noise = np.random.randn(N, 2) * 0.001
                    new_centres = centres + noise
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    new_radius = _compute_radius(new_centres)
                    if new_radius > best_radius + tol:
                        best_radius = new_radius
                        best_centres = new_centres.copy()
                        centres = new_centres
                        step_size = 0.005 # Reset step size
                    else:
                        break
                else:
                    break
            
            # If step size is too small and we've been stuck, break
            if step_size < 1e-5 and iter > 100:
                # Check if we are stuck
                pass
    
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
    
    # Run multiple iterations of local optimization with different random restarts
    # Use a few restarts to escape local minima
    for restart in range(5):
        if restart == 0:
            current_centres = centres.copy()
        else:
            # Perturb the best solution found so far
            current_centres = best_centres + np.random.randn(N, 2) * 0.01
            current_centres = np.clip(current_centres, -0.5, 0.5)
        
        # Run Delaunay-based optimization
        optimised, radius = _optimize_delaunay(current_centres, max_iter=1000, step_size=0.005)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
