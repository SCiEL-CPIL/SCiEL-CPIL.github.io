"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
import random
import copy


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


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _voronoi_centroids(tri, centres):
    """Compute centroids of Voronoi cells for each point."""
    vor = Voronoi(centres)
    centroids = np.zeros_like(centres)
    
    for i, point in enumerate(centres):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 not in region and len(region) > 0:
            # Get vertices of the Voronoi cell
            verts = vor.vertices[region]
            centroids[i] = np.mean(verts, axis=0)
        else:
            # If unbounded (on convex hull), keep original position or project
            # For packing, we want to keep them inside, so maybe just keep original
            # or push towards center? Keeping original is safer for boundary points.
            centroids[i] = point
            
    return centroids


def _force_directed_step(centres, tri, vor_centroids, step_size):
    """
    Perform a force-directed relaxation step using Delaunay edges and Voronoi centroids.
    """
    n = len(centres)
    forces = np.zeros_like(centres)
    
    # 1. Repulsive forces from Delaunay neighbors
    edges = _get_delaunay_edges(tri)
    for i, j in edges:
        diff = centres[i] - centres[j]
        dist_sq = np.dot(diff, diff)
        if dist_sq < 1e-10:
            # Random small push to avoid division by zero
            diff = np.random.randn(2) * 1e-5
            dist_sq = np.dot(diff, diff)
        
        # Force magnitude ~ 1/d^2
        # We scale it to be manageable
        f_mag = 1.0 / dist_sq
        f_vec = diff * f_mag
        
        forces[i] += f_vec
        forces[j] -= f_vec
        
    # 2. Boundary forces
    # Push points away from boundaries if they are close
    margin = 0.05 # Heuristic margin
    for i in range(n):
        x, y = centres[i]
        # Left
        if x < -0.5 + margin:
            forces[i, 0] += (margin - (x + 0.5)) * 10.0
        # Right
        if x > 0.5 - margin:
            forces[i, 0] -= (x - (0.5 - margin)) * 10.0
        # Bottom
        if y < -0.5 + margin:
            forces[i, 1] += (margin - (y + 0.5)) * 10.0
        # Top
        if y > 0.5 - margin:
            forces[i, 1] -= (y - (0.5 - margin)) * 10.0
            
    # 3. Lloyd's attraction to Voronoi centroid
    # This helps with global distribution
    lloyd_force = (vor_centroids - centres) * 0.5
    
    total_forces = forces + lloyd_force
    
    # Normalize forces to prevent huge jumps
    norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_forces = total_forces / norms
    
    new_centres = centres + step_size * normalized_forces
    
    # Clip to boundaries
    new_centres = np.clip(new_centres, -0.5, 0.5)
    
    return new_centres


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a grid arrangement with jitter to break symmetry
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Add jitter
    centres += np.random.randn(N, 2) * 0.01
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Optimization parameters
    max_iterations = 1000
    base_step = 0.05
    
    for restart in range(2):
        if restart > 0:
            # Perturb best solution for second restart
            centres = best_centres + np.random.randn(N, 2) * 0.02
            centres = np.clip(centres, -0.5, 0.5)
            
        for it in range(max_iterations):
            # Decay step size
            step_size = base_step * (1.0 / (1.0 + 0.01 * it))
            
            try:
                tri = Delaunay(centres)
            except Exception:
                # If Delaunay fails (e.g., coincident points), jitter
                centres += np.random.randn(N, 2) * 1e-5
                continue
                
            try:
                vor_centroids = _voronoi_centroids(tri, centres)
            except Exception:
                vor_centroids = centres.copy()
                
            new_centres = _force_directed_step(centres, tri, vor_centroids, step_size)
            
            # Check if valid
            if not np.all(np.isfinite(new_centres)):
                continue
                
            new_radius = _compute_radius(new_centres)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centres = new_centres.copy()
                
            # Update current centres to new ones (greedy)
            # We might want to accept worse moves for annealing, but greedy is simpler and faster
            # To avoid getting stuck, we can occasionally revert or jitter
            if it % 100 == 0:
                # Small jitter to escape local minima
                new_centres += np.random.randn(N, 2) * 0.005
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
            centres = new_centres
            
    return best_centres
# EVOLVE-BLOCK-END
