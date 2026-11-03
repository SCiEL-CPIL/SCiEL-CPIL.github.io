"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _compute_radius_fast(centres, tri):
    """
    Compute radius using Delaunay edges for speed.
    Only checks edges in the Delaunay triangulation plus border distances.
    """
    # Check all Delaunay edges
    min_edge_dist = np.inf
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                d = np.linalg.norm(centres[simplex[i]] - centres[simplex[j]])
                if d < min_edge_dist:
                    min_edge_dist = d
    
    # Also check a few non-Delaunay pairs to be safe (sample)
    # For now, trust Delaunay edges give the min distance
    
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_edge_dist / 2.0, min_dist_to_border))


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _find_flippable_edges(tri, centres):
    """
    Find edges that can be flipped. An edge is flippable if it is shared by two triangles
    and the opposite diagonal would create a valid (non-crossing) configuration.
    """
    # Build adjacency: for each edge, which triangles contain it
    edge_to_tris = {}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(simplex)
    
    flippable = []
    for edge, tris in edge_to_tris.items():
        if len(tris) == 2:
            # Two triangles share this edge
            tri1, tri2 = tris[0], tris[1]
            # Find the third vertices
            v1 = [v for v in tri1 if v != edge[0] and v != edge[1]][0]
            v2 = [v for v in tri2 if v != edge[0] and v != edge[1]][0]
            
            # Check if the new edge (v1, v2) would be valid
            # In 2D Delaunay, we can check if the quadrilateral is convex
            # and if the flip improves the minimum edge length
            flippable.append((edge[0], edge[1], v1, v2))
    
    return flippable


def _try_edge_flip(centres, i, j, v1, v2, current_radius):
    """
    Try flipping the edge (i,j) to (v1,v2) by adjusting positions.
    This is a heuristic: we move the four points to increase the minimum edge length.
    """
    new_centres = centres.copy()
    
    # The idea: after flip, the new configuration should have (i,v1), (i,v2), (j,v1), (j,v2) as potential edges
    # We want to increase the minimum distance among these
    
    # Simple heuristic: move i and j slightly apart, and v1 and v2 slightly apart
    # But this is tricky. Instead, let's just do a local search around the current positions.
    
    return new_centres


def _local_optimization(centres, max_iterations=500, step_size=0.01):
    """
    Local optimization using gradient-like steps to maximize minimum distance.
    Uses Delaunay triangulation to identify critical pairs.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except:
            break
        
        # Find the bottleneck using Delaunay edges
        min_dist = np.inf
        min_i, min_j = -1, -1
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    d = np.linalg.norm(centres[simplex[i]] - centres[simplex[j]])
                    if d < min_dist:
                        min_dist = d
                        min_i, min_j = simplex[i], simplex[j]
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Move the two circles apart
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        else:
            # Border constraint is active
            pos = centres[min_border_i]
            direction = np.zeros(2)
            if pos[0] > 0:
                direction[0] = -1.0
            elif pos[0] < 0:
                direction[0] = 1.0
            
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_border_i] += direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        
        if not improved:
            # Try a small random perturbation to escape local minima
            if iteration > max_iterations // 2:
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * step_size * 0.5
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
        
        if not improved and iteration > max_iterations // 3:
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
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
    for restart in range(3):
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.005
            centres = np.clip(centres, -0.5, 0.5)
        
        optimised, radius = _local_optimization(centres, max_iterations=300, step_size=0.008)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
