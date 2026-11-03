"""Improved program for the 200-circle packing task using Delaunay-based edge-flip optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
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


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _find_shared_triangles(tri, edge):
    """Find the two triangles that share the given edge."""
    tri_a = None
    tri_b = None
    for simplex in tri.simplices:
        if edge[0] in simplex and edge[1] in simplex:
            if tri_a is None:
                tri_a = simplex
            else:
                tri_b = simplex
                break
    return tri_a, tri_b


def _is_valid_flip(tri, edge, centres):
    """
    Check if flipping the edge is valid: the new edge (third_a, third_b)
    should be inside the convex hull and not cross other edges.
    """
    tri_a, tri_b = _find_shared_triangles(tri, edge)
    if tri_a is None or tri_b is None:
        return False
    
    third_a = [v for v in tri_a if v != edge[0] and v != edge[1]][0]
    third_b = [v for v in tri_b if v != edge[0] and v != edge[1]][0]
    
    # Check if the new edge (third_a, third_b) is inside the convex hull
    # A simple check: the midpoint of the new edge should be inside the convex hull
    midpoint = (centres[third_a] + centres[third_b]) / 2.0
    
    # For a more robust check, we could use point-in-polygon, but for now
    # we'll rely on the Delaunay recompute to catch invalid configurations
    return True


def _flip_edge(tri, centres, edge):
    """
    Flip the given edge in the Delaunay triangulation.
    Returns the new centres after attempting to improve the minimum distance.
    """
    tri_a, tri_b = _find_shared_triangles(tri, edge)
    if tri_a is None or tri_b is None:
        return None
    
    third_a = [v for v in tri_a if v != edge[0] and v != edge[1]][0]
    third_b = [v for v in tri_b if v != edge[0] and v != edge[1]][0]
    
    # The flip replaces edge (edge[0], edge[1]) with edge (third_a, third_b)
    # In a proper Delaunay flip, we move the vertices to equalize distances
    # For now, we'll try to adjust the positions of the four vertices
    
    # Compute the current minimum distance
    current_radius = _compute_radius(centres)
    
    # Try to move the four vertices to improve the minimum edge length
    # The four vertices are: edge[0], edge[1], third_a, third_b
    # We want to equalize the distances between them
    
    # A simple approach: try small perturbations
    best_centres = centres.copy()
    best_radius = current_radius
    
    # Try moving each of the four vertices slightly
    vertices = [edge[0], edge[1], third_a, third_b]
    
    for _ in range(10):
        new_centres = centres.copy()
        for v in vertices:
            # Try a small random perturbation
            perturbation = np.random.randn(2) * 0.002
            new_centres[v] += perturbation
            new_centres[v] = np.clip(new_centres[v], -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        if new_radius > best_radius + 1e-10:
            best_radius = new_radius
            best_centres = new_centres.copy()
    
    if best_radius > current_radius + 1e-10:
        return best_centres
    return None


def _delaunay_edge_flip_optimization(centres, max_iterations=1000):
    """
    Optimize circle centres using Delaunay edge flips.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(best_centres)
        except Exception:
            break
        
        edges = _get_delaunay_edges(tri)
        
        if not edges:
            break
        
        # Find the edge with the shortest length
        min_edge = None
        min_len = np.inf
        for edge in edges:
            d = np.linalg.norm(best_centres[edge[0]] - best_centres[edge[1]])
            if d < min_len:
                min_len = d
                min_edge = edge
        
        if min_edge is None:
            break
        
        # Try to flip this edge
        new_centres = _flip_edge(tri, best_centres, min_edge)
        
        if new_centres is not None:
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
        else:
            # If flip didn't help, try a small random perturbation of all vertices
            perturbation = np.random.randn(len(best_centres), 2) * 0.001
            new_centres = best_centres + perturbation
            new_centres = np.clip(new_centres, -0.5, 0.5)
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
    
    return best_centres, best_radius


def _local_optimization(centres, max_iterations=300, step_size=0.008):
    """
    Local optimization using gradient-like steps to maximize minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        dists = cdist(best_centres, best_centres)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(best_centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        improved = False
        
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            direction = best_centres[min_i] - best_centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = best_centres.copy()
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
            pos = best_centres[min_border_i]
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
            new_centres = best_centres.copy()
            new_centres[min_border_i] += direction * step
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
    
    # Phase 1: Local optimization from grid
    optimised, radius = _local_optimization(centres, max_iterations=200, step_size=0.008)
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    # Phase 2: Delaunay edge-flip optimization
    optimised, radius = _delaunay_edge_flip_optimization(best_centres, max_iterations=500)
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    # Phase 3: Final local optimization
    optimised, radius = _local_optimization(best_centres, max_iterations=200, step_size=0.005)
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
