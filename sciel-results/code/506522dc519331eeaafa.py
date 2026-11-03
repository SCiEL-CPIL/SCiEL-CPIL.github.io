"""Delaunay edge-flip approach for 200-circle packing."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist


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


def _get_edges(tri):
    """Extract unique edges from Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _is_valid_flip(tri, centres, edge, third_a, third_b):
    """Check if flipping edge (edge_i, edge_j) to (third_a, third_b) is valid."""
    # The new edge must not cross any existing edges
    # and the new triangles must be valid (non-degenerate)
    
    # Check if the new edge would create a degenerate triangle
    # by checking if the four points form a convex quadrilateral
    
    a, b = edge
    c, d = third_a, third_b
    
    # Get the four points
    p1, p2, p3, p4 = centres[a], centres[b], centres[c], centres[d]
    
    # Check if the quadrilateral is convex by checking cross products
    # of consecutive edges
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Order the four points and check convexity
    # For simplicity, check if the new edge (c,d) is inside the convex hull
    # of the four points and doesn't cross existing edges
    
    # Simple check: ensure the new triangles have positive area
    def triangle_area(p1, p2, p3):
        return 0.5 * abs(cross(p1, p2, p3))
    
    area1 = triangle_area(centres[a], centres[c], centres[d])
    area2 = triangle_area(centres[b], centres[c], centres[d])
    
    if area1 < 1e-10 or area2 < 1e-10:
        return False
    
    # Check if the new edge crosses any existing edges
    new_edge = (min(c, d), max(c, d))
    existing_edges = _get_edges(tri)
    
    for e in existing_edges:
        if e == edge:
            continue
        # Check if new_edge crosses e
        if _segments_intersect(centres, new_edge, e):
            return False
    
    return True


def _segments_intersect(centres, edge1, edge2):
    """Check if two segments intersect (excluding shared endpoints)."""
    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
            return True
        return False
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val > 0:
            return 1
        if val < 0:
            return 2
        return 0
    
    p1, p2 = centres[edge1[0]], centres[edge1[1]]
    p3, p4 = centres[edge2[0]], centres[edge2[1]]
    
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    return False


def _flip_edge(tri, centres, edge, third_a, third_b):
    """
    Flip an edge in the Delaunay triangulation.
    Returns new centres if the flip improves the minimum distance.
    """
    a, b = edge
    c, d = third_a, third_b
    
    # Compute current minimum distance
    current_min = _compute_radius(centres)
    
    # Try moving points a and b apart, and c and d apart
    # to increase the minimum edge length
    
    # Direction from a to b
    dir_ab = centres[b] - centres[a]
    norm_ab = np.linalg.norm(dir_ab)
    if norm_ab > 1e-10:
        dir_ab = dir_ab / norm_ab
    else:
        return None
    
    # Direction from c to d
    dir_cd = centres[d] - centres[c]
    norm_cd = np.linalg.norm(dir_cd)
    if norm_cd > 1e-10:
        dir_cd = dir_cd / norm_cd
    else:
        return None
    
    # Try small moves to increase distances
    best_centres = centres.copy()
    best_min = current_min
    
    for step in [0.001, 0.002, 0.005, 0.01]:
        # Move a and b apart
        new_centres = centres.copy()
        new_centres[a] -= dir_ab * step * 0.5
        new_centres[b] += dir_ab * step * 0.5
        
        # Move c and d apart
        new_centres[c] -= dir_cd * step * 0.5
        new_centres[d] += dir_cd * step * 0.5
        
        # Keep within bounds
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_min = _compute_radius(new_centres)
        if new_min > best_min:
            best_min = new_min
            best_centres = new_centres.copy()
    
    return best_centres


def _find_flippable_edges(tri, centres):
    """Find all flippable edges in the Delaunay triangulation."""
    flippable = []
    
    # Build adjacency: for each edge, find the two triangles that share it
    edge_to_tris = {}
    for idx, simplex in enumerate(tri.simplices):
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(idx)
    
    for edge, tris in edge_to_tris.items():
        if len(tris) == 2:
            # This edge is shared by exactly two triangles
            tri1 = tri.simplices[tris[0]]
            tri2 = tri.simplices[tris[1]]
            
            # Find the third vertices
            third1 = [v for v in tri1 if v not in edge]
            third2 = [v for v in tri2 if v not in edge]
            
            if len(third1) == 1 and len(third2) == 1:
                c, d = third1[0], third2[0]
                
                # Check if the flip is valid
                if _is_valid_flip(tri, centres, edge, c, d):
                    flippable.append((edge, c, d))
    
    return flippable


def _delaunay_optimization(centres, max_rounds=100):
    """
    Optimize using Delaunay edge-flip operations.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for round_num in range(max_rounds):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        # Find flippable edges
        flippable = _find_flippable_edges(tri, centres)
        
        if not flippable:
            break
        
        # Try each flippable edge and keep the best improvement
        best_flip_centres = None
        best_flip_radius = best_radius
        
        for edge, c, d in flippable:
            new_centres = _flip_edge(tri, centres, edge, c, d)
            if new_centres is not None:
                new_radius = _compute_radius(new_centres)
                if new_radius > best_flip_radius:
                    best_flip_radius = new_radius
                    best_flip_centres = new_centres
        
        if best_flip_centres is not None and best_flip_radius > best_radius + 1e-10:
            best_radius = best_flip_radius
            best_centres = best_flip_centres.copy()
        else:
            # No improvement, try a small random perturbation
            if round_num > max_rounds // 2:
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * 0.002
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
            else:
                break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a hexagonal-like arrangement
    # Use a grid with slight offset for better packing
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    centres += np.random.randn(N, 2) * 0.001
    centres = np.clip(centres, -0.5, 0.5)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run Delaunay optimization
    optimised, radius = _delaunay_optimization(centres, max_rounds=50)
    
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    # Try a few random restarts
    for restart in range(2):
        restart_centres = best_centres + np.random.randn(N, 2) * 0.003
        restart_centres = np.clip(restart_centres, -0.5, 0.5)
        
        optimised, radius = _delaunay_optimization(restart_centres, max_rounds=30)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
