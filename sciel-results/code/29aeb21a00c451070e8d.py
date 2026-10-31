"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Axial coordinate system for hex grid
    # A unit hexagon with "flat top" orientation:
    #   vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    #   width (flat-to-flat horizontal) = 2 * cos(30°) * 1 = √3? 
    #   Actually for pointy-top: width = √3, height = 2
    #   For flat-top: width = 2, height = √3
    # Let's use flat-top hexagons.
    # Flat-top hexagon vertices (side=1):
    #   (1, 0), (0.5, √3/2), (-0.5, √3/2), (-1, 0), (-0.5, -√3/2), (0.5, -√3/2)
    # Center-to-vertex distance = 1, center-to-edge distance = √3/2
    # Horizontal extent: -1 to 1 => width 2
    # Vertical extent: -√3/2 to √3/2 => height √3
    
    # In axial coordinates (q, r) for flat-top hexagons:
    # Cartesian: x = 1.5 * q, y = √3 * (r + q/2)
    # Neighbors: (q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)
    
    sqrt3 = np.sqrt(3.0)
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        return x, y
    
    # Generate all 11-cell polyhexes via BFS
    # Start from a single hex, grow by adding adjacent hexes
    # Use canonical form to avoid duplicates
    
    def canonical(cells):
        """Return canonical form of a set of axial coordinates."""
        cells = set(cells)
        if not cells:
            return frozenset()
        # Translate so min q, then min r is at origin
        min_q = min(q for q, r in cells)
        min_r = min(r for q, r in cells if q == min_q)
        translated = frozenset((q - min_q, r - min_r) for q, r in cells)
        return translated
    
    def get_neighbors(q, r):
        return [(q+1,r), (q-1,r), (q,r+1), (q,r-1), (q+1,r-1), (q-1,r+1)]
    
    # BFS to generate polyhexes up to size 11
    # To keep it tractable, we'll use a smarter approach:
    # Generate all connected sets of 11 cells, compute bounding hexagon
    
    # Actually, for 11 cells, the number of free polyhexes is 14,003
    # But we need to be careful about time. Let's use a different approach.
    
    # Alternative: Use the fact that the optimal arrangement is likely compact.
    # Try all "hexagonal" shapes and their variants.
    
    # A regular hexagon of side n has 3n(n-1)+1 cells.
    # For n=2: 7 cells, n=3: 19 cells. So 11 cells fits in a side-3 hexagon.
    
    # Let's enumerate connected subsets of a hexagonal grid of sufficient size.
    # Grid: all (q,r) with |q|<=4, |r|<=4, |q+r|<=4 (side-5 hexagon, 61 cells)
    # But 11 out of 61 is C(61,11) which is huge.
    
    # Better: BFS generation of polyhexes
    # We'll generate all free polyhexes of size 11 and compute their bounding box.
    
    # For efficiency, let's focus on compact shapes.
    # The minimal enclosing hexagon for a set of points can be computed.
    
    # For a set of unit hexagons at positions P_i, the enclosing hexagon must contain
    # all vertices of all hexagons. For flat-top hexagons, the six "normals" are
    # at angles 0°, 60°, 120°, 180°, 240°, 300°.
    
    # The side length of the enclosing hexagon is determined by the maximum projection
    # of the cluster onto these normals, divided by the appropriate factor.
    
    # For a hexagon with side length S and center at origin (flat-top):
    # The vertices are at (S*cos(60k°), S*sin(60k°)) for k=0..5
    # The edges are at distance S*√3/2 from center along normals at 30°, 90°, etc.
    # Wait, let me reconsider.
    
    # A regular hexagon with side length S, flat-top, centered at origin:
    # Vertices: (S, 0), (S/2, S√3/2), (-S/2, S√3/2), (-S, 0), (-S/2, -S√3/2), (S/2, -S√3/2)
    # The six edges have outward normals at angles 30°, 90°, 150°, 210°, 270°, 330°
    # Distance from center to each edge = S * √3 / 2
    
    # For a point (x,y) to be inside, we need:
    # For each normal direction θ_k = 30° + 60°*k:
    #   x*cos(θ_k) + y*sin(θ_k) <= S * √3 / 2
    
    # So S >= max over all points and all k of (x*cos(θ_k) + y*sin(θ_k)) / (√3/2)
    #    = max(...) * 2/√3
    
    # But we also need to account for the hexagon vertices, not just centers.
    # Each unit hexagon at center (cx, cy) has vertices at distance 1 from center.
    # So the effective extent in direction θ is: max over all hexagons of (center_projection + 1)
    # since the farthest vertex in direction θ is at distance 1 from center.
    
    # Wait, that's not quite right. The vertex farthest in direction θ from center is
    # at distance 1 (the circumradius). So the projection of the hexagon in direction θ
    # is center_projection + 1.
    
    # Actually, for a regular hexagon with circumradius 1, the support function in any
    # direction is 1 (since all vertices are at distance 1). No wait, the support function
    # h(θ) = max_{v in hex} v·(cos θ, sin θ). For a regular hexagon with circumradius 1,
    # h(θ) = 1 for all θ (since it's circumscribed by a unit circle). 
    # Actually no: the hexagon is inscribed in the unit circle, so h(θ) <= 1, and
    # h(θ) = 1 when θ points to a vertex.
    
    # The support function of a regular hexagon with circumradius R=1:
    # h(θ) = R * max_k cos(θ - 60°*k) for k=0..5
    # = max(cos(θ), cos(θ-60°), cos(θ-120°), cos(θ-180°), cos(θ-240°), cos(θ-300°))
    
    # For the enclosing hexagon with side length S (circumradius S), the constraint is:
    # For each normal direction φ_k = 30° + 60°*k:
    #   max over all hexagons i of [center_i · n_k + h_i(φ_k)] <= S * √3/2
    # where h_i(φ_k) is the support function of the unit hexagon in direction φ_k.
    
    # Since all inner hexagons have the same orientation (flat-top, aligned with grid),
    # h_i(φ_k) is the same for all i. Let's call it h(φ_k).
    
    # For flat-top hexagon (vertices at 0°, 60°, 120°, 180°, 240°, 300°):
    # h(φ) = max_k cos(φ - 60°*k)
    
    # For φ_k = 30° + 60°*k:
    # h(30°) = max(cos(30°), cos(-30°), cos(-90°), cos(-150°), cos(-210°), cos(-270°))
    # = max(√3/2, √3/2, 0, -√3/2, -√3/2, 0) = √3/2
    
    # Similarly for all φ_k, h(φ_k) = √3/2.
    
    # So the constraint becomes:
    # For each k: max_i (center_i · n_k) + √3/2 <= S * √3/2
    # => S >= (max_i (center_i · n_k) - √3/2) / (√3/2) = max_i(center_i · n_k) / (√3/2) - 1
    
    # Wait, let me redo this. The enclosing hexagon has side length S.
    # The distance from center to each edge is apothem = S * √3/2.
    # A point p is inside if for all k: p · n_k <= S * √3/2.
    # The hexagon i occupies the region {p : (p - c_i) · u <= h_i(u) for all u}
    # For the enclosing hexagon to contain hexagon i, we need:
    # For all k: (c_i + v) · n_k <= S * √3/2 for all v with v · u <= h_i(u) for all u
    # This is equivalent to: c_i · n_k + h_i(n_k) <= S * √3/2
    # So S >= (c_i · n_k + h_i(n_k)) / (√3/2) for all i, k
    # S >= max_{i,k} (c_i · n_k + h_i(n_k)) / (√3/2)
    
    # With h_i(n_k) = √3/2 for all i, k (as computed above):
    # S >= max_{i,k} (c_i · n_k + √3/2) / (√3/2)
    # = max_{i,k} (c_i · n_k) / (√3/2) + 1
    
    # So S = 1 + (2/√3) * max_{i,k} (c_i · n_k)
    
    # where n_k = (cos(30°+60°k), sin(30°+60°k)) for k=0..5
    
    # This gives us a fast way to compute the enclosing side length!
    
    normals = []
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        normals.append((np.cos(angle), np.sin(angle)))
    
    def compute_enclosing_side(cells):
        """Compute the minimum side length of a flat-top hexagon containing all unit hexagons at given axial positions."""
        # Convert to Cartesian centers
        centers = [axial_to_cartesian(q, r) for q, r in cells]
        max_proj = 0.0
        for cx, cy in centers:
            for nx, ny in normals:
                proj = cx * nx + cy * ny
                if proj > max_proj:
                    max_proj = proj
        # S = 1 + (2/√3) * max_proj
        # But we need to handle the case where max_proj could be negative (cluster centered away from origin)
        # Actually, we should center the cluster first.
        # The enclosing hexagon can be placed anywhere, so we should find the optimal center.
        # For a hexagon, the optimal center is such that the max projection is minimized.
        # This is equivalent to finding the minimum enclosing hexagon.
        # For simplicity, let's assume the center is at the centroid of the cluster.
        # Actually, for a regular hexagon, the minimum enclosing hexagon has its center
        # at the point that minimizes max_k (max_i c_i · n_k - min_i c_i · n_k) / 2 + ...
        # This is getting complex. Let's just use the centroid for now.
        
        # Better approach: The minimum side length is:
        # S = (1/√3) * max_k (max_i (c_i · n_k) - min_i (c_i · n_k)) + 1
        # Wait, let me think again.
        
        # The width of the cluster in direction n_k is: w_k = max_i(c_i · n_k) - min_i(c_i · n_k)
        # The enclosing hexagon needs to have width >= w_k + 2*h(n_k) in each normal direction
        # where h(n_k) = √3/2 is the support of the unit hexagon.
        # The width of a regular hexagon with side S in direction n_k is: 2 * S * √3/2 = S√3
        # Wait, the width in the normal direction n_k (which is perpendicular to an edge) is 2 * apothem = 2 * S * √3/2 = S√3.
        # So we need S√3 >= w_k + 2*(√3/2) = w_k + √3
        # => S >= (w_k + √3) / √3 = w_k/√3 + 1
        
        # So S = max_k (w_k / √3 + 1) = 1 + (1/√3) * max_k w_k
        
        max_width = 0.0
        for nx, ny in normals:
            projs = [cx * nx + cy * ny for cx, cy in centers]
            w = max(projs) - min(projs)
            if w > max_width:
                max_width = w
        return 1.0 + max_width / sqrt3
    
    # Now generate all 11-cell polyhexes and find the one with minimum enclosing side
    # Use BFS with canonical form to avoid duplicates
    
    best_side = float('inf')
    best_cells = None
    
    # Start with all 1-cell polyhexes
    polyhexes_by_size = {1: {frozenset([(0,0)])}}
    
    for size in range(1, N):
        if time.time() > deadline:
            break
        next_polyhexes = set()
        for cells in polyhexes_by_size[size]:
            # Get all cells
            cell_list = list(cells)
            # For each cell, try adding each neighbor
            for q, r in cell_list:
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) not in cells:
                        new_cells = cells | {(nq, nr)}
                        canon = canonical(new_cells)
                        if canon not in next_polyhexes:
                            next_polyhexes.add(canon)
        polyhexes_by_size[size + 1] = next_polyhexes
        if size + 1 == N:
            # Evaluate all 11-cell polyhexes
            for cells in next_polyhexes:
                if time.time() > deadline:
                    break
                side = compute_enclosing_side(cells)
                if side < best_side:
                    best_side = side
                    best_cells = cells
    
    if best_cells is None:
        # Fallback: simple linear arrangement
        best_cells = frozenset([(k, 0) for k in range(N)])
        best_side = compute_enclosing_side(best_cells)
    
    # Convert best_cells to Cartesian positions
    inner = []
    for q, r in best_cells:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])
    
    # The outer hexagon center should be at the centroid of the cluster for optimal fit
    # But since we're using the width-based formula, the center doesn't affect the side length
    # Let's set the outer center to (0,0) for simplicity, or to the centroid
    if best_cells:
        cx = sum(axial_to_cartesian(q, r)[0] for q, r in best_cells) / len(best_cells)
        cy = sum(axial_to_cartesian(q, r)[1] for q, r in best_cells) / len(best_cells)
    else:
        cx, cy = 0.0, 0.0
    
    return inner, (cx, cy), best_side, 0.0
# EVOLVE-BLOCK-END

