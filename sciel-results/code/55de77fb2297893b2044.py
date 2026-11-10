# EVOLVE-BLOCK-START
import time
import numpy as np

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))

def _axial_to_cartesian(q, r):
    x = 1.5 * q
    y = np.sqrt(3.0) * (r + q / 2.0)
    return np.array([x, y])

def _outer_side_for_points(points, phi_deg):
    """Minimum side length of a regular hexagon (centered at origin, rotation phi_deg) enclosing points."""
    phi_rad = np.deg2rad(phi_deg)
    normals = np.column_stack((
        np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)),
        np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))
    ))
    proj = points @ normals.T  # (N, 6)
    widths = proj.max(axis=0) - proj.min(axis=0)  # (6,)
    return widths.max() / np.sqrt(3.0)

def _outer_center(points, phi_deg, side):
    """Find center (ox, oy) of outer hexagon with given side and phi that encloses points."""
    phi_rad = np.deg2rad(phi_deg)
    normals = np.column_stack((
        np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)),
        np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))
    ))
    proj = points @ normals.T  # (N, 6)
    # For each normal direction k, we need:
    #   -side*sqrt(3)/2 <= n_k . (p - c) <= side*sqrt(3)/2
    # => n_k . c is in [min_p(n_k.p) + side*sqrt(3)/2, max_p(n_k.p) - side*sqrt(3)/2]
    # We want c that satisfies all 6 constraints. Pick midpoint of feasible region.
    # c projected onto each normal must be in [lo_k, hi_k]
    # c = sum_k alpha_k * n_k where we solve for c directly.
    # Actually, let's just use the centroid of the feasible region in 2D.
    # The feasible region for c is a hexagon (intersection of 6 strips).
    # A simple valid center: average of the midpoints of each strip projected back.
    # More robust: solve 2x2 least squares from the 6 constraints.
    # n_k . c = mid_k, where mid_k = (lo_k + hi_k) / 2
    lo = proj.min(axis=0) + side * np.sqrt(3.0) / 2.0
    hi = proj.max(axis=0) - side * np.sqrt(3.0) / 2.0
    mids = (lo + hi) / 2.0  # (6,)
    # Solve normals @ c = mids in least squares sense
    # normals is (6,2), c is (2,)
    c, _, _, _ = np.linalg.lstsq(normals, mids, rcond=None)
    return c

def _generate_clusters():
    """Generate all connected subsets of size 11 from hexagonal lattice positions."""
    # Generate lattice positions in a generous bounding box
    positions = {}
    coords = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            pos = _axial_to_cartesian(q, r)
            key = (q, r)
            positions[key] = pos
            coords.append(key)
    
    # Adjacency in axial coordinates
    def neighbors(q, r):
        return [(q+1, r), (q-1, r), (q, r+1), (q, r-1), (q+1, r-1), (q-1, r+1)]
    
    # Generate all connected subsets of size 11 using BFS from each starting point
    # To avoid duplicates, use canonical form (lexicographic min of sorted coords)
    seen = set()
    clusters = []
    
    def canonical(subset):
        return tuple(sorted(subset))
    
    def expand(current, visited_set, remaining):
        if remaining == 0:
            c = canonical(current)
            if c not in seen:
                seen.add(c)
                clusters.append(list(c))
            return
        # Try to add each neighbor not yet in current
        for cell in list(current):
            for nb in neighbors(cell[0], cell[1]):
                if nb not in visited_set:
                    current.append(nb)
                    visited_set.add(nb)
                    expand(current, visited_set, remaining - 1)
                    current.pop()
                    visited_set.discard(nb)
    
    # Start from each cell, but use symmetry to reduce
    # For efficiency, start from cells where the cell is the "minimum" in canonical ordering
    # This is complex; let's just do a bounded search
    # Actually, the number of free hexominoes of size 11 is ~4000, so this is manageable
    # but BFS over all starting points with size 11 could be large.
    # Let's limit: start from a subset of positions and use a time limit.
    
    # Simpler approach: generate all size-11 connected sets from a restricted region
    # and deduplicate. Use iterative deepening with a cap.
    
    # Actually, let's use a smarter approach: generate all connected subsets of size 11
    # from the lattice, but cap the total number generated.
    
    max_clusters = 5000
    t_start = time.time()
    
    for start in coords:
        if time.time() - t_start > 30.0:
            break
        if len(clusters) >= max_clusters:
            break
        current = [start]
        visited = {start}
        # Only expand in a way that maintains connectivity
        # Use the constraint that we only add cells adjacent to existing ones
        expand(current, visited, 10)
    
    return clusters

def _cluster_vertices(cluster_coords):
    """Get all vertices of all hexagons in a cluster (for enclosing hex computation)."""
    all_verts = []
    for (q, r) in cluster_coords:
        cx, cy = _axial_to_cartesian(q, r)
        # Center the cluster first
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _cluster_centers(cluster_coords):
    """Get centers of all hexagons in a cluster."""
    return np.array([_axial_to_cartesian(q, r) for (q, r) in cluster_coords])

def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 580.0  # Safety margin
    
    # Generate candidate clusters
    clusters = _generate_clusters()
    
    if len(clusters) == 0:
        # Fallback: simple hexagonal arrangement
        base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (1,1), (2,0), (-1,2), (0,2)]
        clusters = [base]
    
    best_side = np.inf
    best_cluster = None
    best_phi = 0.0
    
    # For each cluster, find the optimal outer hexagon rotation
    for cluster in clusters:
        if time.time() > deadline - 5.0:
            break
        
        centers = _cluster_centers(cluster)
        # Center the cluster at origin
        centroid = centers.mean(axis=0)
        centers -= centroid
        all_verts = _cluster_vertices(cluster) - centroid
        
        # Optimize outer hexagon rotation: minimize max width / sqrt(3)
        # The width in direction n_k is max(proj) - min(proj)
        # We want to minimize max_k width_k
        # This is a 1D problem in phi
        # Use golden section search over [0, 60) degrees (periodicity)
        
        def width_at_phi(phi_deg):
            phi_rad = np.deg2rad(phi_deg)
            normals = np.column_stack((
                np.cos(phi_rad + np.deg2rad(60.0) * np.arange(6)),
                np.sin(phi_rad + np.deg2rad(60.0) * np.arange(6))
            ))
            proj = all_verts @ normals.T
            widths = proj.max(axis=0) - proj.min(axis=0)
            return widths.max()
        
        # Golden section search on [0, 60)
        gr = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
        a, b = 0.0, 60.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = width_at_phi(c)
        fd = width_at_phi(d)
        
        for _ in range(80):
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = width_at_phi(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = width_at_phi(d)
        
        phi_opt = (a + b) / 2.0
        side_opt = width_at_phi(phi_opt) / np.sqrt(3.0)
        
        if side_opt < best_side:
            best_side = side_opt
            best_cluster = cluster
            best_phi = phi_opt
    
    if best_cluster is None:
        # Ultimate fallback
        base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                (1,1), (2,0), (-1,2), (0,2)]
        best_cluster = base
        best_phi = 0.0
        centers = _cluster_centers(base)
        centroid = centers.mean(axis=0)
        centers -= centroid
        all_verts = _cluster_vertices(base) - centroid
        best_side = _outer_side_for_points(centers, 0.0)
    
    # Finalize
    centers = _cluster_centers(best_cluster)
    centroid = centers.mean(axis=0)
    centers -= centroid
    
    # Find outer center
    outer_c = _outer_center(centers, best_phi, best_side)
    
    # Build output
    inner = np.column_stack((centers[:, 0], centers[:, 1], np.full(11, 0.0)))
    outer = np.array([outer_c[0], outer_c[1], best_phi])
    side = float(best_side)
    
    return inner, outer, side

# EVOLVE-BLOCK-END
