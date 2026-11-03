"""Structural change: discrete topological search over hex-lattice subgraphs + LP for min enclosing hexagon."""
import math
import time
import numpy as np
import itertools
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # 5s safety margin within 120s limit


def _compute_all_vertices(inner_hexes):
    """Precompute all vertices of inner hexagons as a numpy array (N*6, 2)."""
    n_inner = len(inner_hexes)
    verts = np.zeros((n_inner * 6, 2))
    for idx, (ix, iy, iangle) in enumerate(inner_hexes):
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + math.pi * k / 3
            verts[idx * 6 + k, 0] = ix + math.cos(a)
            verts[idx * 6 + k, 1] = iy + math.sin(a)
    return verts


def _min_enclosing_hex_side_lp(inner_hexes, angle_deg, all_vertices=None):
    """For a fixed outer hexagon angle, find optimal center and side length using LP."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    n_pts = all_vertices.shape[0]
    if n_pts == 0:
        return 0.0, 0.0, 0.0, angle_deg
    
    # For each of 6 directions, compute normals
    angles = np.array([math.radians(angle_deg + 30 + 60 * k) for k in range(6)])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6, 2)
    
    # Vectorized constraint construction
    rhs = -(all_vertices @ normals.T)  # (n_pts, 6)
    b_ub = rhs.T.flatten()  # (6 * n_pts,)
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], n_pts),
        np.repeat(-normals[:, 1], n_pts),
        np.full(6 * n_pts, -1.0)
    ])
    
    # Add r >= 0 constraint
    A_ub = np.vstack([A_ub_parts, [[0.0, 0.0, -1.0]]])
    b_ub = np.append(b_ub, 0.0)
    
    c = np.array([0.0, 0.0, 1.0])  # minimize r
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
    
    if result.success:
        cx_opt = result.x[0]
        cy_opt = result.x[1]
        r_opt = result.x[2]
    else:
        cx_mean = all_vertices[:, 0].mean()
        cy_mean = all_vertices[:, 1].mean()
        centered = all_vertices - np.array([cx_mean, cy_mean])
        projs = centered @ normals.T
        r_opt = projs.max()
        if r_opt < 0:
            r_opt = 0.0
        cx_opt = cx_mean
        cy_opt = cy_mean
    
    side = r_opt / (math.sqrt(3) / 2)
    return cx_opt, cy_opt, side, angle_deg


def _find_best_outer_for_inner(inner_hexes, all_vertices=None):
    """Find best outer hexagon (center, side, angle) for given inner hexagons."""
    if all_vertices is None:
        all_vertices = _compute_all_vertices(inner_hexes)
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse sweep: 0 to 60 degrees in steps of 3
    for angle_deg in range(0, 60, 3):
        cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_angle = float(angle_deg)
    
    # Fine sweep around best
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 5), min(60.0, best_angle + 5), 0.5):
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    # Very fine sweep
    if best_side < float('inf'):
        for angle_deg in np.arange(max(0.0, best_angle - 1.0), min(60.0, best_angle + 1.0), 0.1):
            cx, cy, side, ang = _min_enclosing_hex_side_lp(inner_hexes, float(angle_deg), all_vertices)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)
    
    return best_center[0], best_center[1], best_side, best_angle


def _generate_hex_lattice_positions(max_radius=4):
    """Generate all hex lattice positions within a given radius from origin."""
    # Hex lattice: positions are (i*a1 + j*a2) where a1, a2 are basis vectors
    # For unit hexagons with edge-to-edge contact, the center-to-center distance is sqrt(3)
    sqrt3 = math.sqrt(3)
    # Basis vectors for hex lattice
    a1 = (sqrt3, 0.0)
    a2 = (sqrt3 / 2, 1.5)
    
    positions = []
    for i in range(-max_radius, max_radius + 1):
        for j in range(-max_radius, max_radius + 1):
            x = i * a1[0] + j * a2[0]
            y = i * a1[1] + j * a2[1]
            # Compute hexagonal distance (number of steps to reach)
            # For hex lattice, distance = max(|i|, |j|, |i+j|) approximately
            # More precisely, use axial coordinates
            dist = max(abs(i), abs(j), abs(i + j))
            if dist <= max_radius:
                positions.append((i, j, x, y, dist))
    
    return positions


def _get_valid_subsets(positions, n_select=11, max_dist=3):
    """Generate all valid subsets of n_select positions from the lattice."""
    # Only consider positions within max_dist from origin
    filtered = [(i, j, x, y) for (i, j, x, y, d) in positions if d <= max_dist]
    
    # If we have exactly n_select or fewer, return all
    if len(filtered) <= n_select:
        return [tuple(filtered)]
    
    # Generate combinations
    return list(itertools.combinations(filtered, n_select))


def _compute_cluster_boundaries(cluster_positions):
    """Compute the bounding characteristics of a cluster of hex centers."""
    if not cluster_positions:
        return 0.0, 0.0, 0.0
    
    xs = [p[2] for p in cluster_positions]
    ys = [p[3] for p in cluster_positions]
    
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    # Estimate required side length based on extent
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    # For a hexagon, the side length is related to the extent in various directions
    # This is a rough estimate
    est_side = max(width, height) / (math.sqrt(3) * 2) + 1.0  # +1 for hex radius
    
    return cx, cy, est_side


def construct():
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Generate hex lattice positions
    all_positions = _generate_hex_lattice_positions(max_radius=4)
    
    # Filter to positions within reasonable distance
    filtered_positions = [(i, j, x, y) for (i, j, x, y, d) in all_positions if d <= 3]
    
    # Generate candidate subsets
    # We want to explore different topologies
    candidate_subsets = []
    
    # Strategy 1: The classic "flower" pattern (center + 6 neighbors + 4 from second ring)
    # Center is at (0,0)
    center = (0, 0, 0.0, 0.0)
    
    # First ring: 6 neighbors
    first_ring = []
    for (i, j, x, y) in filtered_positions:
        if max(abs(i), abs(j), abs(i + j)) == 1:
            first_ring.append((i, j, x, y))
    
    # Second ring: positions at distance 2
    second_ring = []
    for (i, j, x, y) in filtered_positions:
        if max(abs(i), abs(j), abs(i + j)) == 2:
            second_ring.append((i, j, x, y))
    
    # Classic pattern: center + all 6 first ring + 4 from second ring
    if len(first_ring) == 6:
        for combo in itertools.combinations(second_ring, 4):
            subset = [center] + first_ring + list(combo)
            candidate_subsets.append(subset)
    
    # Strategy 2: Other patterns - try different selections from second ring
    # Also try patterns where we don't use all of first ring
    # Pattern: center + 5 from first ring + 5 from second ring
    for combo1 in itertools.combinations(first_ring, 5):
        for combo2 in itertools.combinations(second_ring, 5):
            subset = [center] + list(combo1) + list(combo2)
            candidate_subsets.append(subset)
    
    # Pattern: center + 4 from first ring + 6 from second ring
    for combo1 in itertools.combinations(first_ring, 4):
        for combo2 in itertools.combinations(second_ring, 6):
            subset = [center] + list(combo1) + list(combo2)
            candidate_subsets.append(subset)
    
    # Strategy 3: No center - 6 from first ring + 5 from second ring
    for combo1 in itertools.combinations(first_ring, 6):
        for combo2 in itertools.combinations(second_ring, 5):
            subset = list(combo1) + list(combo2)
            candidate_subsets.append(subset)
    
    # Strategy 4: 5 from first ring + 6 from second ring (no center)
    for combo1 in itertools.combinations(first_ring, 5):
        for combo2 in itertools.combinations(second_ring, 6):
            subset = list(combo1) + list(combo2)
            candidate_subsets.append(subset)
    
    # Deduplicate subsets (by set of (i,j) pairs)
    unique_subsets = []
    seen = set()
    for subset in candidate_subsets:
        key = tuple(sorted((p[0], p[1]) for p in subset))
        if key not in seen:
            seen.add(key)
            unique_subsets.append(subset)
    
    candidate_subsets = unique_subsets
    
    # Evaluate each candidate subset
    best_side = float('inf')
    best_inner = None
    best_outer = None
    best_subset = None
    
    # Sort candidates by estimated side length for early pruning
    def estimate_side(subset):
        cx, cy, est = _compute_cluster_boundaries(subset)
        return est
    
    candidate_subsets.sort(key=estimate_side)
    
    # Evaluate candidates, stopping if deadline approaches
    for subset in candidate_subsets:
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Convert subset to inner hexagons
        inner_hexes = [[p[2], p[3], 0.0] for p in subset]
        
        # Find best outer hexagon
        cx, cy, side, angle = _find_best_outer_for_inner(inner_hexes)
        
        if side < best_side:
            best_side = side
            best_inner = inner_hexes
            best_outer = (cx, cy, angle)
            best_subset = subset
    
    # If no valid subset found, use fallback
    if best_inner is None:
        # Use the classic flower pattern
        if len(first_ring) == 6 and len(second_ring) >= 4:
            subset = [center] + first_ring + second_ring[:4]
            best_inner = [[p[2], p[3], 0.0] for p in subset]
            cx, cy, best_side, angle = _find_best_outer_for_inner(best_inner)
            best_outer = (cx, cy, angle)
        else:
            # Absolute fallback
            best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            best_outer = (0.0, 0.0, 0.0)
            best_side = 40.0
    
    cx, cy, angle = best_outer
    return best_inner, (cx, cy), best_side, angle
