"""Improved program: discrete grid search for hexagon packing."""
import time
import math
import numpy as np
from itertools import product

N = 11  # Number of inner hexagons

# Deadline guard
_DEADLINE = 150.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _check_deadline():
    return _elapsed() > _DEADLINE - 5.0  # 5s safety margin

def construct():
    """
    Place N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each unit hexagon
      center: (x, y) outer hexagon center
      side: outer hexagon side length
      angle: outer hexagon rotation in degrees
    """
    start_time = time.time()
    
    # For a regular hexagon with side length s, the distance from center to vertex is s.
    # The distance from center to edge (apothem) is s * sqrt(3)/2.
    # 
    # A unit hexagon (side=1) has:
    #   - circumradius (center to vertex) = 1
    #   - inradius (center to edge) = sqrt(3)/2
    #
    # For two unit hexagons to not overlap, their centers must be at least 
    # 2 * inradius = sqrt(3) apart IF they are aligned the same way.
    # But with rotation, the minimum center distance for non-overlap depends on relative orientation.
    #
    # Simplest approach: use a hexagonal lattice of positions where all inner hexagons
    # have the same orientation (0 degrees), and place them on a triangular/hex lattice.
    # The minimum center-to-center distance for non-overlapping same-oriented unit hexagons
    # is sqrt(3) (distance between parallel edges touching).
    #
    # Actually, for two identical regular hexagons with the same orientation, the minimum
    # center distance for non-overlap is 2 * (sqrt(3)/2) = sqrt(3) when they share a face,
    # but more generally it's the minimum over all directions of the support function sum.
    # For same-oriented regular hexagons, the minimum center distance is sqrt(3) 
    # (when one is directly "above" the other sharing an edge).
    #
    # Let's use a hexagonal close-packed arrangement.
    # In a hex lattice with spacing d = sqrt(3), we can pack hexagons.
    #
    # The key insight: if we place N hexagons in a compact hexagonal cluster on a 
    # hex lattice with spacing sqrt(3), all with the same orientation, we need to find
    # the smallest enclosing regular hexagon (with any rotation) that contains all of them.
    
    # Hex lattice basis vectors with spacing sqrt(3):
    d = math.sqrt(3.0)
    # Basis: (d, 0) and (d/2, d*sqrt(3)/2) = (d/2, 3/2)
    # Since d = sqrt(3), d*sqrt(3)/2 = 3/2
    ax = np.array([d, 0.0])
    ay = np.array([d/2.0, 1.5])  # d*sqrt(3)/2 = sqrt(3)*sqrt(3)/2 = 3/2
    
    # Generate candidate lattice positions in a hexagonal cluster
    # For N hexagons, we want a compact arrangement.
    # A hexagonal number arrangement: 1, 7, 19, 37, ...
    # For N=11, we can use a 2-row arrangement or similar.
    
    # Let's generate all lattice points within a certain radius and pick the N closest
    # to the origin (or to a cluster center) that form a compact shape.
    
    # Generate lattice points
    max_ring = 3
    points = []
    for i in range(-max_ring, max_ring + 1):
        for j in range(-max_ring, max_ring + 1):
            # Hex lattice coordinate to Cartesian
            x = i * ax[0] + j * ay[0]
            y = i * ax[1] + j * ay[1]
            points.append((x, y, i, j))
    
    # For each candidate cluster of N points, compute the minimum enclosing hexagon side length
    # We'll try different subsets of N points from the lattice.
    
    # First, sort points by distance from origin to prefer central clusters
    points_sorted = sorted(points, key=lambda p: p[0]**2 + p[1]**2)
    
    # Try different strategies for selecting N points
    # Strategy 1: Take the N closest to origin
    # Strategy 2: Try various offsets
    
    best_side = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    def enclosing_hexagon_side(pts, hex_angle_deg=0.0):
        """
        Compute the minimum side length of a regular hexagon (with given orientation)
        centered at the centroid of pts that encloses all pts.
        Actually, we need to find the minimum side length over all centers and the given orientation.
        For a fixed orientation, the minimum enclosing regular hexagon problem can be solved.
        
        For a regular hexagon with side s and orientation theta, centered at (cx, cy),
        a point (x, y) is inside if for all 6 edge directions, the projection is within bounds.
        
        The 6 outward normals of a regular hexagon with orientation theta are at angles:
        theta, theta+60, theta+120, theta+180, theta+240, theta+300
        The apothem is s * sqrt(3)/2.
        
        A point is inside if for each normal direction n_k, the dot product of (point - center)
        with n_k is <= apothem.
        
        For a fixed orientation, to minimize s, we need to minimize the maximum over all points
        and all 6 directions of |dot(point - center, n_k)| / (sqrt(3)/2).
        
        The optimal center for a fixed orientation minimizes the max of these 6 quantities.
        This is a convex optimization problem (minimize t subject to constraints).
        """
        theta = math.radians(hex_angle_deg)
        normals = []
        for k in range(6):
            ang = theta + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Convert pts to numpy array
        pts_arr = np.array([(p[0], p[1]) for p in pts])
        
        # For each normal, compute the range of projections
        # The minimum side length for this orientation is determined by:
        # For each pair of opposite normals, the width in that direction must be <= 2 * apothem = s*sqrt(3)
        # So s >= width_k / sqrt(3) for each k, where width_k is the projection range in direction k.
        # But we can also shift the center, so the width is the range of projections.
        
        # Actually, for a regular hexagon, the constraint is:
        # For each of the 3 pairs of opposite edges, the distance between the two parallel edges
        # is 2 * apothem = s * sqrt(3).
        # The projection of all points onto each normal must fit within [-apothem, apothem]
        # relative to the center.
        
        # So for each normal direction n_k, let proj_k = dot(pts, n_k).
        # The range is max(proj_k) - min(proj_k).
        # We need range_k <= s * sqrt(3) for all 3 independent directions (k=0,1,2 since 3,4,5 are opposites).
        # So s >= max(range_k) / sqrt(3) for k=0,1,2.
        
        min_side = 0.0
        for k in range(3):
            n = normals[k]
            projs = pts_arr @ np.array(n)
            width = projs.max() - projs.min()
            min_side = max(min_side, width / math.sqrt(3.0))
        
        return min_side
    
    # Try different orientations and point selections
    # The outer hexagon can be rotated, so we try several orientations.
    
    # For N=11, let's try various compact arrangements
    # We'll try all subsets of size N from a small set of candidate points
    
    # Get top candidate points (closest to origin)
    # For N=11, we need at least 11 points
    candidate_points = points_sorted[:50]  # Top 50 closest points
    
    # Try different orientations for the outer hexagon
    # Due to 6-fold symmetry, we only need to check orientations in [0, 60)
    # But the inner hexagons can also be rotated. For simplicity, keep inner at 0 degrees.
    
    # Try a few orientations
    outer_angles = [0.0, 15.0, 30.0, 45.0, 60.0]
    
    # For efficiency, try all combinations of N points from a smaller set
    # and for each, compute the enclosing hexagon side for each orientation
    
    # Let's try all C(30, 11) is too many. Instead, try:
    # 1. The N closest points
    # 2. Various shifted clusters
    
    # Try the N closest points first
    for n_points in [N]:
        pts = [(p[0], p[1]) for p in points_sorted[:n_points]]
        for ang in outer_angles:
            side = enclosing_hexagon_side(pts, ang)
            if side < best_side:
                best_side = side
                best_config = pts[:]
                best_angle = ang
                # Center is the midpoint of the bounding box in each direction
                # For a regular hexagon, the center that minimizes the max projection
                # is the centroid of the projections
                theta = math.radians(ang)
                normals = []
                for k in range(6):
                    a = theta + k * math.pi / 3.0
                    normals.append((math.cos(a), math.sin(a)))
                pts_arr = np.array(pts)
                # The optimal center for each normal pair is the midpoint of min and max projection
                # Average over all 3 pairs
                cx = 0.0
                cy = 0.0
                for k in range(3):
                    n = normals[k]
                    projs = pts_arr @ np.array(n)
                    mid_proj = (projs.max() + projs.min()) / 2.0
                    # The center's projection onto n is mid_proj
                    # Solve for cx, cy from 3 such equations (least squares)
                    cx += mid_proj * n[0]
                    cy += mid_proj * n[1]
                # This isn't quite right; let's just use the centroid
                cx = np.mean([p[0] for p in pts])
                cy = np.mean([p[1] for p in pts])
                best_center = (float(cx), float(cy))
    
    # Now try shifted clusters: offset the lattice by small amounts
    # and re-select the N closest points to the offset center
    
    # Try a grid of offsets
    for ox in np.arange(-1.0, 1.0, 0.5):
        for oy in np.arange(-1.0, 1.0, 0.5):
            if _check_deadline():
                break
            # Shift all points
            shifted = [(p[0] + ox, p[1] + oy, p[2], p[3]) for p in points_sorted[:60]]
            # Sort by distance to origin
            shifted_sorted = sorted(shifted, key=lambda p: p[0]**2 + p[1]**2)
            pts = [(p[0], p[1]) for p in shifted_sorted[:N]]
            for ang in outer_angles:
                side = enclosing_hexagon_side(pts, ang)
                if side < best_side:
                    best_side = side
                    best_config = pts[:]
                    best_angle = ang
                    cx = np.mean([p[0] for p in pts])
                    cy = np.mean([p[1] for p in pts])
                    best_center = (float(cx), float(cy))
    
    if best_config is None:
        # Fallback: linear arrangement
        inner = [[d * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        total_width = d * (N - 1)
        side = total_width / math.sqrt(3.0) + 2.0
        return inner, (0.0, 0.0), float(side), 0.0
    
    # Build inner list
    inner = [[float(p[0]), float(p[1]), 0.0] for p in best_config]
    
    return inner, best_center, float(best_side), float(best_angle)


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
