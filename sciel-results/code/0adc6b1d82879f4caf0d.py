"""Packing unit hexagons (aligned, triangular lattice) inside a regular hexagon.

Structural design:
- All inner hexagons share orientation 0 (aligned).
- Centers lie on a triangular lattice with spacing sqrt(3) (exact non-overlap for aligned unit hexes).
- We enumerate compact hexagonal-shaped clusters of N lattice points.
- The outer hexagon (orientation 0, centered at cluster centroid) is sized minimally to enclose all inner hexes.
- Score = outer side length (lower is better).
"""
import time
import math
import numpy as np

N = 11

def _hex_vertices(center, side, angle_deg=0.0):
    """Vertices of a regular hexagon."""
    cx, cy = center
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        th = a + math.radians(60 * k)
        verts.append((cx + side * math.cos(th), cy + side * math.sin(th)))
    return np.array(verts)

def _point_in_hexagon(px, py, center, side, angle_deg=0.0):
    """Check if point (px,py) is inside (or on) a regular hexagon via half-plane tests."""
    cx, cy = center
    a = math.radians(angle_deg)
    for k in range(6):
        th = a + math.radians(60 * k)
        nx = math.cos(th + math.radians(30))
        ny = math.sin(th + math.radians(30))
        dx = px - cx
        dy = py - cy
        if dx * nx + dy * ny > side * math.cos(math.radians(30)) + 1e-9:
            return False
    return True

def _hexagon_contains_hexagon(center_outer, side_outer, angle_outer,
                               center_inner, side_inner, angle_inner):
    """Check if inner hexagon is fully inside outer hexagon."""
    verts = _hex_vertices(center_inner, side_inner, angle_inner)
    for (vx, vy) in verts:
        if not _point_in_hexagon(vx, vy, center_outer, side_outer, angle_outer):
            return False
    return True

def _min_enclosing_hex_side(cluster_points, angle_outer=0.0):
    """Given cluster center points (aligned inner hexes, side=1, angle=0),
    find the minimal side length of an outer regular hexagon (same orientation,
    centered at cluster centroid) that encloses all inner hexagons."""
    pts = np.array(cluster_points)
    centroid = pts.mean(axis=0)
    
    apothem_factor = math.cos(math.radians(30))
    
    # Inner hex vertices offsets (side=1, angle=0)
    inner_verts = []
    for k in range(6):
        th = math.radians(60 * k)
        inner_verts.append((math.cos(th), math.sin(th)))
    inner_verts = np.array(inner_verts)
    
    # All absolute vertex positions relative to centroid
    all_pts = []
    for c in cluster_points:
        for v in inner_verts:
            all_pts.append((c[0] + v[0] - centroid[0], c[1] + v[1] - centroid[1]))
    all_pts = np.array(all_pts)
    
    # For outer hexagon with angle 0, the 6 edge normals
    normals = []
    for k in range(6):
        th = math.radians(30 + 60 * k)
        normals.append((math.cos(th), math.sin(th)))
    normals = np.array(normals)
    
    projections = all_pts @ normals.T
    max_proj = projections.max()
    
    min_side = max_proj / apothem_factor
    return min_side, centroid

def _generate_hex_cluster(radius):
    """Generate a hexagonal cluster of lattice points with given radius."""
    pts = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            dist = max(abs(q), abs(r), abs(-q - r))
            if dist <= radius:
                x = math.sqrt(3) * q + math.sqrt(3) / 2 * r
                y = 1.5 * r
                pts.append((x, y))
    return pts

def _generate_compact_clusters(n_target):
    """Generate candidate compact clusters of exactly n_target points."""
    candidates = []
    seen = set()
    
    for radius in range(1, 6):
        pts = _generate_hex_cluster(radius)
        if len(pts) == n_target:
            key = tuple(sorted(pts))
            if key not in seen:
                seen.add(key)
                candidates.append(pts)
        elif len(pts) > n_target:
            pts_arr = np.array(pts)
            centroid = pts_arr.mean(axis=0)
            dists = np.linalg.norm(pts_arr - centroid, axis=1)
            order = np.argsort(dists)
            selected = [pts[i] for i in order[:n_target]]
            key = tuple(sorted(selected))
            if key not in seen:
                seen.add(key)
                candidates.append(selected)
        elif len(pts) < n_target:
            need = n_target - len(pts)
            base_arr = np.array(pts)
            centroid = base_arr.mean(axis=0)
            next_ring = []
            for q in range(-radius - 1, radius + 2):
                for r in range(-radius - 1, radius + 2):
                    dist = max(abs(q), abs(r), abs(-q - r))
                    if dist == radius + 1:
                        x = math.sqrt(3) * q + math.sqrt(3) / 2 * r
                        y = 1.5 * r
                        next_ring.append((x, y))
            if next_ring:
                next_ring_arr = np.array(next_ring)
                dists = np.linalg.norm(next_ring_arr - centroid, axis=1)
                order = np.argsort(dists)
                extras = [next_ring[i] for i in order[:need]]
                candidate = pts + extras
                key = tuple(sorted(candidate))
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)
    
    return candidates

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    candidates = _generate_compact_clusters(N)
    
    for cluster in candidates:
        if time.time() > deadline:
            break
        
        side, centroid = _min_enclosing_hex_side(cluster)
        
        if side < best_side:
            best_side = side
            best_center = centroid
            best_inner = [(c[0], c[1], 0.0) for c in cluster]
            best_angle = 0.0
    
    if best_inner is None or best_side == float('inf'):
        inner = []
        for i in range(N):
            x = (i - (N - 1) / 2.0) * math.sqrt(3)
            inner.append((x, 0.0, 0.0))
        pts = [(p[0], p[1]) for p in inner]
        side, centroid = _min_enclosing_hex_side(pts)
        best_side = side
        best_inner = inner
        best_center = centroid
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
