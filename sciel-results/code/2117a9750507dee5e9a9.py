"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time
from itertools import combinations

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    for verts in [v1, v2]:
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            # Normal to edge
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            # Project all vertices onto normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 + tol or max2 < min1 + tol:
                return False  # Separating axis found
    return True  # No separating axis, polygons overlap

def _outer_hex_side(inner_positions, outer_center, outer_angle):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    ocx, ocy = outer_center
    o_angle_rad = np.radians(outer_angle)
    
    max_required_inradius = 0
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, iangle, 1.0)
        for vx, vy in verts:
            # Relative to outer center
            dx = vx - ocx
            dy = vy - ocy
            # Project onto each of 6 face normal directions
            for k in range(6):
                theta = o_angle_rad + k * np.pi / 3 + np.pi / 6  # face normals
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = dx * nx + dy * ny
                if dist > max_required_inradius:
                    max_required_inradius = dist
    
    # inradius = side * sqrt(3)/2, so side = inradius / (sqrt(3)/2) = 2*inradius/sqrt(3)
    return 2.0 * max_required_inradius / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check if packing is valid (no overlaps, all inside outer hex)."""
    # Check no overlaps between inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i+1, len(inner_positions)):
            v1 = _hex_vertices(inner_positions[i][0], inner_positions[i][1], inner_positions[i][2])
            v2 = _hex_vertices(inner_positions[j][0], inner_positions[j][1], inner_positions[j][2])
            if _polygons_overlap(v1, v2, tol=-1e-7):
                return False
    # Check all inside outer hex
    ocx, ocy = outer_center
    o_angle_rad = np.radians(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2
    for (ix, iy, iangle) in inner_positions:
        verts = _hex_vertices(ix, iy, iangle, 1.0)
        for vx, vy in verts:
            dx = vx - ocx
            dy = vy - ocy
            for k in range(6):
                theta = o_angle_rad + k * np.pi / 3 + np.pi / 6
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = dx * nx + dy * ny
                if dist > inradius + 1e-7:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Strategy: Discrete search on triangular lattice
    # Inner hexagons are constrained to a triangular lattice with spacing 2 (side=1 hexes)
    # We search over:
    # 1. Which 12 lattice points to use (combinatorial)
    # 2. Global rotation angle of the lattice (continuous, but we grid search)
    # 3. Outer hexagon rotation angle (continuous, grid search)
    # 4. Translation of the cluster (continuous, but we can center it)
    
    # For a triangular lattice with spacing d=2 (centers of non-overlapping unit hexes):
    # Basis vectors: v1 = (2, 0), v2 = (1, sqrt(3))
    # Lattice points: P = i*v1 + j*v2 for integers i, j
    
    # Generate a set of candidate lattice points around origin
    # We need 12 points that form a compact cluster
    
    # Pre-generate candidate lattice points
    max_coord = 5
    lattice_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i * 2.0 + j * 1.0
            y = j * sqrt3
            lattice_points.append((x, y))
    
    # We'll search over subsets of 12 lattice points
    # But C(100, 12) is too large, so we need a smarter approach
    
    # Alternative: Use known compact arrangements on triangular lattice
    # A hexagonal cluster of 12 hexes:
    # Center + 6 around + 5 more to make 12
    
    # Let's define specific compact clusters on the triangular lattice
    # Cluster 1: 1-6-5 (center, ring 1, part of ring 2)
    # Axial coordinates for triangular lattice
    # (q, r) -> (1.5*q, sqrt3*(r + q/2)) for flat-top hexes
    # But for centers on triangular lattice with spacing 2:
    # (i, j) -> (2*i + j, sqrt3*j)
    
    # Define several candidate clusters as sets of (i, j) lattice indices
    clusters = []
    
    # Cluster A: Hexagonal shape (1 center, 6 around, 5 from next ring)
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 partial: (2,0), (2,-1), (1,-2), (-1,2), (0,2) -- pick 5
    # Let's try symmetric ones
    
    # Cluster 1: 2-3-4-3 arrangement (rows)
    # In axial coords (q, r):
    # r=-1: q=0,1,2,3 (4 hexes)
    # r=0: q=-1,0,1 (3 hexes)  
    # r=1: q=-1,0,1 (3 hexes)
    # r=2: q=0,1 (2 hexes)
    # Total: 12
    
    cluster1_axial = [
        (0, -1), (1, -1), (2, -1), (3, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2), (1, 2)
    ]
    
    # Cluster 2: 3-4-3-2 arrangement
    cluster2_axial = [
        (0, -1), (1, -1), (2, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2), (1, 2)
    ]
    
    # Cluster 3: More symmetric 2-2-4-2-2
    cluster3_axial = [
        (0, -2), (1, -2),
        (-1, -1), (0, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1),
        (0, 2), (1, 2)
    ]
    
    # Cluster 4: 1-3-4-3-1
    cluster4_axial = [
        (0, -2),
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, 2)
    ]
    
    # Convert axial to pixel for triangular lattice (spacing 2)
    def axial_to_pixel(q, r):
        # For triangular lattice with spacing 2 (center-to-center)
        x = 2.0 * q + 1.0 * r
        y = sqrt3 * r
        return x, y
    
    def axial_to_pixel_alt(q, r):
        # Alternative mapping
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return x, y
    
    # We'll use multiple cluster definitions and search over rotations
    
    best_side = np.inf
    best_result = None
    
    # Grid search over outer hexagon angle
    outer_angles = np.arange(0, 180, 5)  # 36 values
    
    # For each cluster and outer angle, compute the bounding hexagon
    # and check validity
    
    for cluster_idx, cluster_axial in enumerate([cluster1_axial, cluster2_axial, cluster3_axial, cluster4_axial]):
        if time.time() > deadline - 20:
            break
        
        # Convert to pixel coordinates
        positions = []
        for q, r in cluster_axial:
            x, y = axial_to_pixel(q, r)
            positions.append((x, y))
        
        # Center the cluster
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions = [(x - cx, y - cy) for x, y in positions]
        
        # All inner hexagons have angle 0 (aligned with lattice)
        inner_positions = [(x, y, 0.0) for x, y in positions]
        
        # Try different outer angles
        for oa in outer_angles:
            if time.time() > deadline - 10:
                break
            
            outer_center = (0.0, 0.0)
            side = _outer_hex_side(inner_positions, outer_center, oa)
            
            if side < best_side:
                # Check validity
                if _check_validity(inner_positions, outer_center, side, oa):
                    best_side = side
                    best_result = (inner_positions, outer_center, side, oa)
    
    # If no valid result found, fallback to a simple arrangement
    if best_result is None:
        # Simple fallback: 4x3 grid
        positions = []
        for i in range(3):
            for j in range(4):
                x = j * 2.0
                y = i * sqrt3
                positions.append((x, y))
        
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        positions = [(x - cx, y - cy) for x, y in positions]
        inner_positions = [(x, y, 0.0) for x, y in positions]
        
        for oa in outer_angles:
            outer_center = (0.0, 0.0)
            side = _outer_hex_side(inner_positions, outer_center, oa)
            if _check_validity(inner_positions, outer_center, side, oa):
                best_side = side
                best_result = (inner_positions, outer_center, side, oa)
                break
        
        if best_result is None:
            # Last resort: return a large hexagon
            inner_positions = [(i * 2.0, 0.0, 0.0) for i in range(N)]
            outer_center = (0.0, 0.0)
            side = _outer_hex_side(inner_positions, outer_center, 0.0)
            best_result = (inner_positions, outer_center, side, 0.0)
    
    inner_positions, outer_center, side, oa = best_result
    
    inner = [[float(x), float(y), float(a)] for x, y, a in inner_positions]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(side), float(oa)
# EVOLVE-BLOCK-END
