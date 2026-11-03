import numpy as np
import math
import time

N = 11

def hexagon_vertices(cx, cy, r, angle_deg=0):
    """
    Returns vertices of a regular hexagon with center (cx, cy), radius r (center to vertex),
    rotated by angle_deg.
    """
    angles = [math.radians(angle_deg + i * 60) for i in range(6)]
    return [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]

def point_in_hexagon(x, y, R, angle_deg=0):
    """
    Check if point (x,y) is inside a regular hexagon of side length R,
    centered at origin, rotated by angle_deg.
    Side length R means distance from center to vertex is R.
    """
    # Rotate point by -angle_deg
    rad = math.radians(-angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    x_rot = x * cos_a - y * sin_a
    y_rot = x * sin_a + y * cos_a
    
    # For a hexagon with vertices at 0, 60, ..., side length R (circumradius R)
    # The boundaries are at distance R * cos(30) = R * sqrt(3)/2 from center for the sides?
    # No, circumradius is R. Inradius is R * sqrt(3)/2.
    # The hexagon is defined by:
    # |x_rot| <= R * sqrt(3) / 2
    # |y_rot| <= R
    # |x_rot * sqrt(3) + y_rot| <= R * sqrt(3) ? No.
    
    # Let's use the standard inequalities for a hexagon with inradius r_in = R * sqrt(3)/2.
    # The sides are at distance r_in from the center.
    # The normals are at 30, 90, 150, 210, 270, 330.
    # Condition: dot(p, n_i) <= r_in for all i.
    
    r_in = R * math.sqrt(3) / 2
    normals = [math.radians(30 + i * 60) for i in range(6)]
    for a in normals:
        if x_rot * math.cos(a) + y_rot * math.sin(a) > r_in + 1e-9:
            return False
    return True

def min_outer_hexagon_side(centers, angles, R_guess=10.0):
    """
    Find the minimal R such that all hexagons with given centers and angles
    fit in a regular hexagon of side R centered at origin.
    We assume the outer hexagon is aligned with the lattice (angle 0).
    """
    # We need to find R such that all vertices of all inner hexagons are inside.
    # This is equivalent to finding the max "hexagonal norm" of all vertices.
    # For a point p, the minimal R such that p is in hexagon(R) is:
    # R = max( |x| / (sqrt(3)/2), |y|, |x*sqrt(3)+y|/sqrt(3) ... ) ?
    
    # Let's derive the formula for R given a point (x,y) in the frame where hexagon is standard.
    # Standard hexagon (pointy top? No, flat top usually means vertices at 0? No.)
    # Let's stick to the normal test.
    # For a point (x,y), the required R is:
    # R_min = max_i ( dot(p, n_i) / cos(0) ) ? No.
    # dot(p, n_i) <= R * sqrt(3)/2.
    # So R >= dot(p, n_i) * 2 / sqrt(3).
    # We need to check all 6 normals.
    
    max_R = 0.0
    normals = [math.radians(30 + i * 60) for i in range(6)]
    
    for cx, cy, ang in zip(centers[:,0], centers[:,1], angles):
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            # Rotate to standard frame? No, the outer hexagon is fixed.
            # The inner hexagons are rotated by 'ang'.
            # The point (vx, vy) is in global coordinates.
            # We check if it fits in outer hexagon of side R centered at 0.
            # Condition: dot((vx,vy), n_i) <= R * sqrt(3)/2.
            # So R >= dot((vx,vy), n_i) * 2 / sqrt(3).
            for a in normals:
                proj = vx * math.cos(a) + vy * math.sin(a)
                if proj > 0:
                    req_R = proj * 2 / math.sqrt(3)
                    if req_R > max_R:
                        max_R = req_R
    return max_R

def generate_lattice_cluster(n, shape='hex'):
    """
    Generate n centers on a hexagonal lattice, centered at 0.
    """
    centers = []
    # Use axial coordinates
    # q, r
    # Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    # Distance between centers is sqrt(3).
    
    # Generate a large enough grid and pick n closest to origin
    max_q = 5
    max_r = 5
    candidates = []
    for q in range(-max_q, max_q + 1):
        for r in range(-max_r, max_r + 1):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            dist = math.sqrt(x*x + y*y)
            candidates.append((dist, q, r, x, y))
    
    candidates.sort(key=lambda t: t[0])
    
    # Pick n unique centers
    # Note: multiple (q,r) can map to same (x,y)? No, axial is unique.
    selected = candidates[:n]
    cx = [c[3] for c in selected]
    cy = [c[4] for c in selected]
    angles = [0.0] * n
    return np.array([cx, cy]).T, angles

def construct():
    start_time = time.time()
    deadline = 150.0
    
    best_R = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different cluster shapes
    # 1. Simple hexagonal cluster (closest to origin)
    centers, angles = generate_lattice_cluster(N, 'hex')
    R = min_outer_hexagon_side(centers, angles)
    if R < best_R:
        best_R = R
        best_inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
        
    # 2. Try rotating the inner cluster?
    # Since the outer hexagon is symmetric, rotating the inner cluster by 30 deg
    # might change the fit?
    # The lattice is symmetric, so rotating by 30 deg maps it to itself?
    # No, the lattice of centers is symmetric under 60 deg rotation.
    # The hexagons themselves are symmetric under 60 deg.
    # So the whole configuration is symmetric under 60 deg.
    # Rotating by 30 deg is NOT a symmetry of the hexagonal lattice.
    # Let's try rotating the inner hexagons by 30 deg relative to the lattice?
    # No, the lattice centers are fixed. The hexagons can rotate.
    # If we rotate all hexagons by 30 deg, do they still fit?
    # The distance between centers is sqrt(3).
    # If we rotate hexagons by 30 deg, they might overlap?
    # In a hexagonal lattice, hexagons are tangent.
    # If we rotate them, they will overlap unless we move them.
    # So we should keep the standard orientation (angle 0).
    
    # 3. Try a different shape: "Line"
    # 11 in a row
    centers_line = np.array([[i * math.sqrt(3), 0.0] for i in range(-(N//2), N - N//2)])
    angles_line = [0.0] * N
    R_line = min_outer_hexagon_side(centers_line, angles_line)
    if R_line < best_R:
        best_R = R_line
        best_inner = [[c[0], c[1], a] for c, a in zip(centers_line, angles_line)]
        
    # 4. Try "2 rows"
    # Row 1: 6 hexagons
    # Row 2: 5 hexagons
    # Staggered
    c1 = []
    for i in range(3):
        c1.append((i * math.sqrt(3), 0.0))
        c1.append(((i-1) * math.sqrt(3), 0.0)) # This is messy
    
    # Better: Generate lattice, pick specific ones
    # For 2 rows:
    # y=0: x = -2.5*s3, -1.5*s3, -0.5*s3, 0.5*s3, 1.5*s3, 2.5*s3 (6 hex)
    # y=1.5: x = -2.0*s3, -1.0*s3, 0.0, 1.0*s3, 2.0*s3 (5 hex)
    s3 = math.sqrt(3)
    centers_2row = []
    for i in range(-3, 3):
        centers_2row.append((i * s3, 0.0))
    for i in range(-2, 3):
        centers_2row.append((i * s3, 1.5))
        
    # Check overlap?
    # Distance between (i*s3, 0) and (j*s3, 1.5)
    # If j=i, dist = 1.5. < s3? 1.5 < 1.732. Overlap!
    # So this packing is invalid.
    # The staggered rows must be at distance 1.5 * s3?
    # Vertical distance between rows in hex lattice is 1.5.
    # Horizontal offset is s3/2.
    # Let's use the lattice generator but filter for specific rows.
    
    # The lattice generator already finds the optimal cluster for the lattice.
    # The "closest to origin" heuristic gives the hexagonal cluster.
    # For N=11, this is 7 (radius 1) + 4 (part of radius 2).
    # This is likely the optimal lattice packing.
    
    # Let's verify the R for the hex cluster.
    # If R is still large, we might try to optimize the outer hexagon rotation.
    # But the cluster is symmetric, so rotation doesn't help.
    
    # Final check: return the best found
    if best_inner is None:
        # Fallback
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_R = 40.0
        
    return best_inner, best_center, best_R, best_angle

# EVOLVE-BLOCK-END
