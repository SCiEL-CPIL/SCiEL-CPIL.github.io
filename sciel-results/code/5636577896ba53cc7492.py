"""Packing 12 unit hexagons in a regular hexagon via rigid honeycomb cluster + 2D rotation optimization."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Honeycomb cluster: 7-hex flower + 5 more in a compact arrangement.
    # Honeycomb lattice vectors for unit hexagons (pointy-top, angle 0).
    # Distance between adjacent hex centers = 2 * apothem = sqrt(3).
    s3 = math.sqrt(3.0)
    # Lattice basis
    a1 = (s3, 0.0)
    a2 = (s3 / 2.0, 1.5)
    a3 = (-s3 / 2.0, 1.5)
    a4 = (-s3, 0.0)
    a5 = (-s3 / 2.0, -1.5)
    a6 = (s3 / 2.0, -1.5)

    # 7-hex flower: center + 6 neighbors
    flower = [(0.0, 0.0)] + [a1, a2, a3, a4, a5, a6]
    # Add 5 more hexagons adjacent to the flower to keep it compact.
    # Place them at positions that extend the cluster minimally.
    # Positions for the 5 extra hexes (chosen to form a compact 12-hex cluster)
    extra = [
        (s3 * 1.5, 0.0),      # beyond a1
        (s3 * 0.75, 3.0),     # beyond a2
        (-s3 * 0.75, 3.0),    # beyond a3
        (-s3 * 1.5, 0.0),     # beyond a4
        (s3 * 0.75, -3.0),    # beyond a5
    ]
    # Actually let me use a more standard compact 12-hex cluster.
    # Let's use: center, 6 neighbors, and 5 of the next ring positions.
    # Next ring positions (distance 2*sqrt(3) from center or sqrt(3) from neighbors)
    # Simpler: just use the 7 flower + 5 more at specific spots.
    # Let me redefine extra positions more carefully for compactness.
    extra = [
        (1.5 * s3, 0.0),
        (0.75 * s3, 3.0),
        (-0.75 * s3, 3.0),
        (-1.5 * s3, 0.0),
        (0.75 * s3, -3.0),
    ]

    centers = flower + extra
    # centers is list of (x, y)

    def hex_vertices(cx, cy, angle_deg, side=1.0):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + i * math.pi / 3.0
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with rotation angle_deg."""
        a = math.radians(angle_deg)
        # Transform to hex local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -a
        ca = math.cos(-a)
        sa = math.sin(-a)
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        # Regular hexagon with vertices at angles 0, 60, ... (pointy-top)
        # The hexagon is intersection of 6 half-planes.
        # For a hex with vertices at (side*cos(k*60), side*sin(k*60)),
        # the supporting lines are at distance apothem = side * cos(30) = side * sqrt(3)/2
        # from center, with normals at angles 30, 90, 150, 210, 270, 330 degrees.
        apothem = side * math.sqrt(3.0) / 2.0
        for k in range(6):
            normal_angle = math.radians(30.0 + k * 60.0)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > apothem:
                return False
        return True

    def hexagon_support(cx, cy, angle_deg, side=1.0, dirx=1.0, diry=0.0):
        """Support function: max of dot product of vertices with direction (dirx,diry)."""
        verts = hex_vertices(cx, cy, angle_deg, side)
        return max(vx * dirx + vy * diry for vx, vy in verts)

    def required_outer_side(centers, cluster_angle, outer_angle):
        """
        Compute the minimum outer hexagon side length needed to contain all hexagons.
        The outer hexagon has center at origin, rotation outer_angle, side S.
        For a regular hexagon with side S and rotation outer_angle, the support in direction
        (cos(theta), sin(theta)) is S * (cos(theta - outer_angle) projected onto hex normals).
        
        Actually, for a regular hexagon centered at origin with rotation phi and side S,
        the support function in direction u=(ux,uy) is:
        S * max over the 6 face normals of (dot(u, n_i)) where n_i are the unit normals.
        
        The 6 face normals of a hex with rotation phi are at angles phi + 30 + k*60.
        The apothem (distance from center to each face) is S * sqrt(3)/2.
        So support(u) = S * sqrt(3)/2 * max_i cos(angle(u) - (phi + 30 + k*60)).
        
        For a point p to be inside the outer hex, we need for all 6 face normals n_i:
        dot(p, n_i) <= S * sqrt(3)/2
        
        So S >= max over all points p (which are vertices of inner hexes) and all 6 normals n_i:
        dot(p, n_i) / (sqrt(3)/2)
        """
        oa = math.radians(outer_angle)
        # 6 face normals of outer hex
        normals = []
        for k in range(6):
            na = oa + math.radians(30.0 + k * 60.0)
            normals.append((math.cos(na), math.sin(na)))
        
        max_val = 0.0
        for (cx, cy) in centers:
            verts = hex_vertices(cx, cy, cluster_angle, 1.0)
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    val = (vx * nx + vy * ny)
                    if val > max_val:
                        max_val = val
        
        # S * sqrt(3)/2 >= max_val  =>  S >= max_val * 2 / sqrt(3)
        return max_val * 2.0 / math.sqrt(3.0)

    def objective(params):
        cluster_angle, outer_angle = params
        return required_outer_side(centers, cluster_angle, outer_angle)

    # Coarse grid search over cluster_angle and outer_angle
    # Both angles are periodic with period 60 degrees (hexagonal symmetry)
    # So we only need to search [0, 60) for each.
    best_val = float('inf')
    best_params = (0.0, 0.0)
    
    # Coarse grid: 60 steps each => 3600 evaluations
    n_coarse = 60
    for i in range(n_coarse):
        ca = i * 60.0 / n_coarse
        for j in range(n_coarse):
            oa = j * 60.0 / n_coarse
            val = objective((ca, oa))
            if val < best_val:
                best_val = val
                best_params = (ca, oa)
        
        if time.time() > deadline - 20.0:
            break
    
    # Fine grid search around the best
    best_ca, best_oa = best_params
    n_fine = 30
    step = 60.0 / (n_coarse * n_fine)
    for i in range(-n_fine, n_fine + 1):
        ca = (best_ca + i * step) % 60.0
        for j in range(-n_fine, n_fine + 1):
            oa = (best_oa + j * step) % 60.0
            val = objective((ca, oa))
            if val < best_val:
                best_val = val
                best_params = (ca, oa)
        
        if time.time() > deadline - 10.0:
            break
    
    best_ca, best_oa = best_params
    
    # Even finer search
    n_fine2 = 50
    step2 = step / n_fine2
    for i in range(-n_fine2, n_fine2 + 1):
        ca = (best_ca + i * step2) % 60.0
        for j in range(-n_fine2, n_fine2 + 1):
            oa = (best_oa + j * step2) % 60.0
            val = objective((ca, oa))
            if val < best_val:
                best_val = val
                best_params = (ca, oa)
        
        if time.time() > deadline - 5.0:
            break
    
    best_ca, best_oa = best_params
    final_side = best_val
    
    # Build inner hexagon positions (all with same rotation = cluster_angle)
    inner = [[cx, cy, best_ca] for (cx, cy) in centers]
    
    # Outer hexagon centered at origin
    return inner, (0.0, 0.0), final_side, best_oa
# EVOLVE-BLOCK-END
