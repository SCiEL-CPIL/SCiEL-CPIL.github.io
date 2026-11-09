"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    # Function to compute the minimal enclosing hexagon side length
    # for a set of inner hexagons given their centers, angles, and the outer hexagon's rotation
    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute the minimal side length of a regular hexagon (centered at origin, 
        with given rotation) that contains all inner unit hexagons.
        
        For a regular hexagon with side length L and rotation theta, the boundary
        consists of 6 half-planes. A point (x,y) is inside if for all 6 directions,
        the dot product with the outward normal is <= L * cos(30°) = L * sqrt(3)/2.
        
        Actually, for a regular hexagon with flat top, the distance from center to each side
        is L * sqrt(3) / 2. The 6 outward normals are at angles 0°, 60°, 120°, 180°, 240°, 300°.
        
        For each inner hexagon, we need to find the maximum "extent" in each of the 6 directions
        from the outer hexagon's center. The inner hexagon has 6 vertices; we need the max dot product
        of these vertices with each outward normal direction.
        """
        # Outer hexagon outward normals (unit vectors) for 6 sides
        # For a regular hexagon with flat top and rotation outer_angle:
        # The 6 outward normal directions are at angles: outer_angle, outer_angle+60°, ..., outer_angle+300°
        # Wait, let me reconsider. A regular hexagon with "flat top" has vertices at 0°, 60°, 120°, 180°, 240°, 300°
        # and the outward normals to the sides are at 30°, 90°, 150°, 210°, 270°, 330°.
        
        # Actually, for a regular hexagon with side length L centered at origin:
        # If vertices are at angles k*60° (k=0..5), then the side between vertex k and k+1
        # has its outward normal at angle (k*60° + 30°).
        # The distance from center to each side is L * cos(30°) = L * sqrt(3)/2.
        
        # So for a hexagon with rotation angle phi (rotation of the whole hexagon),
        # the 6 outward normal directions are at angles: phi + 30° + k*60° for k=0..5.
        
        # A point (x,y) is inside if for all 6 directions n_k: dot((x,y), n_k) <= L * sqrt(3)/2
        # So L >= max_k dot((x,y), n_k) / (sqrt(3)/2) for all points.
        
        # For each inner hexagon, we need the maximum extent in each of the 6 normal directions.
        # The inner hexagon has vertices at positions: center + radius * (cos(angle_j), sin(angle_j))
        # where radius = 1 (for unit hexagon, the distance from center to vertex is 1)
        # and angle_j = inner_angle + j*60° for j=0..5.
        
        # The extent in direction n is: dot(center, n) + max_j dot(vertex_offset_j, n)
        # = dot(center, n) + max_j cos(inner_angle + j*60° - angle_n)
        
        # For a regular hexagon, max_j cos(theta + j*60°) = cos(theta mod 60°) adjusted...
        # Actually, max of cos(a + k*60°) for k=0..5 = max of {cos(a), cos(a-60°), cos(a-120°), cos(a-180°), cos(a-240°), cos(a-300°)}
        # = max of cos(a mod 60°) where we take the appropriate value.
        # More precisely, the maximum of cos(x + k*pi/3) for k=0..5 is cos(x mod pi/3) if we reduce x mod pi/3 into [0, pi/3].
        
        # Let's compute it directly for accuracy.
        
        # 6 outward normal directions for outer hexagon with rotation outer_angle
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])
            
            # Vertices of inner hexagon (relative to its center)
            # For a unit hexagon, vertices are at distance 1 from center
            # Vertex angles: inner_angle + j*60°
            # vertex_offset_j = (cos(inner_angle + j*60°), sin(inner_angle + j*60°))
            
            for nk in range(6):
                nx, ny = normals[nk]
                # Extent = dot(center, n) + max_j dot(vertex_offset_j, n)
                # = cx*nx + cy*ny + max_j (cos(inner_angle + j*60°)*nx + sin(inner_angle + j*60°)*ny)
                # = cx*nx + cy*ny + max_j cos(inner_angle + j*60° - atan2(ny, nx))
                
                n_angle = math.atan2(ny, nx)
                max_dot = 0.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val
                
                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent
        
        # L = max_extent / (sqrt(3)/2) = 2 * max_extent / sqrt(3)
        return 2.0 * max_extent / sqrt3

    # Strategy: Try a few good configurations
    # Configuration 1: 12 hexagons in a ring around the center
    # Place 6 hexagons on each of two rings
    
    best_L = float('inf')
    best_config = None
    
    # Try configuration: 6 inner hexagons on a ring of radius r1, 6 on a ring of radius r2
    # With appropriate angles to minimize overlap and enclosing hexagon
    
    # Let's try a symmetric arrangement: 12 hexagons in a flower pattern
    # 1 at center, 6 in first ring, 5 in second ring? No, that's 12.
    # Actually, let's try: 6 hexagons at radius ~2, 6 hexagons at radius ~4
    
    # Better: Try to optimize positions numerically
    
    # Initial guess: hexagonal arrangement
    # Place hexagons at positions that form a larger hexagon pattern
    
    # For now, let's try a simple parametric search
    # Arrangement: 6 hexagons on inner ring, 6 on outer ring
    
    def evaluate_config(centers, angles, outer_angle=0.0):
        return enclosing_side_length(centers, angles, outer_angle)
    
    # Try several configurations
    
    # Config A: 6 on ring radius r1 at angles 0,60,120,180,240,300
    #           6 on ring radius r2 at angles 30,90,150,210,270,330
    
    for r1 in [1.8, 2.0, 2.2, 2.5]:
        for r2 in [3.5, 4.0, 4.5, 5.0]:
            centers = []
            angles = []
            for k in range(6):
                a1 = math.radians(k * 60)
                centers.append((r1 * math.cos(a1), r1 * math.sin(a1)))
                angles.append(0.0)
            for k in range(6):
                a2 = math.radians(k * 60 + 30)
                centers.append((r2 * math.cos(a2), r2 * math.sin(a2)))
                angles.append(0.0)
            
            L = evaluate_config(centers, angles)
            if L < best_L:
                best_L = L
                best_config = (centers, angles, 0.0)
    
    # Config B: All 12 on a single ring
    for r in [3.5, 4.0, 4.5, 5.0, 5.5]:
        centers = []
        angles = []
        for k in range(12):
            a = math.radians(k * 30)
            centers.append((r * math.cos(a), r * math.sin(a)))
            angles.append(0.0)
        
        L = evaluate_config(centers, angles)
        if L < best_L:
            best_L = L
            best_config = (centers, angles, 0.0)
    
    # Config C: Hexagonal lattice arrangement
    # Place hexagons at hex grid positions within a hex of radius 2 (in hex units)
    # Hex grid coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2 gives 19 positions
    # We need 12 of them
    
    # Let's pick 12 positions from the hex grid
    # The 19 positions: center + 6 at distance 1 + 12 at distance 2
    # Use center + 6 at distance 1 + 5 at distance 2 = 12
    
    centers = []
    angles = []
    # Center
    centers.append((0.0, 0.0))
    angles.append(0.0)
    # 6 at distance 1 (hex grid)
    for k in range(6):
        a = math.radians(k * 60)
        x = 1.5 * math.cos(a)
        y = sqrt3 * (math.sin(a))  # approximate
        # Actually, hex grid neighbors are at:
        # (1.5, sqrt3/2), (0, sqrt3), (-1.5, sqrt3/2), (-1.5, -sqrt3/2), (0, -sqrt3), (1.5, -sqrt3/2)
        neighbors = [(1.5, sqrt3/2), (0, sqrt3), (-1.5, sqrt3/2), (-1.5, -sqrt3/2), (0, -sqrt3), (1.5, -sqrt3/2)]
        centers.append(neighbors[k])
        angles.append(0.0)
    # 5 more from distance 2
    # Distance 2 hex grid positions include: (3,0), (1.5, sqrt3*1.5), etc.
    # Let's add a few more
    extra = [(3.0, 0.0), (1.5, 1.5*sqrt3), (-1.5, 1.5*sqrt3), (-3.0, 0.0), (-1.5, -1.5*sqrt3)]
    for pos in extra[:5]:
        centers.append(pos)
        angles.append(0.0)
    
    L = evaluate_config(centers, angles)
    if L < best_L:
        best_L = L
        best_config = (centers, angles, 0.0)
    
    # Refine best configuration with local search
    if best_config is not None:
        centers, angles, outer_angle = best_config
        
        # Simple local optimization: perturb positions slightly
        for iteration in range(200):
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for i in range(N):
                for dx in [-0.1, 0.1]:
                    for dy in [-0.1, 0.1]:
                        new_centers = list(centers)
                        new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                        new_L = evaluate_config(new_centers, angles)
                        if new_L < best_L - 1e-9:
                            best_L = new_L
                            centers = new_centers
                            improved = True
                        if time.time() > deadline - 10.0:
                            break
                    if time.time() > deadline - 10.0:
                        break
                if time.time() > deadline - 10.0:
                    break
            if not improved:
                break
        
        best_config = (centers, angles, outer_angle)
    
    # Extract final configuration
    if best_config is not None:
        centers, angles, outer_angle = best_config
    else:
        # Fallback: simple linear arrangement
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N
        outer_angle = 0.0
        best_L = 40.0
    
    # Convert to required format
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
    
    # Center of outer hexagon is (0, 0)
    center = (0.0, 0.0)
    side = best_L
    angle = outer_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

