import math
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # 11 unit hexagons in axial coordinates (q, r)
    # We'll use a symmetric "flower" pattern plus one extra, or a compact cluster.
    # Let's use the standard hexagonal lattice positions for 11 hexagons:
    # A central hexagon, 6 around it, and 4 more to make a compact shape.
    # Actually, let's use a known good packing: a 3x3 grid minus 2 corners, or similar.
    
    # Let's define positions in axial coordinates (q, r) which map to:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    # A good starting configuration for 11 hexagons:
    # Center at (0,0)
    # Ring 1: 6 hexagons at distance 1
    # Ring 2: 4 hexagons at specific positions to fill gaps
    
    # Let's use a "staggered" 3x4 grid pattern or similar.
    # Positions in axial coordinates:
    axial_positions = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (-1, 2), (-2, 1),  # part of ring 2 (4 hexagons)
    ]
    
    # Convert to Cartesian
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Initial positions
    pts = np.array([axial_to_cartesian(q, r) for q, r in axial_positions])
    
    # Parameters to optimize:
    # - cx, cy: center of outer hexagon
    # - theta: rotation of outer hexagon (0 to pi/6 due to symmetry)
    # - scale: slight scaling of inner configuration to allow for tighter packing
    #   Actually, we can't scale the inner hexagons (they're unit). 
    #   But we can optimize the relative positions slightly? No, they must be disjoint.
    #   The key insight: if we rotate the inner hexagons, we might pack them tighter.
    #   However, for simplicity and robustness, let's first optimize the outer hexagon
    #   position, rotation, and allow the inner configuration to be shifted.
    
    # Actually, the best approach for hexagons:
    # 1. The inner hexagons can be rotated. If we rotate all inner hexagons by the same angle,
    #    and the outer hexagon by the same angle, the problem is invariant.
    # 2. So we can fix the inner hexagons' orientation and optimize the outer hexagon.
    # 3. But we can also allow individual inner hexagons to rotate, which might help.
    
    # For a strong baseline, let's optimize:
    # - cx, cy: outer center
    # - theta: outer rotation
    # - Optionally, small perturbations of inner positions (but this risks overlap)
    
    # Let's use differential evolution to find good (cx, cy, theta)
    
    def objective(params):
        cx, cy, theta = params
        theta = theta % (math.pi / 6.0)  # Hexagon has 60-degree symmetry
        
        # Normals for the outer hexagon
        normals = []
        for k in range(6):
            ang = theta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For each inner hexagon, compute the maximum projection onto each normal
        # The inner hexagon at position (px, py) with orientation 0 has vertices at:
        # For a unit hexagon centered at origin, the maximum projection onto a normal (nx, ny)
        # is the apothem if the normal is aligned with a side, or the circumradius if aligned with a vertex.
        # Actually, for a regular hexagon with side length 1, the support function in direction (nx, ny)
        # is: max over vertices of dot(vertex, (nx, ny))
        # The vertices of a unit hexagon at origin (pointy-top) are:
        # (sqrt(3)/2, 0), (0, 1), (-sqrt(3)/2, 0), (-sqrt(3)/2, 0)... wait.
        
        # Let's use the fact that for a regular hexagon with side length s, the support function
        # in direction making angle alpha with the x-axis is:
        # h(alpha) = s * max over k of cos(alpha - k*pi/3) for the 6 directions to vertices
        # Actually, simpler: the width of a hexagon in any direction is between s*sqrt(3) and 2s.
        # The support function h(n) for a unit hexagon is:
        # h(n) = max( |n_x| * sqrt(3)/2, |n_y|, |n_x| * sqrt(3)/2 - |n_y|/2, ... )
        
        # Let's just compute it directly for each inner hexagon.
        # Since all inner hexagons have the same orientation (0 degrees), we can precompute
        # the support function.
        
        # For a unit hexagon centered at origin with flat-top orientation:
        # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Wait, that's for circumradius 1. For side length 1, circumradius is also 1.
        
        # Let's use pointy-top: vertices at angles 30, 90, 150, 210, 270, 330 degrees
        # Or flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
        
        # Let's use the standard: for a unit hexagon, the maximum projection onto a unit vector
        # at angle phi is:
        # h(phi) = max_{k=0..5} cos(phi - k*pi/3) * 1  (since circumradius = 1)
        # But this is for the circumcircle. The actual hexagon support function is:
        # h(phi) = max( cos(phi), cos(phi - pi/3), cos(phi - 2*pi/3), cos(phi - pi), cos(phi - 4*pi/3), cos(phi - 5*pi/3) )
        # Since cos is periodic and even, this simplifies to:
        # h(phi) = max( cos(phi mod pi/3 - pi/6) )  ... let's just compute it.
        
        # Precompute support function for unit hexagon
        def hex_support(phi):
            # phi is the angle of the normal
            # The support function of a regular hexagon with circumradius R=1 is:
            # h(phi) = R * max_{k} cos(phi - k*pi/3)
            # But since the hexagon is convex, h(phi) = max over vertices of dot(v, n)
            # For R=1, vertices are at angles 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
            angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
            return max(math.cos(phi - a) for a in angles)
        
        # For each inner hexagon at (px, py), the maximum projection onto normal (nx, ny) is:
        # px*nx + py*ny + hex_support(atan2(ny, nx))
        
        max_proj = 0.0
        for nx, ny in normals:
            phi = math.atan2(ny, nx)
            h = hex_support(phi)
            # For each inner hexagon
            projs = pts[:, 0] * nx + pts[:, 1] * ny + h
            max_proj = max(max_proj, np.max(projs))
            # Also check the negative direction (but the hexagon is symmetric, so we need to check both)
            # Actually, the support function already accounts for the shape.
            # The constraint is: for all normals n, max_i (dot(p_i, n) + h(n)) <= D
            # where D is the apothem of the outer hexagon.
            # So we need D >= max over all n and all i of (dot(p_i, n) + h(n))
            # And also D >= max over all n and all i of (-dot(p_i, n) + h(-n))
            # But h(-n) = h(n) for a symmetric hexagon, and -dot(p_i, n) = dot(p_i, -n)
            # So we just need to check all 6 normals.
        
        # The outer hexagon has apothem D = side * sqrt(3)/2
        # So side = D * 2 / sqrt(3)
        # But we also need to center the outer hexagon at (cx, cy)
        # The constraint is: max_i (dot(p_i - c, n) + h(n)) <= D
        # So D >= max_i (dot(p_i, n) - dot(c, n) + h(n))
        
        max_proj_centered = 0.0
        for nx, ny in normals:
            dot_c = cx * nx + cy * ny
            phi = math.atan2(ny, nx)
            h = hex_support(phi)
            projs = pts[:, 0] * nx + pts[:, 1] * ny - dot_c + h
            max_proj_centered = max(max_proj_centered, np.max(projs))
        
        # D must be at least max_proj_centered
        # The side length of the outer hexagon is D * 2 / sqrt(3)
        side = max_proj_centered * 2.0 / math.sqrt(3.0)
        
        # Add a small penalty if side is too small (shouldn't happen)
        return side
    
    # Use differential evolution to optimize (cx, cy, theta)
    bounds = [(-5, 5), (-5, 5), (0, math.pi / 6.0)]
    
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=100, tol=1e-8, polish=True)
        cx, cy, theta = result.x
        best_side = result.fun
    except Exception:
        # Fallback to a simple centering
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        theta = 0.0
        best_side = objective([cx, cy, theta])
    
    # Now, let's try to improve by allowing the inner hexagons to rotate.
    # If we rotate each inner hexagon by an angle, the support function changes.
    # For a hexagon rotated by angle alpha, the support function is h(phi - alpha).
    
    # Let's try rotating all inner hexagons by the same angle as the outer hexagon.
    # This might help if the outer hexagon is rotated.
    
    # Actually, let's try a different approach: optimize the inner positions slightly.
    # But this is complex. Let's stick with the current approach.
    
    # Let's refine with Nelder-Mead
    try:
        result2 = minimize(objective, [cx, cy, theta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if result2.fun < best_side:
            cx, cy, theta = result2.x
            best_side = result2.fun
    except Exception:
        pass
    
    # Ensure theta is in [0, pi/6)
    theta = theta % (math.pi / 6.0)
    
    # Compute the final side length
    def hex_support(phi):
        angles = [0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]
        return max(math.cos(phi - a) for a in angles)
    
    normals = []
    for k in range(6):
        ang = theta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = 0.0
    for nx, ny in normals:
        dot_c = cx * nx + cy * ny
        phi = math.atan2(ny, nx)
        h = hex_support(phi)
        projs = pts[:, 0] * nx + pts[:, 1] * ny - dot_c + h
        max_proj = max(max_proj, np.max(projs))
    
    side = max_proj * 2.0 / math.sqrt(3.0)
    
    # Add a tiny safety margin
    side *= (1.0 + 1e-12)
    
    # Prepare the output
    inner = []
    for x, y in pts:
        inner.append([float(x), float(y), 0.0])  # angle in degrees
    
    center = [float(cx), float(cy)]
    rotation = float(math.degrees(theta))
    
    return inner, center, float(side), rotation
