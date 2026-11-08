"""Hexagonal packing: place unit hexagons on a honeycomb lattice inside a regular hexagon."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Unit regular hexagon vertices (flat-top orientation, angle 0)
    # A regular hexagon with side length s has vertices at distance s from center
    # For flat-top: angles 0, 60, 120, 180, 240, 300 degrees
    # For pointy-top: angles 30, 90, 150, 210, 270, 330 degrees
    
    # We'll use pointy-top orientation for the inner hexagons (default angle=0 in task means pointy-top? 
    # Let's think: the task says angle_degrees for each inner hexagon. 
    # Standard: if angle=0, let's define vertices at 0, 60, 120, 180, 240, 300 (flat-top)
    # Actually, let's be consistent. A regular hexagon with side length 1 and rotation angle θ:
    # vertices at (cos(θ + k*60°), sin(θ + k*60°)) for k=0..5, scaled by side_length
    # Since it's regular, circumradius = side length = 1
    
    # Honeycomb lattice for unit hexagons:
    # Centers of adjacent hexagons in a honeycomb are separated by distance 2*apothem = 2*(sqrt(3)/2) = sqrt(3)
    # The lattice vectors for a honeycomb of flat-top hexagons:
    #   v1 = (3, 0)  -- wait, let me recalculate
    # For flat-top hexagons with side length s=1:
    #   width = 2*s = 2, height = sqrt(3)*s = sqrt(3)
    #   Horizontal spacing between columns: 3/2 * s = 1.5
    #   Vertical spacing: sqrt(3) * s = sqrt(3)
    #   Odd columns shifted by sqrt(3)/2
    
    # Actually, for perfect tiling, the center-to-center distance between adjacent hexagons is 2*apothem = sqrt(3)*s
    # For s=1, that's sqrt(3).
    
    # Let's use axial coordinates for the hexagonal grid.
    # Place N hexagons in a hexagonal arrangement (like a hexagon of hexagons).
    # For N hexagons arranged in a hexagonal shape, we need N = 3k(k-1)+1 for some k, or we use a different count.
    # N=11 doesn't form a perfect hexagonal number. Let's just place them in a compact hexagonal cluster.
    
    # Generate positions on a hexagonal lattice
    # Using cube coordinates (x, y, z) with x+y+z=0
    # Convert to pixel coordinates for flat-top hexagons:
    #   px = 1.5 * x
    #   py = sqrt(3) * (y + x/2)
    
    # For a hexagonal cluster of radius r (in cube coords), we have hexagons where max(|x|,|y|,|z|) <= r
    # Number of hexagons = 3r(r+1)+1
    # r=0: 1, r=1: 7, r=2: 19, r=3: 37
    
    # N=11, so we can't make a perfect hexagonal cluster. Let's place them in a compact arrangement.
    # We'll use a hexagonal grid and pick the 11 closest to origin.
    
    # Generate candidate positions on hexagonal lattice
    # For flat-top hexagons, the lattice points (centers) are at:
    # For column i (can be negative), row j:
    #   cx = 1.5 * i
    #   cy = sqrt(3) * (j + (i & 1) / 2.0) if i is odd, else sqrt(3) * j
    # Wait, let me use a cleaner formulation.
    
    # Axial coordinate (q, r) to pixel for pointy-top:
    #   x = sqrt(3) * (q + r/2)
    #   y = 3/2 * r
    # For flat-top:
    #   x = 3/2 * q
    #   y = sqrt(3) * (r + q/2)
    
    # Let's use flat-top orientation for inner hexagons (angle=0).
    # Lattice vectors:
    #   e1 = (3/2, sqrt(3)/2)  -- this is wrong for flat-top
    
    # Let me just directly compute. For flat-top hexagons with side length 1:
    # The distance from center to a flat edge (apothem) = sqrt(3)/2
    # Adjacent hexagons share an edge, so center-to-center = 2 * apothem = sqrt(3)
    
    # The 6 neighbors of a hexagon at origin in flat-top orientation are at:
    #   (sqrt(3), 0), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), (-sqrt(3), 0), (-sqrt(3)/2, -3/2), (sqrt(3)/2, -3/2)
    # Wait, that doesn't look right either. Let me think more carefully.
    
    # For a flat-top hexagon, the 6 edges are horizontal and at 60-degree angles.
    # The normals to the edges point at angles 0, 60, 120, 180, 240, 300.
    # So neighbors are at distance sqrt(3) in those directions:
    #   (sqrt(3)*cos(0), sqrt(3)*sin(0)) = (sqrt(3), 0)
    #   (sqrt(3)*cos(60), sqrt(3)*sin(60)) = (sqrt(3)/2, 3/2)
    #   etc.
    
    # So the lattice basis vectors can be:
    #   v1 = (sqrt(3), 0)
    #   v2 = (sqrt(3)/2, 3/2)
    
    # Any center position: p = a*v1 + b*v2 for integers a, b
    
    sqrt3 = np.sqrt(3.0)
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3 / 2.0, 1.5])
    
    # Generate all lattice points within some radius, sort by distance from origin, take first N
    # Max radius to search: we need at least N points, so search up to radius ~ sqrt(N) * max_lattice_spacing
    # Let's search up to radius 10 (in lattice coords, so actual distance up to ~10*sqrt(3))
    
    candidates = []
    R = 10  # search radius in lattice indices
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            p = a * v1 + b * v2
            d = np.linalg.norm(p)
            candidates.append((d, a, b, p))
    
    candidates.sort(key=lambda x: x[0])
    
    # Take the first N candidates (closest to origin)
    selected = candidates[:N]
    
    # Place inner hexagons at these positions with angle 0 (flat-top)
    inner = []
    for d, a, b, p in selected:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    # Now compute the minimal enclosing regular hexagon
    # The outer hexagon has center (cx, cy), side length S, and rotation angle phi
    # We want to minimize S such that all inner hexagon vertices are inside the outer hexagon
    
    # A point is inside a regular hexagon with center c, side length S, rotation phi if:
    # For each of the 6 edge normals at angles phi + k*60, the projection of (p-c) onto the normal <= S * apothem_outer
    # where apothem_outer = S * sqrt(3)/2
    
    # So we need: for all k=0..5, for all inner vertices p_i:
    #   dot(p_i - c, n_k) <= S * sqrt(3)/2
    # where n_k = (cos(phi + k*60), sin(phi + k*60))
    
    # To find the minimal S for a given c and phi, we need:
    #   S >= max over all k and all inner vertices p_i of dot(p_i - c, n_k) / (sqrt(3)/2)
    
    # We can optimize over c and phi. Let's try a few strategies:
    # 1. Set c = centroid of all inner hexagon vertices
    # 2. Try different phi values (0, 30, 60, etc.)
    # 3. For each phi, compute optimal c by solving a linear programming-like problem, or just use centroid
    
    # First, get all vertices of all inner hexagons
    all_vertices = []
    for x, y, ang in inner:
        for k in range(6):
            theta = np.radians(ang + k * 60.0)
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Centroid
    centroid = np.mean(all_vertices, axis=0)
    
    best_S = float('inf')
    best_c = centroid
    best_phi = 0.0
    
    # Try different rotations for the outer hexagon
    for phi_deg in range(0, 360, 15):
        phi = np.radians(phi_deg)
        normals = []
        for k in range(6):
            angle = phi + np.radians(k * 60.0)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)
        
        # For a given center c, the required S is:
        # S = max over k of (max over i of dot(v_i - c, n_k)) / (sqrt(3)/2)
        # = (2/sqrt(3)) * max_k (max_i dot(v_i, n_k) - dot(c, n_k))
        
        # To minimize S, we want to choose c to balance the max projections.
        # This is equivalent to finding the smallest hexagon (with fixed orientation) enclosing the points.
        # The optimal c is such that the max projection in each normal direction is balanced.
        
        # For a regular hexagon, the optimal center is the centroid of the vertices of the convex hull
        # projected appropriately, but a good approximation is to use the centroid of all points.
        
        c = centroid
        max_proj = 0.0
        for n in normals:
            projs = np.dot(all_vertices - c, n)
            max_proj = max(max_proj, np.max(projs))
        
        S = (2.0 / sqrt3) * max_proj
        
        if S < best_S:
            best_S = S
            best_c = c
            best_phi = phi_deg
    
    # Also try with the outer hexagon rotated to align with the lattice
    # The lattice has 6-fold symmetry, so the outer hexagon should align with it
    # Let's also try phi = 0, 30, 60, 90, 120, 150
    
    # Refine: for each candidate phi, optimize c using a simple approach
    # The optimal c minimizes max_k (max_i dot(v_i, n_k) - dot(c, n_k))
    # This is a convex optimization problem. We can use scipy if available, or just try a grid.
    
    # For now, let's also try centering on the centroid of the hexagon centers
    center_centroid = np.mean(np.array([[x, y] for x, y, a in inner]), axis=0)
    
    for phi_deg in [0, 30, 60, 90, 120, 150]:
        phi = np.radians(phi_deg)
        normals = []
        for k in range(6):
            angle = phi + np.radians(k * 60.0)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        normals = np.array(normals)
        
        for c in [centroid, center_centroid]:
            max_proj = 0.0
            for n in normals:
                projs = np.dot(all_vertices - c, n)
                max_proj = max(max_proj, np.max(projs))
            
            S = (2.0 / sqrt3) * max_proj
            
            if S < best_S:
                best_S = S
                best_c = c
                best_phi = phi_deg
    
    # Check deadline
    if time.time() > deadline:
        # Return a safe default
        return inner, [0.0, 0.0], 40.0, 0.0
    
    return inner, [float(best_c[0]), float(best_c[1])], float(best_S), float(best_phi)
# EVOLVE-BLOCK-END
