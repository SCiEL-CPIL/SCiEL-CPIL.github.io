import numpy as np
from scipy.optimize import minimize

N = 12

# Precompute unit hexagon vertices (flat-top)
# Angles: 0, 60, 120, 180, 240, 300
angles_deg = np.arange(0, 360, 60)
angles_rad = np.deg2rad(angles_deg)
unit_vertices = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))

# Outer hexagon normals (flat-top)
# Normals are at 0, 60, ...
outer_normals = unit_vertices # Same angles

def construct():
    # Initial guess: Hexagonal packing
    # Side length of outer hexagon S
    # Inner hexagons side 1
    # Distance between centers in honeycomb is sqrt(3)
    
    # Generate initial positions in a hex grid
    # Center at 0,0
    # Rings:
    # 1 center
    # 6 around
    # 12 around (we need 12 total, so 1 + 6 + 5 from next ring? Or 12 in a specific shape)
    # Let's just place them in a loose grid and let optimizer tighten
    
    centers = []
    # Simple initial layout: 4x3 grid slightly distorted
    # Or better, use the known optimal-ish layout for 12
    # 12 hexagons can fit in a larger hexagon of side 4?
    # Side 4 hexagon contains 37 unit hexagons.
    # Side 3 hexagon contains 19.
    # Side 2 hexagon contains 7.
    # So 12 fits in side 3 easily.
    # Let's start with S = 3.5
    
    S = 3.5
    
    # Initial positions
    # Use a hex grid generator
    def hex_grid_positions(n):
        # Returns positions for n hexagons in a hex grid centered at 0
        # 1, 7, 19, 37...
        # We need 12.
        # Take 1 center, 6 ring 1, 5 from ring 2.
        pos = [(0.0, 0.0)]
        # Ring 1
        for k in range(6):
            angle = np.deg2rad(60 * k)
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            pos.append((x, y))
        # Ring 2 (partial)
        # Directions for ring 2 are same as ring 1 but further?
        # Actually, ring 2 vertices are at distance 2*sqrt(3) in directions 0, 60...
        # And also in between? No, hex grid neighbors are at distance sqrt(3).
        # The 12th and 13th etc are at distance 2*sqrt(3) along the 6 main axes?
        # No, the 7th is center. 8-13 are ring 1.
        # 14-25 are ring 2.
        # Ring 2 positions:
        # 6 at distance 2*sqrt(3) along axes 0, 60...
        # 6 at distance sqrt(3) * sqrt(3) ? No.
        # Let's just use a grid of points.
        
        # Simpler: Just scatter them randomly or in a line and let optimizer fix.
        # But good init helps.
        # Let's use the 1 center + 6 ring 1 + 5 more from ring 2.
        # Ring 2 "corners" are at 2*sqrt(3) distance.
        # Ring 2 "edges" are at sqrt(3) distance from ring 1 vertices?
        
        # Let's just generate a grid of 12 points roughly.
        pass

    # Let's use a specific known good start or random start with multiple restarts?
    # Time limit 120s.
    # Let's try a deterministic init:
    # 4 rows of 3?
    # Row 0: 3 hexes
    # Row 1: 3 hexes
    # Row 2: 3 hexes
    # Row 3: 3 hexes
    # Staggered.
    
    dx = np.sqrt(3) * 1.1 # Slightly more than min distance to avoid initial overlap
    dy = 1.5 * 1.1
    
    centers = []
    for i in range(3):
        for j in range(4):
            x = j * dx + (i % 2) * dx / 2
            y = i * dy
            centers.append([x, y])
            
    # Center the cluster
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [[c[0] - cx, c[1] - cy] for c in centers]
    
    angles = [0.0] * N
    
    # Variables: x0, y0, a0, x1, y1, a1, ...
    x0 = []
    for i in range(N):
        x0.append(centers[i][0])
        x0.append(centers[i][1])
        x0.append(angles[i])
    x0 = np.array(x0)
    
    def unpack(x):
        cs = []
        angs = []
        for i in range(N):
            cs.append((x[3*i], x[3*i+1]))
            angs.append(x[3*i+2])
        return cs, angs

    def penalty(x, S_val):
        cs, angs = unpack(x)
        pen = 0.0
        
        # 1. Non-overlap
        for i in range(N):
            for j in range(i+1, N):
                dx = cs[i][0] - cs[j][0]
                dy = cs[i][1] - cs[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = np.sqrt(3)
                if dist < min_dist:
                    pen += (min_dist - dist)**2
        
        # 2. Containment
        # Outer hexagon side S_val, center 0,0, flat-top
        apothem = S_val * np.sqrt(3) / 2.0
        
        for i in range(N):
            cx, cy = cs[i]
            theta = angs[i]
            
            # Rotate unit vertices by theta
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            for vx, vy in unit_vertices:
                # Local vertex
                # Rotate
                rx = cos_t * vx - sin_t * vy
                ry = sin_t * vx + cos_t * vy
                # Translate
                px = cx + rx
                py = cy + ry
                
                # Check against 6 sides of outer hex
                for nx, ny in outer_normals:
                    # Distance from origin to side is apothem
                    # Point must be inside: p . n <= apothem
                    proj = px * nx + py * ny
                    if proj > apothem:
                        pen += (proj - apothem)**2
        
        return pen

    # Optimization loop
    best_S = 5.0
    current_S = 3.2
    x_current = x0
    
    # We want to minimize S such that penalty is 0.
    # We will iteratively decrease S.
    
    for _ in range(50): # Max iterations
        # Optimize positions for current S
        res = minimize(
            lambda x: penalty(x, current_S),
            x_current,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        x_current = res.x
        min_pen = res.fun
        
        if min_pen < 1e-4:
            # Feasible
            best_S = current_S
            # Try smaller S
            current_S -= 0.05
        else:
            # Not feasible, increase S or break
            if current_S < 3.0:
                break
            current_S += 0.1
            # If we increased S, we should probably keep the old x_current or re-optimize?
            # Let's just break if we can't find a solution at a reasonable size
            # Or restart with better init?
            break
            
    # Final check and refinement
    # Re-optimize at best_S to get the best positions
    res_final = minimize(
        lambda x: penalty(x, best_S),
        x_current,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-10}
    )
    
    cs_final, angs_final = unpack(res_final.x)
    
    inner = []
    for i in range(N):
        inner.append([cs_final[i][0], cs_final[i][1], np.rad2deg(angs_final[i])])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
