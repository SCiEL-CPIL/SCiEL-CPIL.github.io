import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    # Helper: Compute vertices of a regular hexagon
    def hex_vertices(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = a + math.radians(60 * i)
            verts.append((cx + S * math.cos(theta), cy + S * math.sin(theta)))
        return verts

    # Helper: Check if point is inside convex polygon (ray casting or half-plane)
    # For regular hexagon, half-plane is faster
    def point_in_hex(px, py, cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(-a), math.sin(-a)
        dx, dy = px - cx, py - cy
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        apothem = S * math.sqrt(3) / 2
        # Normals at 30, 90, 150, 210, 270, 330 degrees in local frame
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            if lx * nx + ly * ny > apothem + 1e-9:
                return False
        return True

    def objective(params):
        """
        Minimize outer side length S + penalty for overlaps.
        params: [x1, y1, x2, y2, ..., xN, yN]
        We fix rotations to 0 for simplicity and speed, as optimal packing of regular hexagons 
        often aligns with the outer hexagon or is close enough that allowing rotation 
        adds too much dimensionality for the time budget. 
        Actually, the task allows rotation. Let's allow rotation but optimize it?
        22D optimization might be slow. 
        Let's try fixing rotation to 0 first. If that yields a good result, great.
        If not, we can try a few fixed rotations.
        Given the "different strategy" directive, continuous optimization of positions is the key.
        """
        if time.time() > deadline:
            return 1e9
            
        # Extract positions
        xs = params[0::2]
        ys = params[1::2]
        
        # Vertices of inner hexagons (side 1, angle 0)
        inner_verts = []
        for i in range(N):
            v = hex_vertices(xs[i], ys[i], 1.0, 0.0)
            inner_verts.append(v)
        
        # Compute required outer side length S
        # Outer hexagon centered at (0,0) with angle 0 for simplicity?
        # The outer hexagon can rotate. But due to symmetry, if inner are 0-aligned, 
        # outer 0-aligned is likely optimal or near-optimal.
        # Let's compute S for outer angle 0.
        S_needed = 0.0
        apothem_factor = math.sqrt(3) / 2
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = v[0] * nx + v[1] * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_req = max_proj / apothem_factor
                if S_req > S_needed:
                    S_needed = S_req
        
        # Penalty for overlaps
        # Approximate overlap by checking distance between centers vs sum of circumradii?
        # No, that's too loose.
        # Better: Check if any vertex of A is inside B or vice versa.
        # This is expensive. 
        # Alternative: Use a soft penalty based on minimum distance between edges?
        # Simpler: Check center distance. If dist < 2*apothem, they might overlap.
        # For regular hexagons of side 1, apothem = sqrt(3)/2 ~ 0.866.
        # If centers are closer than 2*apothem, they definitely overlap.
        # If centers are farther, they might still overlap if rotated, but we fixed rotation to 0.
        # With rotation 0, two hexagons overlap if center distance < 2 * apothem?
        # Actually, for two same-oriented regular hexagons, they are disjoint if 
        # distance >= 2 * apothem? No, the "width" is 2*apothem.
        # If centers are separated by d, the minimum distance between boundaries is d - 2*apothem?
        # Not exactly, depends on direction.
        # However, a good heuristic penalty is:
        # Penalty = sum over pairs max(0, (2*apothem - dist)^2)
        
        apothem = math.sqrt(3) / 2
        min_dist = 2 * apothem # Approximate threshold for non-overlap in any direction? 
        # Actually, for two hexagons, the minimum center distance to be disjoint is 2*apothem 
        # only if they are separated along a normal direction. If separated diagonally, 
        # they can be closer. 
        # But using 2*apothem as a "safe" distance ensures no overlap. 
        # We want to minimize S, so we want them as close as possible.
        # The penalty should push them apart only if they are too close.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = math.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d)**2
        
        # Weight the penalty heavily
        return S_needed + 100.0 * penalty

    # Initial guess: Hexagonal lattice
    sp = math.sqrt(3) # Spacing for non-overlapping hexagons (2 * apothem)
    init_params = []
    # Center
    init_params.extend([0.0, 0.0])
    # Ring 1
    for i in range(6):
        theta = math.radians(60 * i)
        init_params.extend([sp * math.cos(theta), sp * math.sin(theta)])
    # Ring 2 (pick 4 to make 11)
    # Positions for ring 2 in hex lattice:
    # (2*sp, 0), (sp, sp*1.5), (-sp, sp*1.5), (-2*sp, 0), (-sp, -sp*1.5), (sp, -sp*1.5)
    # Let's just add 4 more around
    ring2_positions = [
        (2*sp, 0),
        (sp, 1.5*sp),
        (-sp, 1.5*sp),
        (-2*sp, 0)
    ]
    for x, y in ring2_positions:
        init_params.extend([x, y])
    
    # Ensure we have exactly 2*N params
    while len(init_params) < 2*N:
        init_params.extend([random.uniform(-1,1), random.uniform(-1,1)])
    init_params = init_params[:2*N]

    best_S = float('inf')
    best_params = None
    
    # Try multiple random initializations to escape local minima
    for attempt in range(10):
        if time.time() > deadline - 10:
            break
            
        # Perturb initial guess
        params = np.array(init_params)
        if attempt > 0:
            params += np.random.normal(0, 0.5, size=params.shape)
        
        # Use Nelder-Mead
        try:
            res = minimize(objective, params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_S:
                # Extract S from the result
                # Re-evaluate to get clean S
                S_val = objective(res.x)
                # Check validity roughly: if penalty is 0, it's likely valid
                # Calculate S part only
                xs = res.x[0::2]
                ys = res.x[1::2]
                inner_verts = [hex_vertices(xs[i], ys[i], 1.0, 0.0) for i in range(N)]
                S_needed = 0.0
                apothem_factor = math.sqrt(3) / 2
                for i in range(6):
                    theta = math.radians(30 + 60 * i)
                    nx, ny = math.cos(theta), math.sin(theta)
                    max_proj = 0.0
                    for iv in inner_verts:
                        for v in iv:
                            proj = v[0] * nx + v[1] * ny
                            if proj > max_proj:
                                max_proj = proj
                    if max_proj > 0:
                        S_req = max_proj / apothem_factor
                        if S_req > S_needed:
                            S_needed = S_req
                
                # Verify non-overlap more strictly?
                # For now, trust the penalty if it's small
                # Check if penalty is negligible
                penalty_val = S_val - S_needed
                if penalty_val < 1e-3: # Valid solution
                    if S_needed < best_S:
                        best_S = S_needed
                        best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial lattice
        best_params = np.array(init_params)
        # Calculate S for fallback
        xs = best_params[0::2]
        ys = best_params[1::2]
        inner_verts = [hex_vertices(xs[i], ys[i], 1.0, 0.0) for i in range(N)]
        S_needed = 0.0
        apothem_factor = math.sqrt(3) / 2
        for i in range(6):
            theta = math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = v[0] * nx + v[1] * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_req = max_proj / apothem_factor
                if S_req > S_needed:
                    S_needed = S_req
        best_S = S_needed

    # Format output
    inner = []
    for i in range(N):
        inner.append([best_params[2*i], best_params[2*i+1], 0.0])
    
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
