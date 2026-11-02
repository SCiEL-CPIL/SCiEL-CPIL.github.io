import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def get_min_outer_config(inner_positions):
        """
        Find the best outer hexagon (center, side, angle) for given inner positions.
        We fix the outer center at (0,0) for the objective calculation, 
        but we must return the correct absolute coordinates.
        Since we enforce symmetry around (0,0) in the parameterization, 
        the centroid of the inner hexagons will be close to (0,0).
        We scan the outer angle to find the minimal enclosing side length.
        """
        if not inner_positions:
            return 0.0, 0.0

        # Collect all vertices
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        
        all_verts = np.vstack(all_verts)
        
        best_s = float('inf')
        best_theta = 0.0
        
        # Scan outer angle from 0 to 60 degrees
        # Use a coarse scan first
        for theta in np.arange(0, 60, 2.0):
            outer_rad = np.radians(theta)
            base_normals = np.radians(np.arange(30, 360, 60))
            normals = base_normals + outer_rad
            
            max_proj = 0.0
            for na in normals:
                nx = np.cos(na)
                ny = np.sin(na)
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            
            s = max_proj / (np.sqrt(3) / 2)
            if s < best_s:
                best_s = s
                best_theta = theta
        
        # Refine theta around best
        for theta in np.arange(best_theta - 2.0, best_theta + 2.0, 0.1):
            outer_rad = np.radians(theta)
            base_normals = np.radians(np.arange(30, 360, 60))
            normals = base_normals + outer_rad
            
            max_proj = 0.0
            for na in normals:
                nx = np.cos(na)
                ny = np.sin(na)
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            
            s = max_proj / (np.sqrt(3) / 2)
            if s < best_s:
                best_s = s
                best_theta = theta
                
        return best_s, best_theta

    def generate_symmetric_positions(params):
        """
        params: [r1, r2, theta, delta_r, delta_a]
        r1: radius of ring 1 (6 hexes)
        r2: radius of ring 2 (5 hexes, but we need 12 total: 1 center, 6 ring 1, 5 ring 2? 
            Wait, 1+6+5=12. 
            Standard hex packing:
            Center: (0,0)
            Ring 1: 6 hexes at distance d1, angles 0, 60, ...
            Ring 2: 12 hexes at distance d2? No, we only have 5 left.
            
            Let's use a more flexible symmetric parameterization.
            Actually, the optimal packing of 12 hexes in a hexagon is likely:
            1 center
            6 in first ring
            5 in second ring (placed in the "gaps" or on the vertices of the second ring)
            
            Let's parameterize:
            p0: center (0,0)
            p1..p6: ring 1, radius r1, angle offset a1
            p7..p11: ring 2, radius r2, angle offset a2 (5 hexes, spaced 72 deg? or specific positions?)
            
            To maintain 6-fold symmetry of the *container*, the *content* doesn't have to be 6-fold symmetric if we rotate the container.
            However, it is highly probable that the optimal packing has 6-fold symmetry or 3-fold.
            Let's assume 6-fold symmetry for the first 7 (1+6) and place the remaining 5 in a pattern that breaks symmetry slightly or fits into a 6-fold symmetric pattern with one "missing" spot?
            
            Actually, a common optimal packing for N=12 in a hexagon is:
            1 center
            6 ring 1
            5 ring 2? 
            Or maybe 1 center, 6 ring 1, 5 ring 2 is not symmetric.
            
            Let's try a different symmetry:
            1 center
            6 ring 1
            5 ring 2 (placed at angles 0, 72, 144, 216, 288? No, that's 5-fold).
            
            Let's look at the "hexagonal lattice" approach.
            If we take a hexagonal lattice of side 1, the centers are at:
            (0,0)
            (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2,0), (-1, -sqrt(3)), (1, -sqrt(3)) -> 6 neighbors.
            Next shell:
            (4,0), (3, sqrt(3)), (2, 2sqrt(3)), (0, 2sqrt(3)), (-2, 2sqrt(3)), (-3, sqrt(3)), (-4,0), (-3, -sqrt(3)), (-2, -2sqrt(3)), (0, -2sqrt(3)), (2, -2sqrt(3)), (3, -sqrt(3)) -> 12 neighbors.
            
            Total 1 + 6 + 12 = 19. We only need 12.
            So we take 1 center, 6 ring 1, and 5 from ring 2.
            Which 5? To minimize the bounding box, we should probably pick 5 that are "balanced".
            But if we pick 5, the symmetry is broken.
            
            Alternative: 
            Maybe the optimal packing is NOT a subset of the standard lattice?
            
            Let's use a parameterization that allows 6-fold symmetry for the whole set if possible, or close to it.
            If we assume 6-fold symmetry, we must have 1 + 6k hexes. 12 is not 1+6k.
            So 6-fold symmetry is impossible for 12 hexes if the center is fixed.
            
            What if we don't fix the center?
            If we have 3-fold symmetry: 12 = 3 * 4. 
            4 hexes per sector.
            
            Let's use a general parameterization with 3-fold symmetry to reduce dimensions.
            Define 4 base hexes in a 120-degree sector.
            Then replicate with 120 and 240 degree rotations.
            Total 12 hexes.
            
            Base hexes (x, y, angle):
            h1, h2, h3, h4.
            
            This gives 12 variables (4 * 3).
            Plus outer angle and center (fixed at 0,0 for optimization, but we can shift later? No, fixing center at 0,0 is fine if the packing is centered).
            
            Let's use this 3-fold symmetry approach.
        """
        # params: [x1, y1, a1, x2, y2, a2, x3, y3, a3, x4, y4, a4]
        # Generate 12 hexes
        positions = []
        
        # Base 4
        base = []
        for i in range(4):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            base.append((x, y, a))
            
        # Replicate with 120 and 240 deg rotation
        for rot_deg in [0, 120, 240]:
            rot_rad = np.radians(rot_deg)
            c = np.cos(rot_rad)
            s = np.sin(rot_rad)
            for (x, y, a) in base:
                # Rotate position
                nx = x * c - y * s
                ny = x * s + y * c
                # Rotate angle
                na = a + rot_deg
                positions.append((nx, ny, na))
                
        return positions

    def objective(params):
        if time.time() > deadline:
            return float('inf')
            
        inner_positions = generate_symmetric_positions(params)
        
        # 1. Calculate outer side length
        s, theta = get_min_outer_config(inner_positions)
        
        # 2. Calculate collision penalty (smooth)
        # For each pair, check if they overlap.
        # Use a soft penalty based on the minimum distance between vertices/edges.
        # Simple approximation: Check distance between centers.
        # If distance < 2.0, they might overlap (if aligned).
        # Better: Use SAT with a penalty for penetration depth.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = inner_positions[i]
                cx2, cy2, a2 = inner_positions[j]
                
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                
                # Quick check: distance between centers
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if dist < 3.5: # Only check if close
                    # SAT check
                    axes = []
                    for k in range(6):
                        p1 = v1[k]
                        p2 = v1[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                    for k in range(6):
                        p1 = v2[k]
                        p2 = v2[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                            
                    overlap = False
                    for axis in axes:
                        proj1 = np.dot(v1, axis)
                        proj2 = np.dot(v2, axis)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                            break # Disjoint
                        else:
                            # Calculate penetration depth on this axis
                            overlap_depth = min(max1, max2) - max(min1, min2)
                            if overlap_depth > 0:
                                penalty += overlap_depth
                                overlap = True
                    # If no break, they overlap
                    # The penalty is accumulated from the axes that didn't separate them?
                    # Actually, if they overlap, all axes show overlap.
                    # The "true" penetration depth is the minimum over all axes of the overlap?
                    # No, the minimum translation vector.
                    # For a smooth penalty, summing the positive overlaps is a decent proxy.
                    
        return s + penalty * 10.0

    # Initial guess for 3-fold symmetric packing
    # 4 hexes in a 120 degree sector.
    # Let's place them roughly in a hex lattice pattern.
    # Center (0,0) is not in the base set if we use 3-fold symmetry for 12 hexes?
    # Wait, if we replicate 4 hexes 3 times, we get 12.
    # None of them is necessarily at (0,0) unless one of the base hexes is at (0,0).
    # If one base hex is at (0,0), its rotations are also (0,0). So we'd have 3 hexes at (0,0) -> Collision!
    # So we cannot have a hex at the center with 3-fold symmetry if we replicate 4 hexes.
    # Unless... we don't use 3-fold symmetry for the center.
    
    # Alternative: 
    # 1 hex at center (0,0).
    # 6 hexes in ring 1 (6-fold symmetric).
    # 5 hexes in ring 2 (no symmetry, or 1-fold).
    # This breaks the low-dimensional parameterization.
    
    # Let's stick to the 3-fold symmetry but ensure no hex is at (0,0).
    # Or, use 6-fold symmetry for 12 hexes? Impossible.
    
    # Let's try a different symmetry: 
    # What if we just use the full 36-dimensional space but with a smarter initialization and Nelder-Mead?
    # Nelder-Mead in 36D is slow.
    
    # Let's go back to the "1 center, 6 ring 1, 5 ring 2" idea but parameterize the 5 ring 2 hexes with 6-fold symmetry *minus one*?
    # No, let's just use a general parameterization but with a very good initial guess and a fast local optimizer.
    
    # Revised Strategy:
    # 1. Start with the known good lattice configuration (1+6+5).
    # 2. Use Nelder-Mead on the 36 parameters.
    # 3. To speed up Nelder-Mead, we can fix the outer angle and center, and just optimize inner positions.
    # 4. But the outer angle is also important.
    
    # Let's try optimizing 36 params with Nelder-Mead. It might be slow but precise.
    # To help it, we can use a multi-start approach.
    
    def objective_full(params):
        if time.time() > deadline:
            return float('inf')
        inner_positions = []
        for i in range(N):
            inner_positions.append((params[i*3], params[i*3+1], params[i*3+2]))
            
        s, theta = get_min_outer_config(inner_positions)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                cx1, cy1, a1 = inner_positions[i]
                cx2, cy2, a2 = inner_positions[j]
                
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                
                dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if dist < 3.5:
                    axes = []
                    for k in range(6):
                        p1 = v1[k]
                        p2 = v1[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                    for k in range(6):
                        p1 = v2[k]
                        p2 = v2[(k+1)%6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-9:
                            axes.append(normal / norm)
                            
                    for axis in axes:
                        proj1 = np.dot(v1, axis)
                        proj2 = np.dot(v2, axis)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                            break
                        else:
                            overlap_depth = min(max1, max2) - max(min1, min2)
                            if overlap_depth > 0:
                                penalty += overlap_depth
                                
        return s + penalty * 10.0

    # Initial configuration: Hexagonal lattice
    init_params = np.zeros(3 * N)
    
    # Center
    init_params[0:3] = [0.0, 0.0, 0.0]
    
    # Ring 1
    for k in range(6):
        angle = np.radians(60 * k)
        idx = (k + 1) * 3
        init_params[idx:idx+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        
    # Ring 2 (5 hexagons)
    # Place them at 120, 180, 240, 300, 0? 
    # Let's place them at angles 0, 72, 144, 216, 288? No, that's 5-fold.
    # Let's place them at the vertices of the second ring that are "opposite" to the gaps.
    # Angles: 30, 90, 150, 210, 270? 
    # Let's just use the lattice positions.
    # Lattice ring 2 positions (12 of them). Pick 5.
    # To be balanced, pick 5 that are roughly evenly spaced.
    angles_ring2 = [0, 72, 144, 216, 288] # 5-fold symmetry for the 5 outer hexes?
    # Radius for ring 2 in lattice is approx 4.0?
    # Distance from center to (2, 2sqrt(3)) is sqrt(4+12)=4.
    # Let's use radius 4.0.
    for k in range(5):
        angle = np.radians(angles_ring2[k])
        idx = (k + 7) * 3
        init_params[idx:idx+3] = [4.0 * np.cos(angle), 4.0 * np.sin(angle), 0.0]
        
    # Optimize
    best_params = init_params.copy()
    best_score = objective_full(best_params)
    
    # Use Nelder-Mead
    # maxiter is critical for time
    max_iter = 200
    opts = {'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
    
    # Run Nelder-Mead
    try:
        res = minimize(objective_full, best_params, method='Nelder-Mead', options=opts)
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # If time permits, try a second restart with a different initial guess
    if time.time() < deadline - 30.0:
        # Perturb initial guess
        init_params2 = init_params + np.random.uniform(-0.5, 0.5, 3*N)
        try:
            res2 = minimize(objective_full, init_params2, method='Nelder-Mead', options=opts)
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Extract best inner positions
    inner = []
    for i in range(N):
        inner.append((best_params[3*i], best_params[3*i+1], best_params[3*i+2]))
        
    # Compute final outer side and angle
    s, theta = get_min_outer_config(inner)
    
    # Center is (0,0)
    center = (0.0, 0.0)
    side = s
    angle = theta
    
    # Validation check
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: expand side slightly
        side *= 1.01
        
    return inner, center, side, angle

# EVOLVE-BLOCK-END
