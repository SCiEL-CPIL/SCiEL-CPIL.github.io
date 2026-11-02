"""
Caging formulation: Optimize inner hexagon positions via Simulated Annealing 
to minimize the side length of the minimal containing regular hexagon.
The outer hexagon is optimized analytically for each inner configuration 
by scanning its rotation angle to minimize the maximum extent of the 
inner convex hull along the outer hexagon's face normals.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 135.0  # Safety margin before 150s wall-clock kill

# Precompute constants for hexagon geometry
S3 = math.sqrt(3)
COS60 = 0.5
SIN60 = S3 / 2.0
PI = math.pi
TWO_PI = 2 * PI

def get_hex_vertices_fast(cx, cy, angle_deg):
    """Fast computation of hexagon vertices using precomputed trig values."""
    a = math.radians(angle_deg)
    ca = math.cos(a)
    sa = math.sin(a)
    
    # Base vertices at angles 0, 60, 120, 180, 240, 300
    # (1, 0), (0.5, S3/2), (-0.5, S3/2), (-1, 0), (-0.5, -S3/2), (0.5, -S3/2)
    # Rotated by a:
    # x' = x*ca - y*sa
    # y' = x*sa + y*ca
    
    v = np.empty((6, 2))
    # i=0: (1, 0) -> (ca, sa)
    v[0, 0] = cx + ca
    v[0, 1] = cy + sa
    # i=1: (0.5, S3/2) -> (0.5*ca - S3/2*sa, 0.5*sa + S3/2*ca)
    v[1, 0] = cx + 0.5*ca - SIN60*sa
    v[1, 1] = cy + 0.5*sa + SIN60*ca
    # i=2: (-0.5, S3/2) -> (-0.5*ca - S3/2*sa, -0.5*sa + S3/2*ca)
    v[2, 0] = cx - 0.5*ca - SIN60*sa
    v[2, 1] = cy - 0.5*sa + SIN60*ca
    # i=3: (-1, 0) -> (-ca, -sa)
    v[3, 0] = cx - ca
    v[3, 1] = cy - sa
    # i=4: (-0.5, -S3/2) -> (-0.5*ca + S3/2*sa, -0.5*sa - S3/2*ca)
    v[4, 0] = cx - 0.5*ca + SIN60*sa
    v[4, 1] = cy - 0.5*sa - SIN60*ca
    # i=5: (0.5, -S3/2) -> (0.5*ca + S3/2*sa, 0.5*sa - S3/2*ca)
    v[5, 0] = cx + 0.5*ca + SIN60*sa
    v[5, 1] = cy + 0.5*sa - SIN60*ca
    
    return v

def construct():
    t0 = time.time()
    
    # Helper: Get vertices of a regular hexagon centered at (cx, cy) with side 1 and angle deg
    def get_hex_vertices(cx, cy, angle_deg):
        return get_hex_vertices_fast(cx, cy, angle_deg)

    # Helper: Check if two hexagons are disjoint
    def hexagons_disjoint(h1, h2):
        # h1, h2 are (x, y, angle)
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If far apart, disjoint
        if dist_sq > 4.0: # dist > 2.0
            return True
            
        # Get vertices
        v1 = get_hex_vertices(h1[0], h1[1], h1[2])
        v2 = get_hex_vertices(h2[0], h2[1], h2[2])
        
        # SAT: Check 6 normals from each hexagon
        for i in range(6):
            ex = v1[(i+1)%6, 0] - v1[i, 0]
            ey = v1[(i+1)%6, 1] - v1[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        for i in range(6):
            ex = v2[(i+1)%6, 0] - v2[i, 0]
            ey = v2[(i+1)%6, 1] - v2[i, 1]
            nx = -ey
            ny = ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm == 0: continue
            nx /= norm
            ny /= norm
            
            proj1_min = np.min(v1[:, 0]*nx + v1[:, 1]*ny)
            proj1_max = np.max(v1[:, 0]*nx + v1[:, 1]*ny)
            proj2_min = np.min(v2[:, 0]*nx + v2[:, 1]*ny)
            proj2_max = np.max(v2[:, 0]*nx + v2[:, 1]*ny)
            
            if proj1_max < proj2_min or proj2_max < proj1_min:
                return True
                
        return False

    def is_valid_configuration(inner):
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    return False
        return True

    def compute_min_outer_side(inner):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        We optimize the rotation angle of the outer hexagon.
        The outer hexagon is centered at the centroid of the convex hull of the inner configuration.
        """
        # 1. Get all vertices of inner hexagons
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # 2. Compute centroid of the convex hull (more robust center)
        try:
            hull = ConvexHull(all_verts)
            hull_verts = all_verts[hull.vertices]
            center = np.mean(hull_verts, axis=0)
        except Exception:
            center = np.mean(all_verts, axis=0)
            
        offsets = all_verts - center
        
        # 3. Find optimal rotation angle phi for outer hexagon
        # Outer hexagon normals are at angles phi + 30 + k*60
        # Required side S = max_k ( 2/sqrt(3) * max_v | n_k . offset_v | )
        # We scan phi in [0, 60)
        
        sqrt3 = S3
        best_side = float('inf')
        best_phi = 0.0
        
        # Coarse scan
        for phi_deg in range(0, 60, 2):
            phi = math.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project all offsets
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                # S = 2/sqrt(3) * max_proj
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg
                
        # Fine scan around best_phi
        for phi_deg in np.arange(best_phi - 2, best_phi + 2, 0.1):
            if phi_deg < 0 or phi_deg >= 60: continue
            phi = math.radians(phi_deg)
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            
            if max_extent < best_side:
                best_side = max_extent
                best_phi = phi_deg

        # Polish: Use scipy minimize on the phi angle to refine
        def obj_phi(phi_rad):
            max_extent = 0.0
            for k in range(6):
                normal_angle = phi_rad + PI/6 + k * PI/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                side_k = (2.0 / sqrt3) * max_proj
                if side_k > max_extent:
                    max_extent = side_k
            return max_extent

        # Start from best_phi found in scan
        phi_init = math.radians(best_phi)
        # Try a few local optimizations from nearby starts to avoid local minima
        for start_offset in [0, 0.1, -0.1, 0.2, -0.2]:
            start_phi = phi_init + start_offset
            try:
                res = minimize(obj_phi, start_phi, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 100})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = math.degrees(res.x[0]) % 60.0
            except:
                pass

        return best_side, best_phi, center

    # --- NEW STRATEGY: C6 Symmetric Optimization ---
    
    def generate_symmetric_inner(params):
        """
        params: [x1, y1, theta1, x2, y2, theta2]
        Generates 12 hexagons with C6 symmetry (2 orbits of 6).
        """
        x1, y1, theta1, x2, y2, theta2 = params
        inner = []
        for k in range(6):
            angle_rot = k * 60.0
            # Rotate point (x1, y1) by angle_rot
            rad_rot = math.radians(angle_rot)
            cr = math.cos(rad_rot)
            sr = math.sin(rad_rot)
            nx1 = x1 * cr - y1 * sr
            ny1 = x1 * sr + y1 * cr
            ntheta1 = theta1 + angle_rot
            
            # Rotate point (x2, y2) by angle_rot
            nx2 = x2 * cr - y2 * sr
            ny2 = x2 * sr + y2 * cr
            ntheta2 = theta2 + angle_rot
            
            inner.append([nx1, ny1, ntheta1])
            inner.append([nx2, ny2, ntheta2])
        return inner

    def objective_symmetric(params):
        """
        Objective function for the symmetric optimization.
        Returns the side length of the containing hexagon, plus a penalty for overlaps.
        """
        inner = generate_symmetric_inner(params)
        
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint(inner[i], inner[j]):
                    # Estimate penetration
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Minimum distance for disjoint hexagons is sqrt(3)
                    if dist < S3:
                        overlap_penalty += (S3 - dist)**2
        
        if overlap_penalty > 0:
            # Return large value if overlapping
            return 100.0 + 10.0 * overlap_penalty
        
        # Compute min outer side
        side, phi, center = compute_min_outer_side(inner)
        return side

    # Initial guess for symmetric search
    # Try a configuration where the 12 hexagons form a ring or two rings
    # A good starting point might be derived from a honeycomb cluster
    # Let's try placing them in two hexagonal rings
    # Ring 1: 6 hexagons around a central void? No, we have 12.
    # Maybe 6 in one ring, 6 in another?
    
    # Let's try a few random restarts within the symmetric space
    best_side_sym = float('inf')
    best_params_sym = None
    best_inner_sym = None
    best_phi_sym = 0.0
    best_center_sym = (0,0)
    
    # Time budget for symmetric optimization
    t_sym_start = time.time()
    sym_time_budget = 60.0
    
    # Define a helper to run the optimizer
    def run_sym_optimizer(initial_params):
        try:
            # Use Powell or Nelder-Mead
            # Powell is often better for this kind of non-smooth problem (max function)
            res = minimize(objective_symmetric, initial_params, method='Powell', 
                           options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_side_sym:
                return res.x, res.fun
        except:
            pass
        return None, float('inf')

    # Initial guesses
    # Guess 1: Two rings of 6
    # Radius 1.5 and 3.0?
    guesses = [
        [1.5, 0, 0, 3.0, 0, 0],
        [2.0, 0, 0, 2.0, 0, 30],
        [1.0, 0, 0, 2.5, 0, 0],
        [0.0, 0, 0, 2.0, 0, 0], # One at center? No, C6 symmetry means if one is at center, it's its own orbit. 
                                # But we defined 2 orbits. If x1=y1=0, orbit 1 is just 1 hexagon (repeated 6 times? No, it's the same hexagon 6 times, which is invalid/overlapping).
                                # So we must ensure the orbits are distinct and non-overlapping.
                                # Actually, if x1=y1=0, the 6 copies are identical, so they overlap. 
                                # So we should avoid centers in the orbits unless we handle it specially.
                                # Let's assume non-zero radii.
    ]
    
    # Add some random guesses
    for _ in range(10):
        r1 = random.uniform(1.0, 3.0)
        a1 = random.uniform(0, 60)
        r2 = random.uniform(1.0, 3.0)
        a2 = random.uniform(0, 60)
        t1 = random.uniform(0, 60)
        t2 = random.uniform(0, 60)
        guesses.append([r1*math.cos(math.radians(a1)), r1*math.sin(math.radians(a1)), t1,
                        r2*math.cos(math.radians(a2)), r2*math.sin(math.radians(a2)), t2])

    for g in guesses:
        if time.time() - t_sym_start > sym_time_budget:
            break
        params, val = run_sym_optimizer(g)
        if params is not None and val < best_side_sym:
            best_side_sym = val
            best_params_sym = params
            # Recover best inner, phi, center
            inner = generate_symmetric_inner(params)
            side, phi, center = compute_min_outer_side(inner)
            if side < best_side_sym: # Use the computed side, not the objective which might have penalty
                best_side_sym = side
                best_inner_sym = inner
                best_phi_sym = phi
                best_center_sym = (center[0], center[1])

    # If symmetric search found something good, use it
    if best_inner_sym is not None and best_side_sym < float('inf'):
         # We can try to polish this specific configuration with the original SA logic if time permits
         # But for now, return this result.
         # Let's do a quick polish with SA on the 12 independent variables starting from this symmetric config
         # This breaks the symmetry but might find a slightly better non-symmetric solution.
         
         if time.time() - t0 < DEADLINE - 30:
             current_inner = [list(h) for h in best_inner_sym]
             current_side, _, _ = compute_min_outer_side(current_inner)
             
             T = 0.5
             T_min = 0.01
             cooling = 0.999
             step_size = 0.3
             
             while time.time() - t0 < DEADLINE - 10:
                 idx = random.randint(0, N - 1)
                 new_inner = [list(h) for h in current_inner]
                 dx = random.gauss(0, step_size)
                 dy = random.gauss(0, step_size)
                 da = random.gauss(0, 5.0)
                 
                 new_inner[idx][0] += dx
                 new_inner[idx][1] += dy
                 new_inner[idx][2] += da
                 
                 valid = True
                 for j in range(N):
                     if j == idx: continue
                     if not hexagons_disjoint(new_inner[idx], new_inner[j]):
                         valid = False
                         break
                 
                 if not valid:
                     T *= cooling
                     if T < T_min: T = T_min
                     continue
                     
                 new_side, new_phi, new_center = compute_min_outer_side(new_inner)
                 
                 if new_side < current_side:
                     current_inner = new_inner
                     current_side = new_side
                     if new_side < best_side_sym:
                         best_side_sym = new_side
                         best_inner_sym = [list(h) for h in current_inner]
                         best_phi_sym = new_phi
                         best_center_sym = (new_center[0], new_center[1])
                 else:
                     delta = new_side - current_side
                     if delta < 0: delta = 0
                     if random.random() < math.exp(-delta / T):
                         current_inner = new_inner
                         current_side = new_side
                         if new_side < best_side_sym:
                             best_side_sym = new_side
                             best_inner_sym = [list(h) for h in current_inner]
                             best_phi_sym = new_phi
                             best_center_sym = (new_center[0], new_center[1])
                 
                 T *= cooling
                 if T < T_min: T = T_min

         return best_inner_sym, best_center_sym, float(best_side_sym), float(best_phi_sym)

    # Fallback to original SA if symmetric search failed or didn't find a valid solution
    # ... (Keep the original SA code as a fallback) ...
    
    # Initial Configuration: Honeycomb Cluster
    initial_inner = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5 + (r % 2) * 0.75
            y = r * S3
            initial_inner.append([x, y, 0.0])
            
    cx = np.mean([h[0] for h in initial_inner])
    cy = np.mean([h[1] for h in initial_inner])
    for h in initial_inner:
        h[0] -= cx
        h[1] -= cy

    best_inner = [list(h) for h in initial_inner]
    best_side, best_phi, best_center = compute_min_outer_side(best_inner)
    
    current_inner = [list(h) for h in initial_inner]
    current_side, _, _ = compute_min_outer_side(current_inner)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    step_size = 0.5
    
    step_count = 0
    max_steps = 20000 # Hard limit
    
    while time.time() - t0 < DEADLINE and step_count < max_steps:
        step_count += 1
        
        idx = random.randint(0, N - 1)
        new_inner = [list(h) for h in current_inner]
        
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 10.0)
        
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        valid = True
        for j in range(N):
            if j == idx: continue
            if not hexagons_disjoint(new_inner[idx], new_inner[j]):
                valid = False
                break
                
        if not valid:
            T *= cooling
            if T < T_min: T = T_min
            continue
            
        new_side, new_phi, new_center = compute_min_outer_side(new_inner)
        
        if new_side < current_side:
            current_inner = new_inner
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_inner = [list(h) for h in current_inner]
                best_phi = new_phi
                best_center = new_center
        else:
            delta = new_side - current_side
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_inner = new_inner
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_inner = [list(h) for h in current_inner]
                    best_phi = new_phi
                    best_center = new_center
                    
        T *= cooling
        if T < T_min: T = T_min

    return best_inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_phi)
