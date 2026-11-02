"""
Packing 12 unit regular hexagons inside a regular hexagon of minimum side length.
Strategy:
1. Use C6 symmetry (2 orbits of 6) to reduce dimensionality.
2. Use a smooth surrogate for the non-overlap constraint and the objective function
   to allow gradient-based optimization (L-BFGS-B).
3. Perform a final high-precision polish using exact geometry checks and local search.
"""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 140.0  # Safety margin before 150s wall-clock kill

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
    
    def get_hex_vertices(cx, cy, angle_deg):
        return get_hex_vertices_fast(cx, cy, angle_deg)

    # Helper: Check if two hexagons are disjoint (Exact SAT)
    def hexagons_disjoint(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        if dist_sq > 4.0: # dist > 2.0
            return True
            
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

    def compute_min_outer_side(inner, center_override=None):
        """
        Compute the minimum side length of a regular hexagon containing all inner hexagons.
        Returns (side, phi_deg, center)
        """
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        if center_override is None:
            try:
                hull = ConvexHull(all_verts)
                hull_verts = all_verts[hull.vertices]
                center = np.mean(hull_verts, axis=0)
            except Exception:
                center = np.mean(all_verts, axis=0)
        else:
            center = np.array(center_override)
            
        offsets = all_verts - center
        
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
                
                proj = offsets[:, 0] * nx + offsets[:, 1] * ny
                max_proj = np.max(np.abs(proj))
                
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

        phi_init = math.radians(best_phi)
        for start_offset in [0, 0.05, -0.05, 0.1, -0.1]:
            start_phi = phi_init + start_offset
            try:
                res = minimize(obj_phi, start_phi, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 200})
                if res.success and res.fun < best_side:
                    best_side = res.fun
                    best_phi = math.degrees(res.x[0]) % 60.0
            except:
                pass

        return best_side, best_phi, center

    # --- SMOOTH OPTIMIZATION STRATEGY ---
    
    def generate_symmetric_inner(params):
        """
        params: [x1, y1, theta1, x2, y2, theta2]
        Generates 12 hexagons with C6 symmetry (2 orbits of 6).
        """
        x1, y1, theta1, x2, y2, theta2 = params
        inner = []
        for k in range(6):
            angle_rot = k * 60.0
            rad_rot = math.radians(angle_rot)
            cr = math.cos(rad_rot)
            sr = math.sin(rad_rot)
            
            # Orbit 1
            nx1 = x1 * cr - y1 * sr
            ny1 = x1 * sr + y1 * cr
            ntheta1 = theta1 + angle_rot
            inner.append([nx1, ny1, ntheta1])
            
            # Orbit 2
            nx2 = x2 * cr - y2 * sr
            ny2 = x2 * sr + y2 * cr
            ntheta2 = theta2 + angle_rot
            inner.append([nx2, ny2, ntheta2])
        return inner

    def smooth_objective(params):
        """
        Smooth objective for gradient-based optimization.
        Uses log-sum-exp for the max extent and a smooth approximation for collision.
        """
        inner = generate_symmetric_inner(params)
        
        # 1. Compute all vertices
        all_verts = []
        for h in inner:
            all_verts.append(get_hex_vertices(h[0], h[1], h[2]))
        all_verts = np.vstack(all_verts)
        
        # Center at origin for symmetry (since we enforce C6, center is 0,0)
        center = np.array([0.0, 0.0])
        offsets = all_verts - center
        
        # 2. Compute max extent over 6 normals (smoothed via log-sum-exp)
        # We don't optimize phi here, we assume optimal phi for this configuration
        # Or we can optimize phi jointly? Let's keep phi fixed at 0 for now in the smooth phase
        # and refine later. Actually, let's include phi in the smooth phase?
        # No, let's stick to 6 params for the inner positions.
        # We can estimate the optimal phi cheaply or just use a fixed set of angles.
        # Let's use the 6 principal axes of the hexagon (0, 30, 60, ... degrees for normals)
        # Wait, the outer hexagon rotation is free.
        # Let's add phi as a 7th parameter to the smooth optimization.
        
        # For now, let's just compute the extent for a few candidate phis and take the min.
        # To make it differentiable, we can use a soft-min.
        # But simpler: just evaluate the exact min-phi side length (non-diff) and add a smooth collision penalty.
        # Gradient descent will struggle with the non-diff side length.
        
        # Alternative: Optimize positions to minimize a proxy for the side length.
        # A good proxy is the maximum distance from the center (radius of circumscribed circle).
        # R = max ||v||. S = R * 2/sqrt(3) * (1/something depending on angle).
        # Actually, for a hexagon, S = R * 2/sqrt(3) if aligned.
        # Let's minimize the max norm of vertices.
        
        norms = np.linalg.norm(all_verts, axis=1)
        # Soft max
        beta = 50.0
        log_max_norm = np.log(np.sum(np.exp(beta * norms))) / beta
        # This is an upper bound on the side length (approx)
        
        # 3. Collision Penalty
        # Approximate collision by checking distance between centers.
        # Min distance is sqrt(3).
        # Penalty = sum( max(0, sqrt(3) - dist)^2 )
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < S3:
                    penalty += (S3 - dist)**2
        
        # Combine
        return log_max_norm + 100.0 * penalty

    # Run smooth optimization
    best_side_global = float('inf')
    best_inner_global = None
    best_phi_global = 0.0
    best_center_global = (0,0)
    
    # Initial guesses
    guesses = [
        [1.5, 0, 0, 3.0, 0, 0],
        [2.0, 0, 0, 2.0, 0, 30],
        [1.0, 0, 0, 2.5, 0, 0],
        [1.2, 0.5, 0, 2.2, 0.5, 0],
        [1.8, 0, 30, 2.8, 0, 30],
    ]
    
    # Add random guesses
    for _ in range(5):
        r1 = random.uniform(1.0, 3.0)
        a1 = random.uniform(0, 60)
        r2 = random.uniform(1.0, 3.0)
        a2 = random.uniform(0, 60)
        t1 = random.uniform(0, 60)
        t2 = random.uniform(0, 60)
        guesses.append([r1*math.cos(math.radians(a1)), r1*math.sin(math.radians(a1)), t1,
                        r2*math.cos(math.radians(a2)), r2*math.sin(math.radians(a2)), t2])

    for g in guesses:
        if time.time() - t0 > DEADLINE - 30:
            break
        try:
            res = minimize(smooth_objective, g, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
            if res.fun < 100: # Heuristic check for validity
                inner = generate_symmetric_inner(res.x)
                if is_valid_configuration(inner):
                    side, phi, center = compute_min_outer_side(inner)
                    if side < best_side_global:
                        best_side_global = side
                        best_inner_global = inner
                        best_phi_global = phi
                        best_center_global = (center[0], center[1])
        except Exception:
            pass

    # If smooth optimization found a valid solution, polish it
    if best_inner_global is not None:
        # Polish with SA or local search
        current_inner = [list(h) for h in best_inner_global]
        current_side, _, _ = compute_min_outer_side(current_inner)
        
        T = 0.1
        T_min = 0.001
        cooling = 0.995
        step_size = 0.1
        
        while time.time() - t0 < DEADLINE - 5:
            idx = random.randint(0, N - 1)
            new_inner = [list(h) for h in current_inner]
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            da = random.gauss(0, 2.0)
            
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
                if new_side < best_side_global:
                    best_side_global = new_side
                    best_inner_global = [list(h) for h in current_inner]
                    best_phi_global = new_phi
                    best_center_global = (new_center[0], new_center[1])
            else:
                delta = new_side - current_side
                if random.random() < math.exp(-delta / T):
                    current_inner = new_inner
                    current_side = new_side
                    if new_side < best_side_global:
                        best_side_global = new_side
                        best_inner_global = [list(h) for h in current_inner]
                        best_phi_global = new_phi
                        best_center_global = (new_center[0], new_center[1])
            
            T *= cooling
            if T < T_min: T = T_min

    # Fallback if no valid solution found
    if best_inner_global is None:
        # Use a simple valid configuration
        # Honeycomb cluster
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
        
        best_inner_global = initial_inner
        best_side_global, best_phi_global, center = compute_min_outer_side(best_inner_global)
        best_center_global = (center[0], center[1])
        
        # Quick SA polish
        current_inner = [list(h) for h in best_inner_global]
        current_side, _, _ = compute_min_outer_side(current_inner)
        
        T = 0.5
        T_min = 0.01
        cooling = 0.999
        step_size = 0.3
        
        while time.time() - t0 < DEADLINE - 5:
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
                if new_side < best_side_global:
                    best_side_global = new_side
                    best_inner_global = [list(h) for h in current_inner]
                    best_phi_global = new_phi
                    best_center_global = (new_center[0], new_center[1])
            else:
                delta = new_side - current_side
                if random.random() < math.exp(-delta / T):
                    current_inner = new_inner
                    current_side = new_side
                    if new_side < best_side_global:
                        best_side_global = new_side
                        best_inner_global = [list(h) for h in current_inner]
                        best_phi_global = new_phi
                        best_center_global = (new_center[0], new_center[1])
            
            T *= cooling
            if T < T_min: T = T_min

    return best_inner_global, (float(best_center_global[0]), float(best_center_global[1])), float(best_side_global), float(best_phi_global)
