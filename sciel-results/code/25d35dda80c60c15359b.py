import math
import time
import random
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

    # --- Geometry Helpers ---

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon as a numpy array (6, 2)."""
        angle = math.radians(angle_deg)
        vertices = np.zeros((6, 2))
        for i in range(6):
            theta = angle + i * math.pi / 3
            vertices[i, 0] = cx + side * math.cos(theta)
            vertices[i, 1] = cy + side * math.sin(theta)
        return vertices

    def sat_min_separation(verts1, verts2):
        """
        Calculate the minimum separation between two convex polygons (hexagons).
        Positive means disjoint, negative means overlapping.
        Uses Separating Axis Theorem (SAT) axes from both polygons.
        """
        min_sep = float('inf')
        axes = []
        
        # Collect axes from verts1
        for i in range(6):
            p1 = verts1[i]
            p2 = verts1[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            norm = np.linalg.norm([nx, ny])
            if norm > 1e-10:
                axes.append((nx/norm, ny/norm))
        
        # Collect axes from verts2
        for i in range(6):
            p1 = verts2[i]
            p2 = verts2[(i + 1) % 6]
            edge = p2 - p1
            nx, ny = -edge[1], edge[0]
            norm = np.linalg.norm([nx, ny])
            if norm > 1e-10:
                axes.append((nx/norm, ny/norm))
        
        for nx, ny in axes:
            # Project verts1
            proj1 = verts1[:, 0] * nx + verts1[:, 1] * ny
            min1, max1 = np.min(proj1), np.max(proj1)
            # Project verts2
            proj2 = verts2[:, 0] * nx + verts2[:, 1] * ny
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Calculate separation on this axis
            # If max1 < min2, sep = min2 - max1
            # If max2 < min1, sep = min1 - max2
            # Else sep = 0 (overlap on this axis)
            if max1 < min2:
                sep = min2 - max1
            elif max2 < min1:
                sep = min1 - max2
            else:
                sep = 0.0
            
            if sep < min_sep:
                min_sep = sep
                
        return min_sep

    def is_valid_config(centers, angles, eps=1e-7):
        """Check if all hexagons are disjoint."""
        verts = []
        for i in range(len(centers)):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                sep = sat_min_separation(verts[i], verts[j])
                if sep < -eps:  # Overlap
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """
        Compute the exact minimum side length of the outer hexagon.
        The outer hexagon is centered at (ocx, ocy) with rotation otheta.
        Side length s is determined by the maximum projection of any inner vertex
        onto the 6 normal directions of the outer hexagon.
        For a regular hexagon of side s, the distance from center to a side (apothem) is s * sqrt(3)/2.
        So, s = 2 * max_proj / sqrt(3).
        """
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        
        # Precompute inner vertices
        all_verts = []
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            for vi in range(6):
                vert_angle = angle_rad + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                all_verts.append((vx, vy))
        
        for k in range(6):
            # Normals of the outer hexagon
            # Vertices are at angle + i*pi/3. Sides are perpendicular to these.
            # The normal pointing "out" for side k is at angle otheta + pi/6 + k*pi/3
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            for vx, vy in all_verts:
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
                    
        return 2.0 * max_proj / math.sqrt(3)

    # --- Optimization Objective ---

    def objective(params):
        """
        Objective function for L-BFGS-B.
        Returns a smooth approximation of the outer side length + penalty for overlaps.
        """
        n = 12
        # Unpack parameters
        centers = np.zeros((n, 2))
        angles = np.zeros(n)
        for i in range(n):
            centers[i, 0] = params[i*2]
            centers[i, 1] = params[i*2+1]
            angles[i] = params[24+i]
        
        outer_cx = params[36]
        outer_cy = params[37]
        outer_theta = params[38]
        
        # 1. Compute Smooth Outer Side (Soft Max)
        # We want to minimize the max projection.
        # Projections are linear in params, but max is non-smooth.
        # Use log-sum-exp for a smooth approximation of max.
        # f(x) = (1/beta) * log(sum(exp(beta * x_i))) approximates max(x_i) as beta->inf.
        
        beta_side = 100.0 # High beta for sharpness, but stable enough
        
        otheta_rad = math.radians(outer_theta)
        
        # Collect all projections onto the 6 outer normals
        # There are 12 hexagons * 6 vertices = 72 vertices.
        # 6 normals. Total 432 projections.
        # To save time, we can just compute the max per normal direction and then soft-max those 6?
        # No, the max is over ALL vertices for EACH normal. The side length is determined by the MAX over all normals of (Max over vertices).
        # Actually, side length = 2/sqrt(3) * max_{k} ( max_{v} proj(v, n_k) ).
        # This is max over 6 values. Each value is a max over 72 vertices.
        # We can approximate the inner max with a soft-max over vertices, and outer max with soft-max over normals.
        
        # For efficiency in the objective, let's compute the exact max for the 6 normals, 
        # but we need gradients. L-BFGS-B uses numerical or analytical gradients. 
        # If we don't provide gradients, it uses numerical, which is slow (39 params * 2 evals).
        # Better to provide a smooth function that is differentiable.
        
        # Let's use a simpler smooth approximation for the side length:
        # Sum of exp of the top few projections? Or just soft-max of the 6 "axis maxes".
        
        axis_maxes = np.zeros(6)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # Compute projections of all vertices onto this normal
            projs = np.zeros(72)
            idx = 0
            for i in range(n):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    projs[idx] = (vx - outer_cx) * nx + (vy - outer_cy) * ny
                    idx += 1
            
            # Soft max of these 72 projections
            m = np.max(projs)
            if m > 1e-10:
                axis_maxes[k] = m + (1.0/beta_side) * np.log(np.sum(np.exp(projs - m)))
            else:
                axis_maxes[k] = 0.0 # If all negative, max is 0? No, max could be negative.
                # Actually, if all projs are negative, the hexagon is on the "negative" side.
                # But we are looking for the extent in the positive normal direction.
                # If all are negative, the max is the least negative one.
                # log-sum-exp handles this.
                # If m is very small, exp(projs-m) is fine.
                # Let's just use the formula always.
                # Correction: if m is -100, exp(-100 - (-100)) = 1. Sum=72. log(72)/beta.
                # axis_max = -100 + small. This is correct.
        
        # Now soft max of the 6 axis_maxes
        m_axis = np.max(axis_maxes)
        smooth_max_proj = m_axis + (1.0/beta_side) * np.log(np.sum(np.exp(axis_maxes - m_axis)))
        
        side_length = 2.0 * smooth_max_proj / math.sqrt(3)
        
        # 2. Compute Overlap Penalty
        # We want to penalize if any pair overlaps.
        # Separation sep_ij. If sep_ij < 0, penalty.
        # Smooth hinge: max(0, -sep)^2 ?
        # Or smooth approximation: 0.5 * ( -sep + sqrt(sep^2 + eps^2) )^2 ?
        
        penalty = 0.0
        beta_pen = 50.0
        eps_pen = 0.01
        
        for i in range(n):
            for j in range(i+1, n):
                v1 = get_hex_vertices(centers[i,0], centers[i,1], 1.0, angles[i])
                v2 = get_hex_vertices(centers[j,0], centers[j,1], 1.0, angles[j])
                sep = sat_min_separation(v1, v2)
                
                # If sep < 0, they overlap.
                # We want a smooth function that is 0 when sep >= 0 and grows when sep < 0.
                # Let g(sep) = max(0, -sep).
                # Smooth approx: h(sep) = 0.5 * ( -sep + sqrt(sep^2 + 1/beta^2) )
                # This is close to max(0, -sep).
                # We want to minimize h(sep)^2.
                
                # Actually, simpler: just use a quadratic penalty on the deficit.
                # deficit = max(0, eps - sep).
                # Smooth deficit: d = 0.5 * ( (eps - sep) + sqrt((eps-sep)^2 + 1/beta^2) )
                # Penalty += d^2
                
                # For speed, we can skip detailed calc if sep is already large positive.
                if sep > 0.1:
                    continue
                
                # Calculate smooth deficit
                x = eps_pen - sep
                d = 0.5 * (x + math.sqrt(x*x + 1.0/(beta_pen*beta_pen)))
                penalty += d * d
                
        # Scale penalty to be comparable to side length (~5.0)
        # Side length is ~5. Penalty should dominate if invalid.
        return side_length + 10.0 * penalty

    # --- Initial Configurations ---

    def generate_hexagonal_cluster():
        """
        12 hexagons in a 3x4 grid is not hexagonal.
        A hexagonal cluster of 12 is:
        Center 1
        Ring 1: 6
        Ring 2: 5 (to make 12)
        """
        d = math.sqrt(3) + 0.01 # Slightly larger to avoid initial overlap
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        angles = [0.0]
        
        # Ring 1
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            angles.append(0.0)
            
        # Ring 2 (pick 5)
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for p1 in ring1_vectors:
            for b in basis:
                p2 = p1 + b
                # Check if p2 is not in center or ring 1
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r0 and not is_r1:
                    duplicate = False
                    for pc in ring2_candidates:
                        if np.allclose(p2, pc):
                            duplicate = True
                            break
                    if not duplicate:
                        ring2_candidates.append(p2)
        
        # Sort by distance to pick the 5 closest to form a compact shape
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            angles.append(0.0)
            
        return centers, angles

    def generate_rectangular_grid():
        """
        3 rows x 4 cols.
        """
        dx = math.sqrt(3) + 0.01
        dy = 3.0 + 0.01
        centers = []
        angles = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                centers.append([cx, cy])
                angles.append(0.0)
        return centers, angles

    def generate_random():
        """
        Randomly scattered hexagons.
        """
        centers = []
        angles = []
        for _ in range(12):
            cx = random.uniform(-5, 5)
            cy = random.uniform(-5, 5)
            a = random.uniform(0, 60)
            centers.append([cx, cy])
            angles.append(a)
        return centers, angles

    # --- Main Logic ---

    best_val = float('inf')
    best_params = None

    initial_configs = [
        generate_hexagonal_cluster(),
        generate_rectangular_grid(),
        generate_random(),
        generate_random()
    ]

    for centers, angles in initial_configs:
        if time.time() > deadline - 15.0:
            break
            
        # Flatten params
        params = []
        for c in centers:
            params.append(c[0])
            params.append(c[1])
        for a in angles:
            params.append(a)
        # Outer hexagon params (cx, cy, theta)
        params.append(0.0)
        params.append(0.0)
        params.append(0.0)
        
        params = np.array(params)
        
        # Check initial validity
        if is_valid_config(centers, angles):
            val = compute_outer_side(centers, angles, 0.0, 0.0, 0.0)
            if val < best_val:
                best_val = val
                best_params = params.copy()

        # Optimize
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 200})
            
            if time.time() > deadline - 5.0:
                continue

            # Extract result
            res_centers = [[res.x[i*2], res.x[i*2+1]] for i in range(12)]
            res_angles = [res.x[24+i] for i in range(12)]
            res_ocx = res.x[36]
            res_ocy = res.x[37]
            res_otheta = res.x[38]
            
            # Check validity of the optimized result
            if is_valid_config(res_centers, res_angles):
                val = compute_outer_side(res_centers, res_angles, res_ocx, res_ocy, res_otheta)
                if val < best_val:
                    best_val = val
                    best_params = res.x.copy()
            else:
                # If invalid, check if the initial config was valid and better than current best
                # (We already checked initial config above, so this is a fallback if optimization failed to find valid)
                pass

        except Exception:
            pass

    if best_params is None:
        # Fallback: return the first initial config
        centers, angles = initial_configs[0]
        best_params = []
        for c in centers:
            best_params.extend([c[0], c[1]])
        best_params.extend(angles)
        best_params.extend([0.0, 0.0, 0.0])
        best_params = np.array(best_params)
        best_val = compute_outer_side(
            [[best_params[i*2], best_params[i*2+1]] for i in range(12)],
            [best_params[24+i] for i in range(12)],
            0.0, 0.0, 0.0
        )

    # Final Polish: Optimize only the outer hexagon angle for the best inner configuration
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    ocx = best_params[36]
    ocy = best_params[37]
    otheta = best_params[38]
    
    # Grid search for best outer angle
    best_side = best_val
    best_otheta = otheta
    
    # Coarse search
    for da in np.linspace(-30, 30, 61):
        s = compute_outer_side(centers, angles, ocx, ocy, otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = otheta + da
            
    # Fine search
    for da in np.linspace(-5, 5, 101):
        s = compute_outer_side(centers, angles, ocx, ocy, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
            
    # Very fine search
    for da in np.linspace(-1, 1, 21):
        s = compute_outer_side(centers, angles, ocx, ocy, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da

    # Prepare output
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

