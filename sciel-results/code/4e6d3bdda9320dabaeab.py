import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        thetas = angle + np.arange(6) * (math.pi / 3)
        xs = cx + side * np.cos(thetas)
        ys = cy + side * np.sin(thetas)
        return np.column_stack((xs, ys))

    def sat_gap(poly1, poly2):
        """Returns the minimum separation gap between two convex polygons.
        Positive means separated, negative means overlapping."""
        min_gap = float('inf')
        
        # Check edges of poly1
        for i in range(6):
            edge = poly1[(i + 1) % 6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        # Check edges of poly2
        for i in range(6):
            edge = poly2[(i + 1) % 6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap
        
        return min_gap

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute the minimum outer hexagon side length that contains all inner hexagons."""
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        normal_angles = otheta_rad + math.pi / 6 + np.arange(6) * (math.pi / 3)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            thetas = angle_rad + np.arange(6) * (math.pi / 3)
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            
            dx = vx - ocx
            dy = vy - ocy
            projs = dx[:, np.newaxis] * nx[np.newaxis, :] + dy[:, np.newaxis] * ny[np.newaxis, :]
            current_max = np.max(projs)
            if current_max > max_proj:
                max_proj = current_max
        
        return 2.0 * max_proj / math.sqrt(3)

    def calculate_cost(centers, angles, ocx, ocy, otheta):
        """Objective: outer side length + smooth penalty for overlaps."""
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        penalty = 0.0
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                gap = sat_gap(verts[i], verts[j])
                if gap < 0:
                    # Smooth quadratic penalty for overlap
                    penalty += 100.0 * gap * gap
        
        return side + penalty

    def is_valid(centers, angles, tol=1e-5):
        """Check if configuration is valid (no overlaps)."""
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        for i in range(12):
            for j in range(i + 1, 12):
                if sat_gap(verts[i], verts[j]) < -tol:
                    return False
        return True

    def optimize_sa(centers_init, angles_init, ocx_init, ocy_init, otheta_init):
        """Simulated annealing with smooth penalty."""
        T = 0.5
        T_min = 0.0003
        alpha = 0.982
        steps_per_temp = 20
        
        current_centers = [c.copy() for c in centers_init]
        current_angles = list(angles_init)
        current_ocx = ocx_init
        current_ocy = ocy_init
        current_otheta = otheta_init
        
        current_cost = calculate_cost(current_centers, current_angles, current_ocx, current_ocy, current_otheta)
        
        best_centers = [c.copy() for c in current_centers]
        best_angles = list(current_angles)
        best_ocx = current_ocx
        best_ocy = current_ocy
        best_otheta = current_otheta
        best_cost = current_cost
        
        while T > T_min:
            for _ in range(steps_per_temp):
                if time.time() > deadline - 20:
                    break
                
                new_centers = [c.copy() for c in current_centers]
                new_angles = list(current_angles)
                new_ocx = current_ocx
                new_ocy = current_ocy
                new_otheta = current_otheta
                
                # Perturb 1-2 hexagons
                n_perturb = random.randint(1, 2)
                for _ in range(n_perturb):
                    i = random.randint(0, 11)
                    step_size = T * 0.2
                    new_centers[i][0] += random.gauss(0, step_size)
                    new_centers[i][1] += random.gauss(0, step_size)
                    new_angles[i] += random.gauss(0, step_size * 4)
                
                if random.random() < 0.1:
                    new_ocx += random.gauss(0, T * 0.06)
                    new_ocy += random.gauss(0, T * 0.06)
                    new_otheta += random.gauss(0, T * 0.06)
                
                new_cost = calculate_cost(new_centers, new_angles, new_ocx, new_ocy, new_otheta)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_centers = new_centers
                    current_angles = new_angles
                    current_ocx = new_ocx
                    current_ocy = new_ocy
                    current_otheta = new_otheta
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_centers = [c.copy() for c in current_centers]
                        best_angles = list(current_angles)
                        best_ocx = current_ocx
                        best_ocy = current_ocy
                        best_otheta = current_otheta
            
            T *= alpha
            if time.time() > deadline - 20:
                break
                
        return best_centers, best_angles, best_ocx, best_ocy, best_otheta, best_cost

    def generate_lattice_config():
        """Hexagonal lattice with slight spacing to avoid exact contact."""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v - u, -u, -v, u - v]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v - u, -u, -v, u - v]
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
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
                        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_circular_config():
        """Two concentric rings."""
        centers = [np.array([0.0, 0.0])]
        r = math.sqrt(3) + 0.02
        for i in range(6):
            angle = i * math.pi / 3
            cx = r * math.cos(angle)
            cy = r * math.sin(angle)
            centers.append(np.array([cx, cy]))
        
        r2 = 2 * math.sqrt(3) + 0.02
        for i in range(5):
            angle = i * 2 * math.pi / 5 + math.pi / 5
            cx = r2 * math.cos(angle)
            cy = r2 * math.sin(angle)
            centers.append(np.array([cx, cy]))
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_row_config():
        """Two rows of 6."""
        centers = []
        d = math.sqrt(3) + 0.01
        for i in range(6):
            centers.append(np.array([i * d, 0.0]))
        for i in range(6):
            centers.append(np.array([i * d + d/2, d * math.sqrt(3)/2]))
            
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_staggered_config():
        """3 rows of 4, staggered."""
        centers = []
        d = math.sqrt(3) + 0.01
        h = d * math.sqrt(3) / 2
        for row in range(3):
            y = row * h
            x_offset = (row % 2) * d / 2
            for col in range(4):
                centers.append(np.array([col * d + x_offset, y]))
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    def generate_lattice_sheared_config(shear_factor=0.1):
        """Lattice with shear deformation."""
        d = math.sqrt(3) + 0.01
        u = np.array([d, 0.0])
        v = np.array([d / 2.0 + shear_factor * d, d * math.sqrt(3) / 2.0])
        
        centers = []
        # Generate 3x4 grid in lattice coordinates
        for i in range(-1, 2):
            for j in range(-1, 2):
                p = i * u + j * v
                centers.append(p)
        
        # Sort by distance from center and take 12 closest
        centers.sort(key=lambda p: np.linalg.norm(p))
        centers = [c.copy() for c in centers[:12]]
        
        angles = [30.0] * 12
        return centers, angles, 0.0, 0.0, 0.0

    configs = [
        generate_lattice_config(),
        generate_circular_config(),
        generate_row_config(),
        generate_staggered_config(),
        generate_lattice_sheared_config(0.1),
        generate_lattice_sheared_config(-0.1),
    ]
    
    best_overall_cost = float('inf')
    best_centers = None
    best_angles = None
    best_ocx = 0.0
    best_ocy = 0.0
    best_otheta = 0.0
    
    for centers_init, angles_init, ocx_init, ocy_init, otheta_init in configs:
        if time.time() > deadline - 40:
            break
        
        sa_centers, sa_angles, sa_ocx, sa_ocy, sa_otheta, sa_cost = optimize_sa(
            centers_init, angles_init, ocx_init, ocy_init, otheta_init
        )
        
        # Polish outer hexagon with Nelder-Mead
        def outer_objective(v):
            if time.time() > deadline - 15:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(sa_centers, sa_angles, ox, oy, oa)
        
        result = minimize(outer_objective, [sa_ocx, sa_ocy, sa_otheta], method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        pol_ocx, pol_ocy, pol_otheta = result.x
        pol_cost = result.fun
        
        # Validate
        valid = is_valid(sa_centers, sa_angles, tol=1e-5)
        
        if valid and pol_cost < best_overall_cost:
            best_overall_cost = pol_cost
            best_centers = sa_centers
            best_angles = sa_angles
            best_ocx = pol_ocx
            best_ocy = pol_ocy
            best_otheta = pol_otheta
    
    if best_centers is None:
        best_centers, best_angles, best_ocx, best_ocy, best_otheta = generate_lattice_config()
        best_overall_cost = compute_outer_side(best_centers, best_angles, best_ocx, best_ocy, best_otheta)
    
    best_side = best_overall_cost
    final_ocx, final_ocy, final_otheta = best_ocx, best_ocy, best_otheta
    
    # Final high-precision polish on outer hexagon
    if time.time() < deadline - 5:
        def outer_obj(v):
            if time.time() > deadline - 3:
                return 1e6
            ox, oy, oa = v[0], v[1], v[2]
            return compute_outer_side(best_centers, best_angles, ox, oy, oa)
        
        result2 = minimize(outer_obj, [final_ocx, final_ocy, final_otheta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
        if result2.fun < best_side:
            best_side = result2.fun
            final_ocx, final_ocy, final_otheta = result2.x
    
    # Fine grid search for outer hexagon
    if time.time() < deadline - 2:
        for da in np.linspace(-0.2, 0.2, 9):
            for dx in np.linspace(-0.003, 0.003, 7):
                for dy in np.linspace(-0.003, 0.003, 7):
                    if time.time() > deadline - 1:
                        break
                    s = compute_outer_side(best_centers, best_angles, final_ocx + dx, final_ocy + dy, final_otheta + da)
                    if s < best_side:
                        best_side = s
                        final_ocx, final_ocy, final_otheta = final_ocx + dx, final_ocy + dy, final_otheta + da
                if time.time() > deadline - 1:
                    break
            if time.time() > deadline - 1:
                break

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (float(final_ocx), float(final_ocy))
    outer_side = float(best_side)
    outer_angle = float(final_otheta)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
