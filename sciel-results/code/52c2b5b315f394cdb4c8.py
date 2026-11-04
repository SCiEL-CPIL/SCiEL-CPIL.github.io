import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 138.0
    random.seed(42)
    np.random.seed(42)

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            nx = -edge[1]
            ny = edge[0]
            min1, max1 = project(v1, nx, ny)
            min2, max2 = project(v2, nx, ny)
            if max1 < min2 + eps or max2 < min1 + eps:
                return True
        return False

    def min_center_dist_sq(a1, a2):
        """Compute the minimum center-center distance squared for two unit hexagons
        with given rotation angles to be disjoint. Returns a lower bound on dist_sq
        such that if actual dist_sq < this, they definitely overlap."""
        # The worst case for minimum distance is when one hexagon's vertex points
        # toward the other's center. The apothem is sqrt(3)/2.
        # For two hexagons to be disjoint, their Minkowski sum (which is a hexagon
        # with side length 2) must not contain the origin. The support function of
        # the Minkowski sum in direction d is h1(d) + h2(d) where h is the support
        # function of each hexagon.
        # A simple lower bound: if center distance < 2*apothem = sqrt(3), they might
        # still be disjoint if rotated appropriately. The true minimum depends on
        # relative angle. For safety, use a conservative bound.
        # Actually, for two unit regular hexagons, the minimum distance between
        # centers for non-overlap is at least 2*sqrt(3)/2 = sqrt(3) when aligned,
        # but can be smaller with rotation. The absolute minimum is when a vertex
        # of one touches the center of the other: distance = 1 (circumradius).
        # But that's not disjoint - they'd overlap. The minimum for disjoint is
        # when edges are parallel and touching: sqrt(3). With rotation, it can be
        # less. Let's use sqrt(3) - 0.1 as a safe lower bound for the fast check.
        return (math.sqrt(3) - 0.15) ** 2

    def quick_disjoint_check(centers, angles):
        n = len(centers)
        min_d2 = min_center_dist_sq(0, 0)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_d2:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        max_proj = 0.0
        otheta_rad = math.radians(otheta_deg)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def calculate_cost(centers, angles, ocx, ocy, otheta_deg, fast=True):
        side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        if fast:
            if not quick_disjoint_check(centers, angles):
                return side + 50.0
        else:
            verts = []
            for i in range(12):
                v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts.append(v)
            for i in range(12):
                for j in range(i + 1, 12):
                    if not are_disjoint_sat(verts[i], verts[j]):
                        return side + 100.0
        return side

    def polish_outer(centers, angles, ocx, ocy, otheta_deg):
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta_deg)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta_deg
        
        scales_pos = [0.5, 0.1, 0.01, 0.001, 0.0001]
        scales_ang = [10.0, 1.0, 0.1, 0.01, 0.001]
        
        for scale_p, scale_a in zip(scales_pos, scales_ang):
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for dx in [-scale_p, 0, scale_p]:
                    for dy in [-scale_p, 0, scale_p]:
                        for da in [-scale_a, 0, scale_a]:
                            s = compute_outer_side(centers, angles, 
                                                   best_ocx + dx, best_ocy + dy, 
                                                   (best_otheta + da) % 60)
                            if s < best_side - 1e-12:
                                best_side = s
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = (best_otheta + da) % 60
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    def generate_lattice_solution():
        eps = 0.01
        d = math.sqrt(3) + eps
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [
            u, v, v - u, -u, -v, u - v
        ]
        for vec in ring1_vectors:
            centers.append(vec.copy())
            
        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        
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
            
        centers_list = [[c[0], c[1]] for c in centers]
        angles_list = [30.0] * 12
        
        all_verts = []
        for x, y, a in zip([c[0] for c in centers_list], [c[1] for c in centers_list], angles_list):
            verts = get_hex_vertices(x, y, 1.0, a)
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        
        cx = np.mean(all_verts[:, 0])
        cy = np.mean(all_verts[:, 1])
        
        ocx, ocy, otheta, side = polish_outer(centers_list, angles_list, cx, cy, 0.0)
        
        return centers_list, angles_list, ocx, ocy, otheta, side

    def generate_random_solution(scale=2.0):
        centers = [[random.gauss(0, scale), random.gauss(0, scale)] for _ in range(12)]
        angles = [random.uniform(0, 60) for _ in range(12)]
        
        all_verts = []
        for i in range(12):
            verts = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            all_verts.extend(verts)
        all_verts = np.array(all_verts)
        
        cx = np.mean(all_verts[:, 0])
        cy = np.mean(all_verts[:, 1])
        
        ocx, ocy, otheta, side = polish_outer(centers, angles, cx, cy, random.uniform(0, 60))
        
        return centers, angles, ocx, ocy, otheta, side

    def optimize_slsqp(centers, angles, ocx, ocy, otheta, max_time=10.0):
        n = len(centers)
        start_time = time.time()
        
        def flatten(centers, angles):
            v = []
            for i in range(n):
                v.extend([centers[i][0], centers[i][1], angles[i]])
            return v
        
        def unflatten(v):
            c = []
            a = []
            for i in range(n):
                c.append([v[3*i], v[3*i+1]])
                a.append(v[3*i+2])
            return c, a
        
        def objective(v):
            if time.time() > start_time + max_time or time.time() > deadline - 5:
                return 1e6
            c, a = unflatten(v)
            return calculate_cost(c, a, ocx, ocy, otheta, fast=True)
        
        x0 = flatten(centers, angles)
        
        try:
            result = minimize(objective, x0, method='SLSQP',
                            options={'maxiter': 300, 'ftol': 1e-8})
            if result.fun < 1e5:
                c, a = unflatten(result.x)
                if quick_disjoint_check(c, a):
                    return c, a
        except Exception:
            pass
        return centers, angles

    def high_precision_polish(centers, angles, ocx, ocy, otheta, max_time=5.0):
        """Final high-precision polish using Nelder-Mead with tight tolerances."""
        n = len(centers)
        start_time = time.time()
        
        def flatten(centers, angles):
            v = []
            for i in range(n):
                v.extend([centers[i][0], centers[i][1], angles[i]])
            return v
        
        def unflatten(v):
            c = []
            a = []
            for i in range(n):
                c.append([v[3*i], v[3*i+1]])
                a.append(v[3*i+2])
            return c, a
        
        def objective(v):
            if time.time() > start_time + max_time or time.time() > deadline - 3:
                return 1e6
            c, a = unflatten(v)
            return calculate_cost(c, a, ocx, ocy, otheta, fast=False)
        
        x0 = flatten(centers, angles)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < 1e5:
                c, a = unflatten(result.x)
                # Verify with SAT
                verts = []
                for i in range(12):
                    v = get_hex_vertices(c[i][0], c[i][1], 1.0, a[i])
                    verts.append(v)
                valid = True
                for i in range(12):
                    for j in range(i + 1, 12):
                        if not are_disjoint_sat(verts[i], verts[j]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return c, a
        except Exception:
            pass
        return centers, angles

    # Start with lattice solution
    centers, angles, ocx, ocy, otheta, best_side = generate_lattice_solution()
    
    # Try SLSQP from lattice
    if time.time() < deadline - 25:
        centers_new, angles_new = optimize_slsqp(centers, angles, ocx, ocy, otheta, max_time=15.0)
        if centers_new != centers:
            ocx, ocy, otheta, side_new = polish_outer(centers_new, angles_new, ocx, ocy, otheta)
            if side_new < best_side:
                centers, angles = centers_new, angles_new
                best_side = side_new

    # Try multiple diverse restarts with SLSQP
    n_restarts = 6
    for r in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if r == 0:
            # Perturb best
            centers_r = [[c[0] + random.gauss(0, 0.3), c[1] + random.gauss(0, 0.3)] for c in centers]
            angles_r = [(a + random.gauss(0, 3.0)) % 60 for a in angles]
            ocx_r, ocy_r, otheta_r = ocx, ocy, otheta
        elif r == 1:
            # Different lattice perturbation
            eps = 0.01
            d = math.sqrt(3) + eps + random.gauss(0, 0.05)
            u = np.array([d, 0.0])
            v = np.array([d/2.0, d*math.sqrt(3)/2.0])
            ring1_vectors = [u, v, v-u, -u, -v, u-v]
            centers_r = [[0.0, 0.0]] + [[vec[0], vec[1]] for vec in ring1_vectors]
            basis = [u, v, v-u, -u, -v, u-v]
            ring2_cands = []
            for p1 in ring1_vectors:
                for b in basis:
                    p2 = p1 + b
                    if np.linalg.norm(p2) > 0.1 and all(np.linalg.norm(p2 - p1j) > 0.1 for p1j in ring1_vectors):
                        if all(np.linalg.norm(p2 - pc) > 0.1 for pc in ring2_cands):
                            ring2_cands.append(p2)
            ring2_cands.sort(key=lambda p: np.linalg.norm(p))
            centers_r += [[c[0], c[1]] for c in ring2_cands[:5]]
            angles_r = [random.uniform(0, 60) for _ in range(12)]
            all_v = []
            for x, y, a in zip([c[0] for c in centers_r], [c[1] for c in centers_r], angles_r):
                all_v.extend(get_hex_vertices(x, y, 1.0, a))
            all_v = np.array(all_v)
            ocx_r, ocy_r = np.mean(all_v[:, 0]), np.mean(all_v[:, 1])
            otheta_r = random.uniform(0, 60)
        else:
            centers_r, angles_r, ocx_r, ocy_r, otheta_r, _ = generate_random_solution(scale=1.5)
        
        # Polish outer
        ocx_r, ocy_r, otheta_r, side_r = polish_outer(centers_r, angles_r, ocx_r, ocy_r, otheta_r)
        
        # Run SLSQP
        time_left = deadline - time.time() - 15
        if time_left > 5:
            centers_opt, angles_opt = optimize_slsqp(centers_r, angles_r, ocx_r, ocy_r, otheta_r, max_time=min(12.0, time_left/2))
            
            # Polish outer for optimized config
            ocx_f, ocy_f, otheta_f, side_f = polish_outer(centers_opt, angles_opt, ocx_r, ocy_r, otheta_r)
            
            if side_f < best_side:
                centers, angles = centers_opt, angles_opt
                ocx, ocy, otheta = ocx_f, ocy_f, otheta_f
                best_side = side_f

    # High-precision polish
    if time.time() < deadline - 10:
        centers_hp, angles_hp = high_precision_polish(centers, angles, ocx, ocy, otheta, max_time=min(8.0, deadline - time.time() - 8))
        if centers_hp != centers:
            ocx, ocy, otheta, side_hp = polish_outer(centers_hp, angles_hp, ocx, ocy, otheta)
            if side_hp < best_side:
                centers, angles = centers_hp, angles_hp
                best_side = side_hp

    # Final outer polish
    ocx, ocy, otheta, best_side = polish_outer(centers, angles, ocx, ocy, otheta)
    
    # Verify disjointness
    verts = []
    for i in range(12):
        v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        verts.append(v)
    
    valid = True
    for i in range(12):
        for j in range(i + 1, 12):
            if not are_disjoint_sat(verts[i], verts[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to lattice if invalid
        centers, angles, ocx, ocy, otheta, best_side = generate_lattice_solution()

    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(12)]
    outer_center = (ocx, ocy)
    outer_side = best_side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
