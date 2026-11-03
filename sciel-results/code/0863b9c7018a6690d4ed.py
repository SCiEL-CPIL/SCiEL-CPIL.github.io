import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return np.array(vertices)

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """
        Computes the side length of the outer hexagon required to contain
        the given inner hexagons.
        """
        otheta_rad = math.radians(otheta_deg)
        max_proj = 0.0
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

    def compute_outer_side_vec(x, n=12):
        """Vectorized outer side computation from flat parameter vector."""
        centers = np.array([[x[i*3], x[i*3+1]] for i in range(n)])
        angles_rad = np.array([math.radians(x[i*3+2]) for i in range(n)])
        ocx, ocy, otheta = x[3*n], x[3*n+1], x[3*n+2]
        otheta_rad = math.radians(otheta)
        
        # All 6 vertex offsets per hexagon: angle_rad + vi*pi/3
        vi = np.arange(6) * math.pi / 3  # (6,)
        # vertex angles: (n, 6)
        vert_angles = angles_rad[:, None] + vi[None, :]  # (n, 6)
        cos_va = np.cos(vert_angles)  # (n, 6)
        sin_va = np.sin(vert_angles)  # (n, 6)
        
        # Vertex positions: (n, 6)
        vx = centers[:, 0:1] + cos_va  # (n, 6)
        vy = centers[:, 1:2] + sin_va  # (n, 6)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = (vx - ocx) * nx + (vy - ocy) * ny  # (n, 6)
            max_proj = max(max_proj, np.max(proj))
        return 2.0 * max_proj / math.sqrt(3)

    def is_valid_sat(centers, angles):
        """
        Exact validity check using Separating Axis Theorem.
        """
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        def project(vertices, nx, ny):
            projs = np.dot(vertices, [nx, ny])
            return np.min(projs), np.max(projs)

        def are_disjoint(v1, v2, eps=1e-9):
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

        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint(verts[i], verts[j]):
                    return False
        return True

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta):
        """
        High-precision polish of outer hexagon parameters.
        """
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta
        
        # Multi-scale grid search
        scales = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            new_otheta = best_otheta + da_deg
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, new_otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = new_otheta
                                improved = True
        return best_ocx, best_ocy, best_otheta, best_side

    def optimize_cobyla(x0):
        """
        COBYLA optimization with soft overlap penalty.
        Variables: x1, y1, a1, ..., x12, y12, a12, ocx, ocy, otheta
        Uses smooth soft-max for the containment objective and a
        log-sum-exp soft penalty for pairwise distance violations.
        """
        n = 12
        beta = 50.0  # sharpness of soft penalty
        
        def objective(x):
            if time.time() > deadline - 30:
                return 1e10
            side = compute_outer_side_vec(x, n)
            
            # Soft overlap penalty using log-sum-exp
            # For each pair (i,j), compute dist^2; penalty if dist^2 < 3
            centers = np.array([[x[i*3], x[i*3+1]] for i in range(n)])
            # Pairwise squared distances
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dx = x[i*3] - x[j*3]
                    dy = x[i*3+1] - x[j*3+1]
                    d2 = dx*dx + dy*dy
                    # Soft penalty: if d2 < 3, penalize
                    # Use smooth max(0, 3 - d2) approximated by log-sum-exp
                    # log(1 + exp(beta*(3 - d2))) / beta approximates max(0, 3-d2)
                    arg = beta * (3.0 - d2)
                    if arg > 500:
                        penalty += (3.0 - d2)
                    elif arg < -500:
                        penalty += 0.0
                    else:
                        penalty += math.log(1.0 + math.exp(arg)) / beta
            
            return side + 100.0 * penalty
        
        try:
            res = minimize(objective, x0, method='COBYLA',
                           options={'maxiter': 500, 'rhobeg': 0.1, 'tol': 1e-10})
            return res.x
        except Exception:
            pass
        return x0

    # Generate starts
    starts = []
    
    # 1. Hexagonal Lattice
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    ring1_vectors = [
        u, v, v - u, -u, -v, u - v
    ]
    for vec in ring1_vectors:
        centers_init.append(vec.copy())
        
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
        centers_init.append(ring2_candidates[i].copy())
        
    x0 = []
    for c in centers_init:
        x0.append(c[0])
        x0.append(c[1])
    for i in range(12):
        x0.append(30.0)
    x0.extend([0.0, 0.0, 0.0])
    starts.append(x0)
    
    # 2. Tight hexagonal packing (1 center + 6 ring + 5 outer)
    d_tight = math.sqrt(3) + 0.001  # minimal spacing
    u_t = np.array([d_tight, 0.0])
    v_t = np.array([d_tight / 2.0, d_tight * math.sqrt(3) / 2.0])
    
    centers_tight = [np.array([0.0, 0.0])]
    ring1_t = [
        u_t, v_t, v_t - u_t, -u_t, -v_t, u_t - v_t
    ]
    for vec in ring1_t:
        centers_tight.append(vec.copy())
    
    # 5 more from ring 2, closest to center
    ring2_t = []
    basis_t = [u_t, v_t, v_t - u_t, -u_t, -v_t, u_t - v_t]
    for i in range(6):
        p1 = ring1_t[i]
        for b in basis_t:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_t:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_t:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_t.append(p2)
    ring2_t.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers_tight.append(ring2_t[i].copy())
    
    x_tight = []
    for c in centers_tight:
        x_tight.append(c[0])
        x_tight.append(c[1])
    for i in range(12):
        x_tight.append(30.0)
    x_tight.extend([0.0, 0.0, 0.0])
    starts.append(x_tight)
    
    # 3. Random Starts
    random.seed(42)
    for _ in range(3):
        centers_rand = []
        for _ in range(12):
            centers_rand.append([random.uniform(-3, 3), random.uniform(-3, 3)])
        x_rand = []
        for c in centers_rand:
            x_rand.append(c[0])
            x_rand.append(c[1])
        for i in range(12):
            x_rand.append(random.uniform(0, 60))
        x_rand.extend([0.0, 0.0, 0.0])
        starts.append(x_rand)

    def _are_disjoint_pair(v1, v2, eps=1e-9):
        """Check if two hexagons are disjoint using SAT."""
        def project(vertices, nx, ny):
            projs = np.dot(vertices, [nx, ny])
            return np.min(projs), np.max(projs)
        
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

    best_result = None
    best_val = float('inf')

    for x0 in starts:
        if time.time() > deadline - 25:
            break
        
        # Run COBYLA
        res_x = optimize_cobyla(x0)
        
        # Extract parameters
        centers = [[res_x[i*3], res_x[i*3+1]] for i in range(12)]
        angles = [res_x[i*3+2] for i in range(12)]
        ocx, ocy = res_x[36], res_x[37]
        otheta = res_x[38]
        
        # Polish outer hexagon
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        
        # Validate with SAT
        if is_valid_sat(centers, angles):
            if best_side < best_val:
                best_val = best_side
                best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)
        else:
            # Try to repair: nudge overlapping hexagons apart
            repaired = False
            for _attempt in range(50):
                if time.time() > deadline - 25:
                    break
                # Find overlapping pair
                verts = [get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]) for i in range(12)]
                overlap_pair = None
                for i in range(12):
                    for j in range(i+1, 12):
                        if not _are_disjoint_pair(verts[i], verts[j]):
                            overlap_pair = (i, j)
                            break
                    if overlap_pair:
                        break
                if overlap_pair is None:
                    repaired = True
                    break
                i, j = overlap_pair
                # Nudge apart
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-9:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                nudge = 0.05
                centers[i][0] += nudge * dx / dist
                centers[i][1] += nudge * dy / dist
                centers[j][0] -= nudge * dx / dist
                centers[j][1] -= nudge * dy / dist
                verts[i] = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                verts[j] = get_hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            
            if repaired and is_valid_sat(centers, angles):
                best_ocx, best_ocy, best_otheta, best_side = polish_outer(
                    centers, angles, ocx, ocy, otheta
                )
                if best_side < best_val:
                    best_val = best_side
                    best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    if best_result is None:
        # Fallback to initial lattice
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_ocx, best_ocy, best_otheta, best_side = polish_outer(
            centers, angles, ocx, ocy, otheta
        )
        best_result = (centers, angles, best_ocx, best_ocy, best_otheta, best_side)

    centers, angles, ocx, ocy, otheta, side = best_result
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (ocx, ocy)
    outer_side = side
    outer_angle = otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
