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

    def project(vertices, nx, ny):
        projs = np.dot(vertices, [nx, ny])
        return np.min(projs), np.max(projs)

    def are_disjoint_sat(v1, v2, eps=1e-9):
        """
        Returns True if v1 and v2 are disjoint (including touching).
        Uses Separating Axis Theorem.
        """
        # Check axes from v1
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
        
        # Check axes from v2
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

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        max_proj = 0.0
        otheta_rad = math.radians(otheta)
        for k in range(6):
            normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                # Vertices of inner hexagon
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - 5:
            return 1e9

        # Unpack parameters
        # 0-23: centers (12*2)
        # 24-35: angles (12)
        # 36-37: outer center (2)
        # 38: outer angle
        
        centers = []
        for i in range(12):
            centers.append([params[i * 2], params[i * 2 + 1]])
        angles = [params[24 + i] for i in range(12)]
        ocx, ocy = params[36], params[37]
        otheta = params[38]
        
        # Compute outer side length
        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        
        # Check for overlaps using SAT
        verts = []
        for i in range(12):
            v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
            
        for i in range(12):
            for j in range(i + 1, 12):
                if not are_disjoint_sat(verts[i], verts[j]):
                    # Penalty for overlap
                    return side + 10.0
            
        return side

    # Initial configuration: Hexagonal lattice
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers_init = [np.array([0.0, 0.0])]
    
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
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
        
    assert len(centers_init) == 12
    
    # Prepare initial vector for optimization
    x0 = []
    for c in centers_init:
        x0.append(c[0])
        x0.append(c[1])
    for i in range(12):
        x0.append(30.0) # Initial angles
    x0.extend([0.0, 0.0, 0.0]) # Outer center and angle
    
    # Run Optimization
    best_result = x0
    best_val = objective(x0)
    
    try:
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-6})
        
        if time.time() < deadline - 5:
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
    except Exception:
        pass

    # Extract best parameters
    centers = []
    for i in range(12):
        centers.append([best_result[i * 2], best_result[i * 2 + 1]])
    angles = [best_result[24 + i] for i in range(12)]
    ocx, ocy = best_result[36], best_result[37]
    otheta = best_result[38]
    
    # Final Polish: Grid search on outer parameters
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Refine outer hexagon orientation and position
    for da in np.linspace(-10, 10, 41):
        for dx in np.linspace(-0.5, 0.5, 21):
            for dy in np.linspace(-0.5, 0.5, 21):
                s = compute_outer_side(centers, angles, ocx + dx, ocy + dy, otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_otheta = ocx + dx, ocy + dy, otheta + da
                    
    # Verify final validity (safety check)
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
        # Fallback to initial lattice if optimization resulted in invalid configuration
        centers = [[c[0], c[1]] for c in centers_init]
        angles = [30.0] * 12
        ocx, ocy, otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
