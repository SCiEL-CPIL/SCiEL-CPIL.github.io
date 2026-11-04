import math
import numpy as np

def construct():
    # Define the tight hexagonal lattice cluster with small gap
    gap = 1.001
    sqrt3 = math.sqrt(3)
    u = np.array([sqrt3 * gap, 0.0])
    v = np.array([sqrt3 * gap / 2.0, 1.5 * gap])
    
    centers = [np.array([0.0, 0.0])]
    ring1 = [u, v, v - u, -u, -v, u - v]
    for vec in ring1:
        centers.append(vec.copy())
        
    # Generate ring 2 candidates
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    for r1_vec in ring1:
        for b in basis:
            p = r1_vec + b
            is_r0 = np.allclose(p, np.array([0.0, 0.0]), atol=1e-6)
            is_r1 = False
            for r1j in ring1:
                if np.allclose(p, r1j, atol=1e-6):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p, pc, atol=1e-6):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p)
    
    ring2_candidates.sort(key=lambda p: (np.arctan2(p[1], p[0]), np.linalg.norm(p)))
    for i in range(5):
        centers.append(ring2_candidates[i].copy())

    centers = [(c[0], c[1]) for c in centers]
    angles = [0.0] * 12

    def compute_min_side(centers, angles, otheta_deg):
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(12):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    best_side = float('inf')
    best_otheta = 0.0
    
    for da in np.linspace(0, 60, 121):
        s = compute_min_side(centers, angles, da)
        if s < best_side:
            best_side = s
            best_otheta = da
            
    for da in np.linspace(best_otheta - 5, best_otheta + 5, 201):
        s = compute_min_side(centers, angles, da)
        if s < best_side:
            best_side = s
            best_otheta = da
            
    for da in np.linspace(best_otheta - 1, best_otheta + 1, 101):
        s = compute_min_side(centers, angles, da)
        if s < best_side:
            best_side = s
            best_otheta = da

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
