import math
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time
import random

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    def get_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        verts = np.zeros((6, 2))
        for i in range(6):
            theta = angle + i * math.pi / 3
            verts[i, 0] = cx + side * math.cos(theta)
            verts[i, 1] = cy + side * math.sin(theta)
        return verts
    
    def min_separation(v1, v2):
        """Minimum separation between two convex hexagons using SAT."""
        min_sep = float('inf')
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                nx, ny = -edge[1], edge[0]
                len_n = math.sqrt(nx*nx + ny*ny)
                if len_n < 1e-15:
                    continue
                nx /= len_n
                ny /= len_n
                proj1 = np.array([p1[0]*nx + p1[1]*ny for p1 in v1])
                proj2 = np.array([p1[0]*nx + p1[1]*ny for p1 in v2])
                sep = min(proj1.max() - proj2.min(), proj2.max() - proj1.min())
                if sep < min_sep:
                    min_sep = sep
        return min_sep
    
    def is_valid(centers, angles, eps=1e-9):
        verts = []
        for i in range(12):
            v = get_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        for i in range(12):
            for j in range(i + 1, 12):
                if min_separation(verts[i], verts[j]) < -eps:
                    return False
        return True
    
    def compute_outer_side(centers, angles, otheta_deg=0.0):
        """Compute minimum enclosing regular hexagon side length."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            max_p = float('-inf')
            for i in range(12):
                cx, cy = centers[i]
                angle_rad = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle_rad + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = vx * nx + vy * ny
                    if proj > max_p:
                        max_p = proj
            if max_p > max_proj:
                max_proj = max_p
        return 2.0 * max_proj / math.sqrt(3)
    
    def objective(params):
        """Objective: outer hexagon side length + penalty for overlaps."""
        centers = [[params[i*2], params[i*2+1]] for i in range(12)]
        angles = [params[24+i] for i in range(12)]
        
        # Compute overlap penalty
        verts = []
        for i in range(12):
            v = get_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            verts.append(v)
        
        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                sep = min_separation(verts[i], verts[j])
                if sep < 0:
                    penalty += sep * sep
        
        side = compute_outer_side(centers, angles, 0.0)
        return side + 10.0 * penalty
    
    def generate_hex_cluster():
        """Generate 12 hexagons in a hexagonal cluster pattern."""
        d = math.sqrt(3) + 0.02
        u = np.array([d, 0.0])
        v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
        
        centers = [np.array([0.0, 0.0])]
        ring1 = [u, v, v - u, -u, -v, u - v]
        for vec in ring1:
            centers.append(vec.copy())
        
        # Add 5 more from ring 2
        basis = [u, v, v - u, -u, -v, u - v]
        ring2_candidates = []
        for i in range(6):
            for b in basis:
                p2 = ring1[i] + b
                is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
                is_r1 = any(np.allclose(p2, r) for r in ring1)
                if not is_r0 and not is_r1:
                    if not any(np.allclose(p2, pc) for pc in ring2_candidates):
                        ring2_candidates.append(p2)
        
        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())
        
        params = []
        for cx, cy in centers:
            params.extend([float(cx), float(cy)])
        params.extend([30.0] * 12)
        return np.array(params)
    
    def generate_rect_grid():
        """Generate 12 hexagons in a 3x4 rectangular grid."""
        dx = math.sqrt(3) + 0.02
        dy = 3.0 + 0.02
        params = []
        for row in range(3):
            for col in range(4):
                cx = (col - 1.5) * dx
                cy = (row - 1.0) * dy
                params.extend([cx, cy])
        params.extend([30.0] * 12)
        return np.array(params)
    
    def generate_linear_chain():
        """Generate 12 hexagons in a linear chain."""
        d = math.sqrt(3) + 0.02
        params = []
        for i in range(12):
            cx = (i - 5.5) * d
            cy = 0.0
            params.extend([cx, cy])
        params.extend([30.0] * 12)
        return np.array(params)
    
    def generate_compact_cluster():
        """Generate a more compact cluster with varied angles."""
        random.seed(123)
        d = math.sqrt(3) + 0.01
        centers = [[0.0, 0.0]]
        for i in range(11):
            angle = random.uniform(0, 2 * math.pi)
            cx = centers[-1][0] + d * math.cos(angle)
            cy = centers[-1][1] + d * math.sin(angle)
            centers.append([cx, cy])
        
        params = []
        for cx, cy in centers:
            params.extend([cx, cy])
        params.extend([random.choice([0, 30, 60]) for _ in range(12)])
        return np.array(params)
    
    def polish(params, max_iter=500):
        """High-precision polish using Nelder-Mead."""
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-12, 'fatol': 1e-12})
            return res.x
        except Exception:
            return params
    
    def polish_lbfbs(params, max_iter=300):
        """Polish using L-BFGS-B."""
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12})
            return res.x
        except Exception:
            return params
    
    # Generate multiple initial configurations
    inits = []
    inits.append(generate_hex_cluster())
    inits.append(generate_rect_grid())
    inits.append(generate_linear_chain())
    inits.append(generate_compact_cluster())
    
    # Add perturbed versions
    random.seed(42)
    for base in inits[:2]:
        for _ in range(3):
            perturbed = base.copy()
            for i in range(24):
                perturbed[i] += random.gauss(0, 0.1)
            for i in range(12):
                perturbed[24+i] = random.uniform(-15, 15)
            inits.append(perturbed)
    
    best_params = None
    best_val = float('inf')
    
    # Phase 1: Quick L-BFGS-B on all inits
    for init in inits:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, init, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10})
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish on best candidates
    if best_params is not None:
        if time.time() < deadline - 15:
            polished = polish(best_params, max_iter=300)
            val = objective(polished)
            if val < best_val:
                best_val = val
                best_params = polished
    
    # Phase 3: Try more inits with Nelder-Mead if time permits
    for init in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(objective, init, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            val = res.fun
            if val < best_val:
                best_val = val
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if best_params is not None and time.time() < deadline - 5:
        polished = polish(best_params, max_iter=500)
        val = objective(polished)
        if val < best_val:
            best_val = val
            best_params = polished
    
    if best_params is None:
        best_params = inits[0]
    
    # Extract result
    centers = [[best_params[i*2], best_params[i*2+1]] for i in range(12)]
    angles = [best_params[24+i] for i in range(12)]
    
    # Fine-tune outer hexagon rotation
    best_side = compute_outer_side(centers, angles, 0.0)
    best_otheta = 0.0
    
    for da in np.linspace(-30, 30, 61):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
    
    for da in np.linspace(-5, 5, 101):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
    
    for da in np.linspace(-1, 1, 41):
        s = compute_outer_side(centers, angles, best_otheta + da)
        if s < best_side:
            best_side = s
            best_otheta = best_otheta + da
    
    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (0.0, 0.0)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle
