"""Packing 12 unit hexagons in a regular hexagon - SA + L-BFGS-B polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return True
        if dist_sq < 2.8:
            return False
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        all_edges = []
        for i in range(6):
            all_edges.append(v1[i] - v1[(i+1) % 6])
            all_edges.append(v2[i] - v2[(i+1) % 6])
        for edge in all_edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm == 0:
                continue
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return True
        return False

    def compute_min_side_fast(inner, center, angle):
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        return max_extent * 2.0 / sqrt3

    def objective(params):
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+3]
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        side = compute_min_side_fast(inner, center, angle)
        return side

    def smooth_objective(params, beta=50.0):
        """Smooth surrogate: soft-max containment + quadratic separation penalty."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+3]
        
        sqrt3 = np.sqrt(3)
        
        # Soft-max containment
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        max_extent = 0.0
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            # Soft-max
            max_e = (1.0/beta) * np.log(np.sum(np.exp(beta * extents)))
            if max_e > max_extent:
                max_extent = max_e
        
        side = max_extent * 2.0 / sqrt3
        
        # Separation penalty
        sep_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist_sq = dx*dx + dy*dy
                # If too close, penalize
                if dist_sq < 2.8:
                    sep_penalty += (2.8 - dist_sq) ** 2
        
        return side + 10.0 * sep_penalty

    def smooth_objective_grad(params, beta=50.0):
        """Numerical gradient of smooth objective."""
        eps = 1e-7
        grad = np.zeros_like(params)
        f0 = smooth_objective(params, beta)
        for i in range(len(params)):
            params[i] += eps
            f1 = smooth_objective(params, beta)
            params[i] -= eps
            grad[i] = (f1 - f0) / eps
        return grad

    def initial_guess():
        params = np.zeros(3*N + 4)
        positions = []
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((2*np.cos(theta), 2*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((4*np.cos(theta), 4*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+3] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = smooth_objective(best_params, 50.0)
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.5
    max_steps = 50000
    
    step_count = 0
    while time.time() < deadline - 10.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        idx = random.randint(0, 3*N + 3)
        if idx == 3*N + 2:
            continue
            
        current_step = step_size * (0.5 + 0.5 * T)
        new_params[idx] += random.gauss(0, current_step)
        
        # Use smooth objective for SA to allow passing through overlaps
        new_score = smooth_objective(new_params, 50.0)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            # Only update best if it's actually valid (hard check)
            # To save time, check hard validity every 100 steps or when new_score is very low
            if step_count % 50 == 0 or new_score < best_score:
                hard_score = objective(current_params)
                if hard_score < best_score:
                    best_score = hard_score
                    best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # L-BFGS-B polish phase
    if time.time() < deadline - 5.0 and best_score < 1e5:
        # Try polishing from best solution
        start_params = best_params.copy()
        try:
            result = minimize(
                smooth_objective,
                start_params,
                method='L-BFGS-B',
                jac=smooth_objective_grad,
                args=(50.0,),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < 1e5:
                polished = result.x
                polished_score = objective(polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_params = polished
        except Exception:
            pass

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+3])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final validation
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        inner = [[0.0, 0.0, 0.0]] * N
        center = (0.0, 0.0)
        side = 10.0
        angle = 0.0

    return inner, center, side, angle
# EVOLVE-BLOCK-END
