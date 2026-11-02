"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def hexagon_normals(vertices):
        """Returns 6 unit edge normals for a hexagon given its vertices."""
        normals = []
        for i in range(6):
            e = vertices[(i+1) % 6] - vertices[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                normals.append(n / nn)
        return normals

    def compute_overlap_penalty(h1, h2):
        """
        Computes a smooth penalty based on the sum of overlaps along all separating axes.
        Returns 0 if disjoint.
        """
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # If dist > 2.0, definitely disjoint (max distance between any two points on unit hexagons)
        if dist_sq > 4.0:
            return 0.0

        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        n1 = hexagon_normals(v1)
        n2 = hexagon_normals(v2)
        
        total_penalty = 0.0
        for n in n1 + n2:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 > min2 and max2 > min1:
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > 0:
                    total_penalty += overlap
        
        return total_penalty

    def compute_min_side_fast(inner, center, angle):
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
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
        """Objective: minimize outer side length + penalty for overlaps."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                penalty += compute_overlap_penalty(inner[i], inner[j])
        
        if penalty > 1e-6:
             return 1000.0 * penalty + compute_min_side_fast(inner, center, angle)
        
        side = compute_min_side_fast(inner, center, angle)
        return side

    def initial_guess():
        """Generate a valid initial honeycomb packing."""
        params = np.zeros(3*N + 3)
        sqrt3 = np.sqrt(3)
        
        positions = []
        positions.append((0, 0, 0.0))
        for i in range(6):
            theta = i * np.pi / 3
            r = np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            r = 2 * np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    # Phase 1: Simulated Annealing for global search
    current_params = best_params.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.9995
    step_size = 0.3
    
    step_count = 0
    sa_deadline = deadline - 40.0  # Leave 40s for NM refinement
    
    while time.time() < sa_deadline:
        step_count += 1
        new_params = current_params.copy()
        
        n_perturb = 1 if random.random() < 0.7 else 2
        for _ in range(n_perturb):
            idx = random.randint(0, 3*N + 2)
            current_step = step_size * (0.5 + 0.5 * T)
            new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / T):
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Phase 2: Multi-start Nelder-Mead local refinement
    nm_deadline = deadline - 3.0
    
    if time.time() < nm_deadline:
        def nm_objective(x):
            return objective(x)
        
        # First NM from best SA result
        result = minimize(nm_objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun

        # Multi-start from perturbed points
        n_starts = 5
        for s in range(n_starts):
            if time.time() >= nm_deadline:
                break
            
            perturbed = best_params.copy()
            # Random perturbation with decreasing magnitude
            scale = 0.1 * (1 - s / n_starts) + 0.02
            for idx in range(len(perturbed)):
                perturbed[idx] += random.gauss(0, scale)
            
            p_score = objective(perturbed)
            if p_score < best_score:
                result2 = minimize(nm_objective, perturbed, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if result2.fun < best_score:
                    best_params = result2.x
                    best_score = result2.fun

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = compute_min_side_fast(inner, center, angle)
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if compute_overlap_penalty(inner[i], inner[j]) > 1e-6:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use a known valid configuration (hexagonal grid)
        inner = []
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((np.sqrt(3)*np.cos(theta), np.sqrt(3)*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((2*np.sqrt(3)*np.cos(theta), 2*np.sqrt(3)*np.sin(theta), 0))
        for pos in positions:
            inner.append([float(pos[0]), float(pos[1]), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
