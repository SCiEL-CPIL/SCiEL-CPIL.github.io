"""Hexagon cluster packing: 12 unit hexes using conformal map + global rotation search."""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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
            max_abs_e = np.max(np.abs(extents))
            if max_abs_e > max_extent:
                max_extent = max_abs_e
        
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
        
        side = compute_min_side_fast(inner, center, angle)
        
        if penalty > 1e-9:
            return side + 500.0 * penalty * penalty
        
        return side

    def mobius_transform(z, a, b):
        """
        Möbius transformation: w = (z - a) / (1 - conj(a) * z)
        Maps unit disk to itself. a is the center of the target disk region.
        """
        conj_a = np.conj(a)
        denom = 1 - conj_a * z
        # Handle near-singular case
        mask = np.abs(denom) < 1e-10
        if np.any(mask):
            denom[mask] = 1e-10
        w = (z - a) / denom
        return w

    def generate_circle_packing_config(n_points, seed=None):
        """
        Generate approximate circle packing positions in unit disk.
        Uses a simple iterative relaxation approach.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Initial positions: Fibonacci spiral in unit disk
        golden_angle = np.pi * (3 - np.sqrt(5))
        positions = np.zeros((n_points, 2))
        for i in range(n_points):
            r = np.sqrt((i + 0.5) / n_points) * 0.95
            theta = i * golden_angle
            positions[i, 0] = r * np.cos(theta)
            positions[i, 1] = r * np.sin(theta)
        
        # Relaxation: push points apart, keep in disk
        for iteration in range(200):
            for i in range(n_points):
                for j in range(n_points):
                    if i == j:
                        continue
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    min_dist = 0.35  # Approximate minimum distance for unit hex centers
                    if dist < min_dist and dist > 1e-10:
                        # Push apart
                        force = (min_dist - dist) / dist * 0.5
                        positions[i] += force * diff
                        positions[j] -= force * diff
            
            # Keep in unit disk
            for i in range(n_points):
                r = np.linalg.norm(positions[i])
                if r > 0.95:
                    positions[i] *= 0.95 / r
        
        return positions

    def hexagon_to_circle_mapping(inner_positions, outer_angle_deg):
        """
        Map hexagon containment problem to circle packing via conformal map.
        The key insight: a regular hexagon with apothem R can be approximated by a circle of radius R.
        We use this to get initial positions, then refine.
        
        Returns initial positions for inner hexagons.
        """
        # Generate circle packing positions
        circle_positions = generate_circle_packing_config(N, seed=42)
        
        # Scale to appropriate size
        # For unit hexagons, center-to-center touching distance is sqrt(3)
        # In circle packing, we used min_dist=0.35, so scale factor is sqrt(3)/0.35
        scale = np.sqrt(3) / 0.35
        hex_positions = circle_positions * scale
        
        # Center at origin
        hex_positions -= np.mean(hex_positions, axis=0)
        
        return hex_positions

    def initial_guess_conformal():
        """Generate initial packing using conformal map approach."""
        params = np.zeros(3*N + 3)
        
        hex_positions = hexagon_to_circle_mapping(None, 0.0)
        
        for i in range(N):
            params[3*i] = hex_positions[i, 0]
            params[3*i+1] = hex_positions[i, 1]
            params[3*i+2] = 0.0
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    def initial_guess_honeycomb():
        """Generate a valid initial honeycomb packing as fallback."""
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

    # Try multiple initializations
    best_params = None
    best_score = np.inf
    
    initializations = [
        ("conformal", initial_guess_conformal),
        ("honeycomb", initial_guess_honeycomb)
    ]
    
    for name, init_func in initializations:
        if time.time() > deadline - 20:
            break
        try:
            params = init_func()
            score = objective(params)
            if score < best_score:
                best_score = score
                best_params = params.copy()
        except Exception:
            pass
    
    if best_params is None:
        best_params = initial_guess_honeycomb()
        best_score = objective(best_params)
    
    # Phase 1: Differential Evolution for global search (faster convergence than SA)
    if time.time() < deadline - 30:
        # Use scipy's differential evolution
        bounds = [(-5, 5)] * (3*N) + [(-2, 2), (-2, 2), (-30, 30)]
        
        de_result = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_params = de_result.x.copy()
    
    # Phase 2: Multiple Nelder-Mead refinements from perturbed starts
    nm_deadline = deadline - 5.0
    
    # First refinement from best
    if time.time() < nm_deadline:
        result = minimize(objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    
    # Additional refinements from perturbed starts
    n_perturb_trials = 3
    for trial in range(n_perturb_trials):
        if time.time() > nm_deadline:
            break
        
        perturbed = best_params.copy()
        # Perturb only inner hexagon positions, keep center and angle
        for i in range(N):
            perturbed[3*i] += random.gauss(0, 0.1)
            perturbed[3*i+1] += random.gauss(0, 0.1)
        
        p_score = objective(perturbed)
        if p_score < best_score:
            result = minimize(objective, perturbed, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
    
    # Final high-precision refinement
    if time.time() < deadline - 2:
        result = minimize(objective, best_params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()

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
        # Fallback: use honeycomb configuration
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
