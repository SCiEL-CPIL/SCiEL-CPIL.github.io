"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

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

    def hexagons_disjoint_fast(h1, h2):
        """Fast check for disjointness using SAT (Separating Axis Theorem)."""
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        
        # Max distance between centers for disjoint hexagons is 2.0 (touching vertices)
        # If dist > 2.0, definitely disjoint
        if dist_sq > 4.0:
            return True
        
        # Min distance for overlap is approx sqrt(3) ~ 1.732 (touching edges)
        # If dist < 1.7, definitely overlap
        if dist_sq < 2.8: # 1.7^2
            return False
        
        # Precise SAT check
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
        """Vectorized computation of minimum outer side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        
        # Center offset
        offsets = all_verts - np.array(center)
        
        # Edge normals of outer hexagon
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            # Dot product of all offsets with normal
            extents = np.dot(offsets, normal)
            max_e = np.max(extents)
            if max_e > max_extent:
                max_extent = max_e
        
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """Objective: minimize outer side length while ensuring disjointness."""
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                if not hexagons_disjoint_fast(inner[i], inner[j]):
                    return 1e6
        
        # Compute side
        side = compute_min_side_fast(inner, center, angle)
        
        return side

    def initial_guess():
        params = np.zeros(3*N + 3)
        positions = []
        # Hexagonal packing for 12 hexagons: 1 center, 6 ring 1, 5 ring 2
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
        params[3*N+2] = 0.0
        return params

    def smooth_objective(params, beta):
        """
        Smooth surrogate using soft-min for containment and soft-max for overlap penalty.
        beta controls the smoothness (higher = sharper).
        """
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Compute all vertices
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        # Soft containment: for each of 6 normals, compute max projection using log-sum-exp
        # Then take max over all 6 normals
        sqrt3 = np.sqrt(3)
        
        # For each normal direction, compute the "soft max" of projections
        normal_projections = []
        for k in range(6):
            normal_angle = np.radians(angle) + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = np.dot(offsets, normal)
            
            # Soft max via log-sum-exp
            max_ext = np.max(extents)
            if max_ext > 0:
                soft_max = max_ext + np.log(np.sum(np.exp((extents - max_ext) / beta))) * beta
            else:
                # If all negative, soft max is close to max (which is negative)
                # But we want positive extent, so use abs or just max
                soft_max = max_ext
            
            normal_projections.append(soft_max)
        
        # The outer side is proportional to the max over all 6 normals
        # Use soft max over the 6 normal projections
        max_proj = np.max(normal_projections)
        if max_proj > 0:
            soft_side = max_proj + np.log(np.sum(np.exp((np.array(normal_projections) - max_proj) / beta))) * beta
            soft_side = soft_side * 2.0 / sqrt3
        else:
            soft_side = 2.0 / sqrt3  # Minimum side
        
        # Soft overlap penalty: for each pair, compute soft min of separation distance
        # If separation < 0, it's overlapping. We want to penalize this.
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                h1 = inner[i]
                h2 = inner[j]
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                # Approximate minimum separation for two unit hexagons
                # If dist < 2.0, they might overlap. Use a smooth penalty.
                # The exact minimum separation depends on relative rotation, but we approximate.
                # For simplicity, use a quadratic penalty when dist < 2.0
                if dist < 2.0:
                    # Smooth penalty: (2.0 - dist)^2
                    overlap_penalty += (2.0 - dist) ** 2
        
        return soft_side + 10.0 * overlap_penalty

    def polish(params, iterations=5000, lr=0.01):
        """
        High-precision polish pass using gradient-free local search with smaller steps.
        """
        current = params.copy()
        current_score = objective(current)
        
        for it in range(iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Try small perturbations
            best_neighbor = None
            best_neighbor_score = current_score
            
            for _ in range(10):
                candidate = current.copy()
                idx = random.randint(0, len(candidate) - 1)
                candidate[idx] += random.gauss(0, lr)
                score = objective(candidate)
                
                if score < best_neighbor_score:
                    best_neighbor_score = score
                    best_neighbor = candidate
            
            if best_neighbor is not None:
                current = best_neighbor
                current_score = best_neighbor_score
                lr *= 0.95  # Slowly reduce learning rate
            else:
                # No improvement, reduce learning rate more aggressively
                lr *= 0.9
            
            if lr < 0.001:
                break
        
        return current, current_score

    # Phase 1: Simulated Annealing with smooth objective for broad exploration
    best_params = initial_guess()
    best_score = objective(best_params)
    
    current_params = best_params.copy()
    current_score = best_score
    
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.9995
    step_size = 0.3
    max_steps = 30000
    
    step_count = 0
    beta = 0.5  # Smoothness parameter
    
    while time.time() < deadline - 20.0 and step_count < max_steps:
        step_count += 1
        new_params = current_params.copy()
        
        # Perturb one random parameter
        idx = random.randint(0, 3*N + 2)
        
        # Adaptive step size
        current_step = step_size * (0.5 + 0.5 * T)
        new_params[idx] += random.gauss(0, current_step)
        
        # Use smooth objective for acceptance decision
        smooth_score = smooth_objective(new_params, beta)
        
        # Accept if better in smooth sense, or with probability based on temperature
        if smooth_score < current_score or random.random() < np.exp(-(smooth_score - current_score) / max(T, 1e-10)):
            current_params = new_params
            current_score = smooth_score
            
            # Update best based on exact objective
            exact_score = objective(current_params)
            if exact_score < best_score:
                best_score = exact_score
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        # Periodically re-evaluate with exact objective to keep best accurate
        if step_count % 100 == 0:
            exact_score = objective(current_params)
            if exact_score < best_score:
                best_score = exact_score
                best_params = current_params.copy()

    # Phase 2: High-precision polish on the best solution found
    polished_params, polished_score = polish(best_params, iterations=10000, lr=0.02)
    if polished_score < best_score:
        best_score = polished_score
        best_params = polished_params

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
            if not hexagons_disjoint_fast(inner[i], inner[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Try to repair by slightly moving overlapping hexagons apart
        for _ in range(100):
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint_fast(inner[i], inner[j]):
                        # Move hexagon i away from j
                        dx = inner[i][0] - inner[j][0]
                        dy = inner[i][1] - inner[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        if dist > 0:
                            inner[i][0] += 0.01 * dx / dist
                            inner[i][1] += 0.01 * dy / dist
                        else:
                            inner[i][0] += 0.01
            # Re-check
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not hexagons_disjoint_fast(inner[i], inner[j]):
                        valid = False
                        break
                if not valid: break
            if valid:
                break
        
        if not valid:
            inner = [[0.0, 0.0, 0.0]] * N
            center = (0.0, 0.0)
            side = 10.0
            angle = 0.0
        else:
            # Recompute side after repair
            side = compute_min_side_fast(inner, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END
