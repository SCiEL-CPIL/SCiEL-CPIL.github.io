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

    def get_axes(vertices):
        """Get the 6 normal axes for a hexagon."""
        axes = []
        for i in range(6):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal /= norm
                axes.append(normal)
        return axes

    def sat_overlap_penalty(v1, v2, axes1, axes2):
        """
        Compute a smooth penalty for overlap between two hexagons.
        Returns 0 if disjoint, positive value if overlapping.
        """
        min_gap = np.inf
        for axis in axes1:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Gap is negative if overlapping
            gap = min(max1, max2) - max(min1, min2)
            if gap < min_gap:
                min_gap = gap
                
        for axis in axes2:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = min(max1, max2) - max(min1, min2)
            if gap < min_gap:
                min_gap = gap
                
        if min_gap < 0:
            # Overlapping. Penalty is square of the "depth"
            # min_gap is the most "separating" axis, but if it's negative, they overlap.
            # The magnitude indicates how deep the overlap is on that axis.
            return min_gap * min_gap
        else:
            return 0.0

    def compute_min_side(inner, center, angle):
        """Compute the minimum outer hexagon side length required to contain all inner hexagons."""
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        
        # Precompute all vertices for all inner hexagons
        all_verts = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
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

    def check_validity(inner):
        """Check if all hexagons are pairwise disjoint using SAT."""
        vertices_list = []
        axes_list = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vertices_list.append(v)
            axes_list.append(get_axes(v))
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = vertices_list[i]
                v2 = vertices_list[j]
                axes1 = axes_list[i]
                axes2 = axes_list[j]
                
                disjoint = False
                for axis in axes1:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                        disjoint = True
                        break
                if disjoint:
                    continue
                for axis in axes2:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                        disjoint = True
                        break
                if not disjoint:
                    return False
        return True

    def objective(params, lambda_penalty):
        """
        Objective: Minimize outer side length + lambda * overlap penalty.
        params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        """
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        center = (params[3*N], params[3*N+1])
        angle = params[3*N+2]
        
        # Compute overlap penalty
        penalty = 0.0
        vertices_list = []
        axes_list = []
        for i in range(N):
            v = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            vertices_list.append(v)
            axes_list.append(get_axes(v))
            
        for i in range(N):
            for j in range(i + 1, N):
                p = sat_overlap_penalty(vertices_list[i], vertices_list[j], axes_list[i], axes_list[j])
                penalty += p
                
        # Compute side
        side = compute_min_side(inner, center, angle)
        
        return side + lambda_penalty * penalty

    def initial_guess():
        params = np.zeros(3*N + 3)
        # Hexagonal packing: 1 center, 6 ring 1, 5 ring 2
        positions = []
        positions.append((0, 0, 0))
        for i in range(6):
            theta = i * np.pi / 3
            # Distance between centers of touching hexagons is 2 * sqrt(3) / 2 * sqrt(3)? 
            # Side 1 hexagon: width 2, height sqrt(3). 
            # Touching centers distance: 2 * apothem = 2 * (sqrt(3)/2) * 1 = sqrt(3)? No.
            # Distance between centers of two touching unit hexagons is 2 * (sqrt(3)/2) * 2? 
            # Let's use the lattice coordinates.
            # In a hex lattice, distance between neighbors is 2 * (sqrt(3)/2) * 2? 
            # Actually, distance is 2 * sqrt(3) / 2 * 2? 
            # Side length 1. Width = 2. Height = sqrt(3).
            # If flat-top, horizontal neighbors are 2 apart? No, 2 * cos(30) * 2?
            # Let's just use a safe initial distance, e.g., 3.5
            positions.append((3.5*np.cos(theta), 3.5*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((7.0*np.cos(theta), 7.0*np.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    current_params = best_params.copy()
    
    # Initial penalty high to enforce separation
    lambda_start = 100.0
    lambda_end = 1.0
    
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.5
    max_steps = 50000
    
    step_count = 0
    best_score = objective(current_params, lambda_start)
    best_params = current_params.copy()
    
    while time.time() < deadline and step_count < max_steps:
        step_count += 1
        
        # Anneal lambda
        lambda_current = lambda_start * (lambda_end / lambda_start) ** (step_count / max_steps)
        
        # Perturb one random parameter
        idx = random.randint(0, 3*N + 2)
        current_step = step_size * (0.5 + 0.5 * T)
        new_params = current_params.copy()
        new_params[idx] += random.gauss(0, current_step)
        
        new_score = objective(new_params, lambda_current)
        
        # Accept
        if new_score < best_score or random.random() < np.exp(-(new_score - best_score) / T):
            current_params = new_params
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    
    # Final validation
    if not check_validity(inner):
        # If invalid, try to fix by scaling out or fallback
        # Simple fallback: spread them out linearly
        inner = [[float(i * 3.0), 0.0, 0.0] for i in range(N)]
        center = (0.0, 0.0)
        angle = 0.0
        # Ensure they fit
        # Side length will be large, but valid
        pass

    side = compute_min_side(inner, center, angle)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
