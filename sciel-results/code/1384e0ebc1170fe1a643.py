# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_dist_to_hexagon_edge(px, py, verts):
    """Minimum distance from point (px,py) to any edge of polygon with vertices verts (n,2)."""
    n = len(verts)
    min_d = np.inf
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        t = np.clip(np.dot([px, py] - p1, edge) / np.dot(edge, edge), 0, 1)
        closest = p1 + t * edge
        d = np.hypot(px - closest[0], py - closest[1])
        if d < min_d:
            min_d = d
    return min_d


def _hexagon_overlap_penalty(verts1, verts2, margin=0.0):
    """
    Compute overlap penalty between two hexagons using SAT (separating axis test).
    Returns 0 if separated, positive if overlapping.
    """
    edges = []
    for v in [verts1, verts2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            normal = np.array([-edge[1], edge[0]]) / edge_len
            edges.append(normal)
    
    max_overlap = 0.0
    for normal in edges:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        overlap = min(max1, max2) - max(min1, min2)
        if overlap > margin:
            max_overlap = max(max_overlap, overlap - margin)
    return max_overlap


def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside hexagon with vertices outer_verts using cross products."""
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True


def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Start with a reasonable initial configuration
    # Use a hexagonal lattice arrangement
    inner_positions = []
    # Center
    inner_positions.append([0.0, 0.0])
    # 6 neighbors
    for i in range(6):
        angle = np.radians(60 * i)
        inner_positions.append([2.5 * np.cos(angle), 2.5 * np.sin(angle)])
    # 6 more in second ring
    for i in range(6):
        angle = np.radians(60 * i + 30)
        inner_positions.append([4.33 * np.cos(angle), 4.33 * np.sin(angle)])
    
    inner_positions = np.array(inner_positions)
    angles = np.zeros(12)
    
    # Initial outer hexagon
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    outer_side = 8.0
    
    best_inner = inner_positions.copy()
    best_angles = angles.copy()
    best_outer_center = outer_center.copy()
    best_outer_angle = outer_angle
    best_outer_side = outer_side
    best_score = 1.0 / outer_side
    
    def evaluate_config(positions, ang_arr, o_center, o_angle, o_side):
        """Evaluate a configuration, return (is_valid, score)."""
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(ang_arr)):
            return False, 0.0
        if o_side < np.sqrt(12) - 1e-6:
            return False, 0.0
        
        # Compute outer hexagon vertices
        outer_verts = _hex_vertices(o_center[0], o_center[1], o_side, o_angle)
        
        # Check outer vertices are distinct
        for i in range(6):
            for j in range(i+1, 6):
                if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-8:
                    return False, 0.0
        
        # Check each inner hexagon
        inner_verts_list = []
        for i in range(12):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, ang_arr[i])
            inner_verts_list.append(verts)
            
            # Check all vertices are inside outer hexagon
            for v in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < -1e-6:
                        return False, 0.0
        
        # Check pairwise non-overlap
        for i in range(12):
            for j in range(i+1, 12):
                overlap = _hexagon_overlap_penalty(inner_verts_list[i], inner_verts_list[j], margin=0.0)
                if overlap > 1e-6:
                    return False, 0.0
        
        return True, 1.0 / o_side
    
    # Check initial config
    valid, score = evaluate_config(inner_positions, angles, outer_center, outer_angle, outer_side)
    if valid:
        best_inner = inner_positions.copy()
        best_angles = angles.copy()
        best_outer_center = outer_center.copy()
        best_outer_angle = outer_angle
        best_outer_side = outer_side
        best_score = score
    
    # Optimization: use gradient-free approach with random perturbations
    # Try to minimize outer hexagon size while keeping all hexagons inside and non-overlapping
    
    # First, find minimum enclosing hexagon for current positions
    def compute_min_enclosing_hexagon(positions, ang_arr):
        """Find minimum side length hexagon (axis-aligned, centered at centroid) enclosing all hexagons."""
        centroid = positions.mean(axis=0)
        
        # For a hexagon with side s centered at centroid, the apothem (distance from center to edge) is s * sqrt(3)/2
        # For each inner hexagon vertex, compute required s
        max_required = 0.0
        for i in range(12):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, ang_arr[i])
            for v in verts:
                # Distance from centroid to this vertex
                d = np.linalg.norm(v - centroid)
                # For a regular hexagon with side s, the maximum distance from center to vertex is s
                # But we need the vertex to be inside, so we need s >= d
                # Actually for a hexagon with side s, vertices are at distance s from center
                # So we need s >= d
                max_required = max(max_required, d)
        
        # Add small margin
        return max_required + 0.01, centroid
    
    # Try optimization with simulated annealing / random search
    rng = np.random.default_rng(42)
    
    # Strategy: iteratively try to shrink outer hexagon and reposition inner hexagons
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Try to find a better configuration
        # Use a local search: perturb positions and try to fit in smaller outer hexagon
        
        # Compute current bounding
        centroid = inner_positions.mean(axis=0)
        
        # Try shrinking outer hexagon slightly
        shrink_factor = 0.995
        new_outer_side = best_outer_side * shrink_factor
        
        # Try to reposition inner hexagons to fit in smaller outer hexagon
        # Use a simple relaxation: push overlapping hexagons apart and pull toward center
        new_positions = inner_positions.copy()
        new_angles = angles.copy()
        
        # Apply forces
        forces = np.zeros_like(new_positions)
        
        # Repulsion between overlapping hexagons
        for i in range(12):
            for j in range(i+1, 12):
                vi = _hex_vertices(new_positions[i, 0], new_positions[i, 1], 1.0, new_angles[i])
                vj = _hex_vertices(new_positions[j, 0], new_positions[j, 1], 1.0, new_angles[j])
                overlap = _hexagon_overlap_penalty(vi, vj, margin=0.0)
                if overlap > 0:
                    # Push apart
                    diff = new_positions[i] - new_positions[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-8:
                        diff = np.array([rng.standard_normal(2), rng.standard_normal(2)])
                        dist = np.linalg.norm(diff)
                    force = diff / dist * overlap * 10.0
                    forces[i] += force
                    forces[j] -= force
        
        # Attraction toward center (to keep compact)
        for i in range(12):
            forces[i] += (centroid - new_positions[i]) * 0.1
        
        # Constraint: keep vertices inside outer hexagon
        outer_verts = _hex_vertices(0, 0, new_outer_side, 0)
        for i in range(12):
            verts = _hex_vertices(new_positions[i, 0], new_positions[i, 1], 1.0, new_angles[i])
            for v in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < 0:
                        # Point is outside, push back
                        normal = np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])])
                        normal = normal / np.linalg.norm(normal)
                        forces[i] += normal * (-cross) * 5.0
        
        # Update positions
        step_size = 0.1
        new_positions += forces * step_size
        
        # Center the configuration
        new_centroid = new_positions.mean(axis=0)
        new_positions -= new_centroid
        
        # Evaluate new configuration
        valid, new_score = evaluate_config(new_positions, new_angles, np.array([0.0, 0.0]), 0.0, new_outer_side)
        
        if valid and new_score > best_score:
            best_inner = new_positions.copy()
            best_angles = new_angles.copy()
            best_outer_center = np.array([0.0, 0.0])
            best_outer_angle = 0.0
            best_outer_side = new_outer_side
            best_score = new_score
            # Accept and continue shrinking
            inner_positions = new_positions.copy()
            angles = new_angles.copy()
        else:
            # Try random perturbation
            if rng.random() < 0.3:
                perturbation = rng.standard_normal((12, 2)) * 0.5
                trial_positions = inner_positions + perturbation
                trial_valid, trial_score = evaluate_config(trial_positions, angles, np.array([0.0, 0.0]), 0.0, best_outer_side)
                if trial_valid and trial_score >= best_score:
                    inner_positions = trial_positions.copy()
                    if trial_score > best_score:
                        best_inner = trial_positions.copy()
                        best_outer_side = best_outer_side
                        best_score = trial_score
    
    # Final evaluation and cleanup
    valid, final_score = evaluate_config(best_inner, best_angles, best_outer_center, best_outer_angle, best_outer_side)
    
    if not valid:
        # Fall back to seed configuration
        inner_hex_data = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0],
        ])
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Build output arrays
    inner_hex_data = np.column_stack([best_inner, best_angles])
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return inner_hex_data, outer_hex_data, best_outer_side


# EVOLVE-BLOCK-END
