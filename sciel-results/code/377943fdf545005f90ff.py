# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _check_validity(inner, outer_center, outer_angle, s):
    """
    Strict validity check.
    Returns True if valid, False otherwise.
    1. Finite values.
    2. s >= sqrt(12).
    3. Inner hexagons do not overlap.
    4. Inner hexagons are inside outer hexagon.
    """
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_center)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
    
    # Check each inner hexagon
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        
        # Check if all vertices are inside outer hexagon
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                # cross(p2-p1, v-p1) must be >= -1e-6
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False

        # Check overlap with other inner hexagons
        for j in range(i+1, 12):
            cx2, cy2, ang2 = inner[j]
            verts2 = _hex_vertices(cx2, cy2, 1.0, ang2)
            
            # Quick distance check
            dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
            if dist > 3.5:
                continue

            # Separating Axis Test
            normals = []
            for v in [verts, verts2]:
                for k in range(6):
                    p1 = v[k]
                    p2 = v[(k+1)%6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm_n = np.linalg.norm(normal)
                    if norm_n < 1e-9: continue
                    normal = normal / norm_n
                    normals.append(normal)
            
            overlap = True
            for normal in normals:
                proj_i = np.dot(verts, normal)
                proj_j = np.dot(verts2, normal)
                min_i, max_i = np.min(proj_i), np.max(proj_i)
                min_j, max_j = np.min(proj_j), np.max(proj_j)
                
                if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                    overlap = False
                    break
            
            if overlap:
                return False

    return True

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    shifted_centers = inner[:, :2] - outer_center
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(shifted_centers[i, 0], shifted_centers[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a_rad = np.radians(outer_angle)
    unit_outer_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    max_proj = 0
    for k in range(6):
        p1 = unit_outer_verts[k]
        p2 = unit_outer_verts[(k+1)%6]
        edge = p2 - p1
        # Outward normal
        normal = np.array([edge[1], -edge[0]])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        
        # Ensure normal points outward
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
            
        projs = all_verts @ normal
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    min_s = max_proj * 2.0 / np.sqrt(3)
    return max(min_s, np.sqrt(12))

def _objective(params, outer_center):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., x12, y12, a12, outer_angle]
    Returns s + penalty if invalid.
    """
    inner = params[:36].reshape(12, 3).copy()
    outer_angle = params[36]
    
    s = _compute_min_outer_side(inner, outer_center, outer_angle)
    
    # Check validity for penalty
    if not _check_validity(inner, outer_center, outer_angle, s):
        # Calculate penalty based on overlap depth and out-of-bounds
        penalty = 0.0
        outer_verts = _hex_vertices(outer_center[0], outer_center[1], s, outer_angle)
        
        for i in range(12):
            verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            
            # Out of bounds penalty
            for v in verts:
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k+1)%6]
                    cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                    if cross < 0:
                        penalty += -cross
            
            # Overlap penalty (simplified: distance based)
            for j in range(i+1, 12):
                dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                min_dist = 2 * np.sqrt(3) # Approximate min distance for non-overlap
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
                    
        return s + penalty * 10.0
        
    return s

def _generate_seeds():
    """Generate multiple diverse seeds."""
    seeds = []
    
    # Seed 1: Honeycomb (1 center, 6 ring, 5 second ring)
    L = 2 * np.sqrt(3) + 0.1
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    # 5 points from second ring
    second_ring_coords = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)]
    for (q, r) in second_ring_coords:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 2: 3x4 Lattice
    L = 2 * np.sqrt(3) + 0.1
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * np.sqrt(3)/2 * L
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 3: Rotated Honeycomb
    pts = seeds[0].copy()
    # Rotate angles slightly
    pts[:, 2] += 15.0
    seeds.append(pts)
    
    # Seed 4: Random Cluster
    np.random.seed(123)
    points = [[0.0, 0.0, 0.0]]
    for _ in range(11):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        a = np.random.uniform(0, 60)
        points.append([x, y, a])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600.0
    
    outer_center = np.array([0.0, 0.0])
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate seeds first
    for seed in seeds:
        for angle_deg in [0, 30]:
            s = _compute_min_outer_side(seed, outer_center, angle_deg)
            if _check_validity(seed, outer_center, angle_deg, s):
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_outer_angle = angle_deg
                    
    if best_inner is None:
        # Fallback if no seed is valid
        best_inner = seeds[0].copy()
        best_s = _compute_min_outer_side(best_inner, outer_center, 0)
        best_outer_angle = 0

    # Multi-Start Local Search
    # Use Nelder-Mead for robustness with non-smooth penalty
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break
            
        # Initialize from seed
        x0 = np.concatenate([seed.flatten(), [0.0]])
        
        # Run optimization
        try:
            result = minimize(
                _objective, 
                x0, 
                args=(outer_center,),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s = _compute_min_outer_side(candidate_inner, outer_center, candidate_angle)
            
            if _check_validity(candidate_inner, outer_center, candidate_angle, candidate_s):
                if candidate_s < best_s:
                    best_s = candidate_s
                    best_inner = candidate_inner
                    best_outer_angle = candidate_angle
                    
        except Exception:
            pass

    # Final Refinement on the best solution found
    if time.time() < deadline - 10.0 and best_inner is not None:
        x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        try:
            result = minimize(
                _objective, 
                x0, 
                args=(outer_center,),
                method='Powell',
                options={
                    'maxiter': 500,
                    'xtol': 1e-8,
                    'ftol': 1e-8
                }
            )
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s = _compute_min_outer_side(candidate_inner, outer_center, candidate_angle)
            
            if _check_validity(candidate_inner, outer_center, candidate_angle, candidate_s):
                if candidate_s < best_s:
                    best_s = candidate_s
                    best_inner = candidate_inner
                    best_outer_angle = candidate_angle
        except Exception:
            pass

    # Final validity check
    if not _check_validity(best_inner, outer_center, best_outer_angle, best_s):
        # Fallback to first seed
        best_inner = seeds[0].copy()
        best_s = _compute_min_outer_side(best_inner, outer_center, 0)
        best_outer_angle = 0

    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
