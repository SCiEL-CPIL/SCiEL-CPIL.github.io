# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
    """Compute outward unit normals for edges of a convex polygon given in CCW order."""
    n = len(verts)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i+1) % n]
        edge = p2 - p1
        # For CCW polygon, the outward normal is to the right of the edge direction
        # right normal of (dx, dy) is (dy, -dx)
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _min_separation_distance(verts1, normals1, verts2, normals2):
    """
    Compute the minimum separation between two convex polygons.
    If they overlap, return 0.
    Otherwise, return the minimum distance such that separating axis exists.
    """
    all_normals = np.vstack([normals1, normals2])
    min_sep = float('inf')
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            # Overlap along this axis
            return 0.0
        
        if sep < min_sep:
            min_sep = sep
    
    return min_sep

def _compute_s_and_overlap(inner, outer_angle):
    """
    Compute the required outer hexagon side length s and the total overlap penalty.
    inner: (12, 3) array of (x, y, angle_deg)
    outer_angle: angle of the outer hexagon in degrees
    
    Returns:
        s: minimum side length of outer hexagon
        penalty: sum of squared overlaps (0 if no overlaps)
    """
    centers = inner[:, :2]
    angles = inner[:, 2]
    
    # Compute all inner vertices
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    
    # Compute s: maximum projection onto outer hexagon outward normals
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    
    outer_normals = _edge_normals(outer_unit_verts)
    
    # For each outer normal, find max projection of all inner vertices
    # The distance from center to edge is sqrt(3)/2 for unit hexagon
    # So s * sqrt(3)/2 >= max_proj
    # s >= max_proj * 2 / sqrt(3)
    
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            # Quick check: if center distance is too large, no overlap
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.5:
                continue
            
            sep = _min_separation_distance(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if sep < 0.001:
                # Overlap or very close
                penalty += max(0.0, 0.001 - sep) ** 2
    
    return s, penalty

def _generate_seeds():
    """Generate multiple initial configurations."""
    seeds = []
    
    # Seed 1: Triangular lattice (3x4)
    L = 2 * np.sqrt(3) + 0.05
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r/2.0) * L
            y = r * np.sqrt(3)/2 * L
            points.append([x, y, 0.0])
    
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    seeds.append(np.array(points))
    
    # Seed 2: Hexagonal cluster
    # Central hexagon + 6 surrounding + 5 more
    # Actually, let's use a proper hexagonal arrangement
    # For 12 hexagons, we can use a "flower" pattern
    # Center + 6 neighbors + 5 more in second ring
    
    r1 = 2 * np.sqrt(3) + 0.05
    r2 = 2 * np.sqrt(3) + 0.05
    
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([r1 * np.cos(angle), r1 * np.sin(angle), 0.0])
    
    # Add 5 more in second ring
    # Second ring positions (axial coords with radius 2)
    second_ring = [
        (2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0), (-1, -1), (0, -2), (1, -1)
    ]
    # Pick 5 of these 8
    for (q, r) in second_ring[:5]:
        x = (q + r/2.0) * r2
        y = r * np.sqrt(3)/2 * r2
        points.append([x, y, 0.0])
    
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    seeds.append(np.array(points))
    
    # Seed 3: Two rows of 6
    L = 2 * np.sqrt(3) + 0.05
    points = []
    for r in range(2):
        for q in range(6):
            x = q * L
            y = r * np.sqrt(3) * L
            points.append([x, y, 0.0])
    
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    seeds.append(np.array(points))
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0  # Leave 10s safety margin
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate seeds
    for seed in seeds:
        for angle_deg in [0, 30]:
            s, penalty = _compute_s_and_overlap(seed, angle_deg)
            if penalty < 1e-6:
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_outer_angle = angle_deg
    
    if best_inner is None:
        # Fallback
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap(best_inner, 0)
        best_outer_angle = 0
    
    # Simulated Annealing with focus on minimizing s
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.1
    T_min = 0.0001
    n_iterations = 30000
    
    np.random.seed(42)
    
    it = 0
    while it < n_iterations:
        if time.time() > deadline - 60.0:
            break
        
        it += 1
        T = T_initial * (T_min / T_initial) ** (it / n_iterations)
        
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        # Perturb
        n_perturb = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * 0.1
            new_inner[idx, 1] += np.random.randn() * 0.1
            new_inner[idx, 2] += np.random.randn() * 0.05
        
        if np.random.rand() < 0.1:
            new_outer_angle += np.random.randn() * 0.05
        
        new_s, new_penalty = _compute_s_and_overlap(new_inner, new_outer_angle)
        
        if new_penalty < 1e-6:
            # Valid configuration
            if new_s < current_s:
                current_inner = new_inner
                current_s = new_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            else:
                delta = new_s - current_s
                if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
        else:
            # Invalid: accept with small probability if penalty is small
            if new_penalty < 0.01 and np.random.rand() < 0.05:
                current_inner = new_inner
                current_s = new_s + new_penalty
                current_outer_angle = new_outer_angle
    
    # Final refinement with Nelder-Mead - use smooth penalty to handle nonsmoothness
    from scipy.optimize import minimize
    
    def objective_smooth(params):
        if time.time() > deadline - 10.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        
        s, penalty = _compute_s_and_overlap(inner, outer_angle)
        # Use soft penalty: s + alpha * penalty with smooth transition
        # When penalty is small, we're near valid; when large, strong penalty
        return s + 50.0 * penalty
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    # Multi-restart Nelder-Mead with tighter tolerances
    for restart in range(3):
        if time.time() > deadline - 15.0:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb initial point for diversity
            x_start = x0 + np.random.randn(37) * 0.01
        
        try:
            result = minimize(
                objective_smooth, x_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s, candidate_penalty = _compute_s_and_overlap(candidate_inner, candidate_angle)
            
            if candidate_penalty < 1e-6 and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
                x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        except Exception:
            pass
    
    # High-precision polish pass with even tighter tolerance
    if time.time() < deadline - 20.0:
        try:
            result = minimize(
                objective_smooth, x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s, candidate_penalty = _compute_s_and_overlap(candidate_inner, candidate_angle)
            
            if candidate_penalty < 1e-6 and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
        except Exception:
            pass
    
    # Final validity check
    s_final, penalty_final = _compute_s_and_overlap(best_inner, best_outer_angle)
    if penalty_final > 1e-4:
        # Revert to seed
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
