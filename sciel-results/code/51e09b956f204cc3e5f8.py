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
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normals[i] = normal / norm
    return normals

def _sat_overlap(verts1, normals1, verts2, normals2):
    """
    Check if two convex polygons overlap using SAT.
    Returns (overlaps, min_separation) where min_separation is the maximum
    negative separation (penetration depth) across all axes, or positive if separated.
    """
    all_normals = np.vstack([normals1, normals2])
    max_separation = -np.inf  # We want the axis with least separation (most overlap)
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # Separation: if max1 < min2, separation = min2 - max1 (positive = separated)
        # If max2 < min1, separation = min1 - max2
        # Otherwise, separation is negative (overlap)
        sep1 = min2 - max1  # positive if 1 is before 2
        sep2 = min1 - max2  # positive if 2 is before 1
        # The actual separation on this axis is max(sep1, sep2) if both positive,
        # but we care about the minimum separation (most overlap)
        # If both are negative, they overlap on this axis
        # The separation is max(sep1, sep2) - if this is > 0, separated on this axis
        sep = max(sep1, sep2)
        if sep > max_separation:
            max_separation = sep
    # max_separation > 0 means separated, <= 0 means overlapping
    overlaps = max_separation <= 1e-6
    return overlaps, max_separation

def _compute_all_verts_normals(inner):
    """Compute all vertices and normals for the 12 inner hexagons."""
    centers = inner[:, :2]
    angles = inner[:, 2]
    
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    
    for i in range(12):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, angles[i])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    
    return all_verts, all_normals

def _compute_s_and_overlap_fast(inner, outer_angle):
    """
    Fast computation of required outer hexagon side length and overlap check.
    Returns (s, is_valid) where is_valid is True if no overlaps.
    """
    centers = inner[:, :2]
    angles = inner[:, 2]
    
    all_verts, all_normals = _compute_all_verts_normals(inner)
    
    a_rad = np.radians(outer_angle)
    outer_unit_verts = np.column_stack([
        np.cos(a_rad + np.radians(60) * np.arange(6)),
        np.sin(a_rad + np.radians(60) * np.arange(6))
    ])
    outer_normals = _edge_normals(outer_unit_verts)
    
    all_inner_verts_flat = all_verts.reshape(72, 2)
    
    max_proj = 0.0
    for normal in outer_normals:
        projs = np.dot(all_inner_verts_flat, normal)
        max_k = np.max(projs)
        if max_k > max_proj:
            max_proj = max_k
    
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlaps, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlaps:
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _resolve_collisions(inner, outer_angle, max_iter=50):
    """
    Resolve collisions between inner hexagons using iterative projection.
    Returns (resolved_inner, was_resolved).
    """
    current = inner.copy()
    for _ in range(max_iter):
        centers = current[:, :2]
        all_verts, all_normals = _compute_all_verts_normals(current)
        
        resolved = True
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlaps, min_sep = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlaps:
                    resolved = False
                    # Find the axis of minimum separation
                    all_normals_ij = np.vstack([all_normals[i], all_normals[j]])
                    best_axis = None
                    best_sep = np.inf
                    for normal in all_normals_ij:
                        proj1 = np.dot(all_verts[i], normal)
                        proj2 = np.dot(all_verts[j], normal)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        sep = max(min2 - max1, min1 - max2)
                        if sep < best_sep:
                            best_sep = sep
                            best_axis = normal
                    
                    if best_axis is not None and best_sep < 0:
                        # Move apart along the axis
                        # Determine which hex is "behind" on this axis
                        proj1 = np.dot(all_verts[i], best_axis)
                        proj2 = np.dot(all_verts[j], best_axis)
                        if np.mean(proj1) < np.mean(proj2):
                            # i is behind, push i back and j forward
                            move = (-best_sep - 1e-7) / 2.0
                            current[i, :2] -= best_axis * move
                            current[j, :2] += best_axis * move
                        else:
                            # j is behind
                            move = (-best_sep - 1e-7) / 2.0
                            current[j, :2] -= best_axis * move
                            current[i, :2] += best_axis * move
        
        if resolved:
            break
    
    # Verify no overlaps remain
    centers = current[:, :2]
    all_verts, all_normals = _compute_all_verts_normals(current)
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            overlaps, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlaps:
                return inner.copy(), False
    
    return current, True

def _generate_lattice_seed(rotation_deg, scale=1.0):
    """
    Generate a seed based on a triangular lattice with 12 hexagons.
    The 12 positions are chosen to form a compact hexagonal cluster.
    rotation_deg: global rotation of the lattice
    scale: spacing multiplier
    """
    # Triangular lattice basis vectors
    # Standard spacing for non-overlapping unit hexagons is 2*sqrt(3)
    L = 2 * np.sqrt(3) * scale
    
    # Define 12 lattice points in axial coordinates (q, r)
    # These form a hexagonal shape of radius 2 (1+6+5=12 points)
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,1) -- wait, need 5 more to make 12 total
    # Let's use a specific set of 12 points that form a nice cluster
    
    # Points for a "flower" shape: center + 6 neighbors + 5 from next ring
    # To make it symmetric and compact, let's pick:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (-2,0), (0,-2), (2,-2) -- this is getting asymmetric
    
    # Better: use the standard hexagonal packing of 12.
    # A regular hexagon of side length 2 (in lattice units) has 1+6+12=19 points.
    # We only have 12.
    # Let's try: Center (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 5 more: (2,0), (1,1), (0,2), (-1,2), (-2,1) -- this creates a "Pac-Man" shape
    
    # Actually, for optimization, the exact initial topology matters less if we allow free movement.
    # But the lattice constraint helps.
    # Let's stick to the 1+6+5 pattern but make it as symmetric as possible.
    # Or just use the 12 points from the previous seed generation which worked well.
    
    # Let's reuse the "Hexagonal cluster" logic but apply rotation and scale.
    points = [[0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle)])
    # Second ring: 5 points
    # To maintain symmetry, we can't just pick 5. 
    # Let's pick the 6 points of the second ring and drop one? No, that breaks symmetry.
    # Let's use the 12 points from the previous code's seed 1, which was:
    # Center + 6 + 5 specific ones.
    # Let's just generate the full 19-point hex and take the first 12?
    # No, let's use the same 12 points as the previous "Seed 1" but rotated.
    
    # Seed 1 points in previous code:
    # (0,0)
    # 6 neighbors at distance L
    # 5 points: (2,0), (1,1), (0,2), (-1,1), (-2,0) in some coordinate system
    
    # Let's explicitly define the 12 axial coordinates (q, r) for a compact shape
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    # Convert axial to cartesian
    # x = L * (q + r/2)
    # y = L * r * sqrt(3)/2
    cart_points = []
    for q, r in coords:
        x = L * (q + r / 2.0)
        y = L * r * np.sqrt(3) / 2
        cart_points.append([x, y])
    
    pts = np.array(cart_points)
    
    # Center the points
    center = np.mean(pts, axis=0)
    pts -= center
    
    # Apply rotation
    theta = np.radians(rotation_deg)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    pts = pts @ rot_matrix.T
    
    # Create inner array (x, y, angle)
    # The angle of each hexagon should match the lattice rotation to maintain alignment
    inner = np.zeros((12, 3))
    inner[:, 0] = pts[:, 0]
    inner[:, 1] = pts[:, 1]
    inner[:, 2] = rotation_deg
    
    return inner

def _generate_seeds():
    """Generate multiple initial configurations."""
    seeds = []
    
    # Seed 1: Hexagonal cluster (center + 6 + 5) - Original
    L = 2 * np.sqrt(3) + 0.01
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r/2.0)
        y = L * r * np.sqrt(3)/2
        points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 2: Triangular lattice 3x4
    L = 2 * np.sqrt(3) + 0.01
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 3: Two rows of 6
    L = 2 * np.sqrt(3) + 0.01
    points = []
    for row in range(2):
        for col in range(6):
            x = col * L
            y = row * L * np.sqrt(3)
            points.append([x, y, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 4: Lattice-based with rotation
    seeds.append(_generate_lattice_seed(0.0))
    seeds.append(_generate_lattice_seed(30.0))
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid and s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Collision-resolution-based local search
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    np.random.seed(42)
    n_cr_iterations = 30000
    
    for it in range(n_cr_iterations):
        if time.time() > deadline - 150.0:
            break
        
        # Generate a perturbation
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        n_perturb = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        indices = np.random.choice(12, size=n_perturb, replace=False)
        for idx in indices:
            new_inner[idx, 0] += np.random.randn() * 0.15
            new_inner[idx, 1] += np.random.randn() * 0.15
            new_inner[idx, 2] += np.random.randn() * 0.05
        
        if np.random.rand() < 0.2:
            new_outer_angle += np.random.randn() * 0.05
        
        # Try to resolve collisions
        resolved_inner, was_resolved = _resolve_collisions(new_inner, new_outer_angle)
        
        if was_resolved:
            new_s, new_valid = _compute_s_and_overlap_fast(resolved_inner, new_outer_angle)
            if new_valid:
                if new_s < current_s:
                    current_inner = resolved_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    # Accept worse solutions with some probability (annealing-like)
                    T = 0.02 * (0.99 ** it)
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                        current_inner = resolved_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
        else:
            # Collision couldn't be resolved, reject
            pass
    
    # Phase 2: Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 30.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
        return s if is_valid else s + 10.0
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    for restart in range(5):
        if time.time() > deadline - 40.0:
            break
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(37) * 0.005
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate_inner = result.x[:36].reshape(12, 3).copy()
            candidate_angle = result.x[36]
            candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, candidate_angle)
            if candidate_valid and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = candidate_angle
                x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with collision resolution
    if time.time() < deadline - 50.0:
        np.random.seed(123)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(20000):
            if time.time() > deadline - 50.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.01
            new_inner[idx, 1] += np.random.randn() * 0.01
            new_inner[idx, 2] += np.random.randn() * 0.005
            if np.random.rand() < 0.3:
                new_outer_angle += np.random.randn() * 0.005
            
            # Try collision resolution
            resolved_inner, was_resolved = _resolve_collisions(new_inner, new_outer_angle, max_iter=20)
            
            if was_resolved:
                new_s, new_valid = _compute_s_and_overlap_fast(resolved_inner, new_outer_angle)
                if new_valid:
                    if new_s < current_s:
                        current_inner = resolved_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
                        if current_s < best_s:
                            best_inner = current_inner.copy()
                            best_s = current_s
                            best_outer_angle = current_outer_angle
                    else:
                        T_fine = 0.005 * (0.999 ** it)
                        delta = new_s - current_s
                        if np.random.rand() < np.exp(-delta / max(T_fine, 1e-10)):
                            current_inner = resolved_inner
                            current_s = new_s
                            current_outer_angle = new_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
