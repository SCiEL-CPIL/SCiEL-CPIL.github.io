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
    Returns True if they overlap (invalid), False if separated (valid).
    """
    all_normals = np.vstack([normals1, normals2])
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        # If projections don't overlap on this axis, shapes are separated
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return False  # Separated, no overlap
    return True  # No separating axis found, they overlap

def _compute_all_verts_normals(inner):
    """Precompute vertices and normals for all inner hexagons."""
    all_verts = np.zeros((12, 6, 2))
    all_normals = np.zeros((12, 6, 2))
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        all_verts[i] = verts
        all_normals[i] = _edge_normals(verts)
    return all_verts, all_normals

def _compute_s_and_overlap_fast(inner, outer_angle):
    """
    Compute required outer hexagon side length and check overlaps.
    Returns (s, is_valid).
    """
    centers = inner[:, :2]
    
    all_verts, all_normals = _compute_all_verts_normals(inner)
    
    # Compute s: outer hexagon must contain all inner vertices
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
    
    # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
    # max_proj = s * sqrt(3)/2 => s = max_proj * 2/sqrt(3)
    s = max_proj * 2.0 / np.sqrt(3)
    s = max(s, np.sqrt(12))
    
    # Check overlaps using SAT
    is_valid = True
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            if _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _resolve_collisions(inner, max_iters=200):
    """
    Resolve collisions between hexagons by pushing them apart.
    Returns modified inner array.
    """
    inner = inner.copy()
    for _ in range(max_iters):
        all_verts, all_normals = _compute_all_verts_normals(inner)
        moved = False
        for i in range(12):
            for j in range(i+1, 12):
                if _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j]):
                    # Find the axis of minimum overlap
                    all_normals = np.vstack([all_normals[i], all_normals[j]])
                    min_overlap = float('inf')
                    best_normal = None
                    for normal in all_normals:
                        proj1 = np.dot(all_verts[i], normal)
                        proj2 = np.dot(all_verts[j], normal)
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        overlap = min(max1, max2) - max(min1, min2)
                        if overlap < min_overlap:
                            min_overlap = overlap
                            best_normal = normal
                    
                    if best_normal is not None and min_overlap > 0:
                        push = 0.5 * (min_overlap + 1e-4)
                        # Determine direction: push i away from j
                        diff = inner[i, :2] - inner[j, :2]
                        if np.dot(diff, best_normal) < 0:
                            best_normal = -best_normal
                        inner[i, 0] += push * best_normal[0]
                        inner[i, 1] += push * best_normal[1]
                        inner[j, 0] -= push * best_normal[0]
                        inner[j, 1] -= push * best_normal[1]
                        moved = True
        
        if not moved:
            break
    
    return inner

def _generate_seeds():
    """Generate initial configurations using hexagonal lattice patterns."""
    seeds = []
    
    # Seed 1: Compact hexagonal cluster (center + ring1 + part of ring2)
    L = 2 * np.sqrt(3) + 0.01
    points = [[0.0, 0.0, 0.0]]
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    
    # Add 5 from second ring (axial coords)
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r / 2.0)
        y = L * r * np.sqrt(3) / 2
        points.append([x, y, 0.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 2: 3x4 triangular lattice
    L = 2 * np.sqrt(3) + 0.01
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row / 2.0) * L
            y = row * L * np.sqrt(3) / 2
            points.append([x, y, 0.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 3: 4x3 triangular lattice (transposed)
    L = 2 * np.sqrt(3) + 0.01
    points = []
    for row in range(4):
        for col in range(3):
            x = (col + row / 2.0) * L
            y = row * L * np.sqrt(3) / 2
            points.append([x, y, 0.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 4: Hexagonal cluster with 30-degree rotation
    L = 2 * np.sqrt(3) + 0.01
    points = [[0.0, 0.0, 30.0]]
    for k in range(6):
        angle = np.radians(60 * k + 30)
        points.append([L * np.cos(angle), L * np.sin(angle), 30.0])
    
    second_ring = [(2, 0), (1, 1), (0, 2), (-1, 1), (-2, 0)]
    for (q, r) in second_ring:
        x = L * (q + r / 2.0)
        y = L * r * np.sqrt(3) / 2
        points.append([x, y, 30.0])
    
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 5: Two rows of 6
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
    
    return seeds

def _lattice_packing(basis_a, basis_b, global_angle_deg, axial_coords):
    """
    Generate hexagon positions from a triangular lattice.
    basis_a, basis_b: 2D vectors for lattice basis
    global_angle_deg: rotation applied to all hexagons
    axial_coords: list of (q, r) axial coordinates
    """
    positions = []
    for (q, r) in axial_coords:
        pos = q * basis_a + r * basis_b
        positions.append([pos[0], pos[1], global_angle_deg])
    
    pts = np.array(positions)
    # Center the configuration
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    return pts

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    # Evaluate all seeds
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid:
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: Lattice-based optimization
    # Optimize lattice basis vectors and global angle
    from scipy.optimize import minimize
    
    # Axial coordinates for 12 hexagons in a compact hexagonal arrangement
    # Center + ring1(6) + 5 from ring2
    axial_coords = [
        (0, 0),      # center
        (1, 0), (1, -1), (0, -1),  # ring1 (3 of 6)
        (-1, 0), (-1, 1), (0, 1),  # ring1 (other 3)
        (2, 0), (2, -1), (1, -2),  # ring2 (3)
        (-2, 0), (-1, 1),          # ring2 (2 more to reach 12)
    ]
    # That's 1+6+5 = 12
    
    def lattice_objective(params):
        if time.time() > deadline - 30:
            return 1e10
        # params: [ax, ay, bx, by, global_angle]
        basis_a = np.array([params[0], params[1]])
        basis_b = np.array([params[2], params[3]])
        global_angle = params[4]
        
        inner = _lattice_packing(basis_a, basis_b, global_angle, axial_coords)
        s, is_valid = _compute_s_and_overlap_fast(inner, 0)
        
        if not is_valid:
            # Add penalty for overlaps
            return s + 5.0
        return s
    
    # Initial guess: standard hexagonal lattice
    L0 = 2 * np.sqrt(3)
    x0 = np.array([L0, 0.0, L0/2, L0*np.sqrt(3)/2, 0.0])
    
    # Multiple restarts for lattice optimization
    np.random.seed(42)
    for restart in range(8):
        if time.time() > deadline - 120:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            # Perturb the lattice parameters
            perturbation = np.random.randn(5) * 0.3
            x_start = x0 + perturbation
        
        try:
            result = minimize(
                lattice_objective, x_start,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            basis_a = np.array([result.x[0], result.x[1]])
            basis_b = np.array([result.x[2], result.x[3]])
            global_angle = result.x[4]
            
            candidate_inner = _lattice_packing(basis_a, basis_b, global_angle, axial_coords)
            candidate_s, candidate_valid = _compute_s_and_overlap_fast(candidate_inner, 0)
            
            if candidate_valid and candidate_s < best_s:
                best_inner = candidate_inner
                best_s = candidate_s
                best_outer_angle = 0
        except Exception:
            pass
    
    # Phase 2: Free-position refinement with collision resolution
    if time.time() < deadline - 80:
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        T = 0.02
        n_iters = 15000
        
        for it in range(n_iters):
            if time.time() > deadline - 80:
                break
            
            new_inner = current_inner.copy()
            
            # Perturb 1-2 hexagons
            n_perturb = np.random.choice([1, 2], p=[0.6, 0.4])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.05
                new_inner[idx, 1] += np.random.randn() * 0.05
                new_inner[idx, 2] += np.random.randn() * 0.02
            
            # Resolve collisions
            new_inner = _resolve_collisions(new_inner, max_iters=50)
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, current_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(T, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
            else:
                if np.random.rand() < 0.02:
                    current_inner = new_inner
                    current_s = new_s + 0.3
            
            # Cool down
            T *= 0.9995
        
        # Optimize outer angle
        if time.time() < deadline - 40:
            for angle_deg in np.arange(-15, 16, 1):
                s_test, valid_test = _compute_s_and_overlap_fast(best_inner, angle_deg)
                if valid_test and s_test < best_s:
                    best_s = s_test
                    best_outer_angle = angle_deg
    
    # Phase 3: Fine-tuning
    if time.time() < deadline - 30:
        T_fine = 0.003
        n_fine = 8000
        np.random.seed(123)
        
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(n_fine):
            if time.time() > deadline - 30:
                break
            
            new_inner = current_inner.copy()
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.003
            new_inner[idx, 1] += np.random.randn() * 0.003
            new_inner[idx, 2] += np.random.randn() * 0.001
            
            new_inner = _resolve_collisions(new_inner, max_iters=30)
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, current_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
                else:
                    delta = new_s - current_s
                    if np.random.rand() < np.exp(-delta / max(T_fine, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
