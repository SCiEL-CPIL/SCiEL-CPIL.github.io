# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(verts):
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
    all_normals = np.vstack([normals1, normals2])
    min_overlap = float('inf')
    best_axis = None
    best_sign = 0
    
    for normal in all_normals:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-6:
            return 0.0, None, 0
        elif max2 < min1 - 1e-6:
            return 0.0, None, 0
        
        if max1 < min2:
            overlap = min2 - max1
            sign = 1
        elif max2 < min1:
            overlap = min1 - max2
            sign = -1
        else:
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0, None, 0
            c1 = (min1 + max1) / 2
            c2 = (min2 + max2) / 2
            sign = 1 if c1 > c2 else -1
        
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = normal
            best_sign = sign
    
    return min_overlap, best_axis, best_sign

def _compute_all_verts_normals(inner):
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
    centers = inner[:, :2]
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
            overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
            if overlap > 0:
                is_valid = False
                break
        if not is_valid:
            break
    
    return s, is_valid

def _resolve_collisions(inner, outer_angle, max_iters=50, push_factor=0.6):
    inner = inner.copy()
    
    for _ in range(max_iters):
        all_verts, all_normals = _compute_all_verts_normals(inner)
        centers = inner[:, :2]
        
        any_overlap = False
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist > 2.0 * np.sqrt(3) + 0.1:
                    continue
                overlap, axis, sign = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                if overlap > 1e-8 and axis is not None:
                    any_overlap = True
                    push = sign * axis * overlap * push_factor
                    inner[i, 0] += push[0]
                    inner[i, 1] += push[1]
                    inner[j, 0] -= push[0]
                    inner[j, 1] -= push[1]
        
        if not any_overlap:
            break
    
    return inner

def _generate_seeds():
    seeds = []
    L = 2 * np.sqrt(3) + 0.005
    
    # Seed 1: Hexagonal cluster (center + 6 + 5)
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
    
    # Seed 4: Circular arrangement with center
    points = [[0.0, 0.0, 0.0]]
    for k in range(11):
        angle = np.radians(360.0 * k / 11.0)
        r = L * 1.5
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 5: 4-4-4 grid
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
    
    # Seed 6: Compact hexagonal arrangement (center + 5 around, then 6 more)
    points = [[0.0, 0.0, 0.0]]
    for k in range(5):
        angle = np.radians(72 * k + 18)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    for k in range(6):
        angle = np.radians(60 * k)
        r = L * np.sqrt(3)
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 7: Ring of 6 + inner cluster
    points = []
    for k in range(6):
        angle = np.radians(60 * k)
        points.append([L * 1.5 * np.cos(angle), L * 1.5 * np.sin(angle), 0.0])
    for k in range(6):
        angle = np.radians(60 * k + 30)
        points.append([L * 0.5 * np.cos(angle), L * 0.5 * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 8: 2-3-3-2-2 arrangement
    points = []
    for col in range(2):
        points.append([col * L, 0.0, 0.0])
    for col in range(3):
        points.append([(col + 0.5) * L, L * np.sqrt(3)/2, 0.0])
    for col in range(3):
        points.append([(col + 0.5) * L, L * np.sqrt(3), 0.0])
    for col in range(2):
        points.append([(col + 1.0) * L, L * np.sqrt(3) * 1.5, 0.0])
    for col in range(2):
        points.append([(col + 1.0) * L, L * np.sqrt(3) * 2.0, 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 9: Rotated triangular lattice 3x4
    angle_rot = np.radians(15)
    points = []
    for row in range(3):
        for col in range(4):
            x = (col + row/2.0) * L
            y = row * L * np.sqrt(3)/2
            xr = x * np.cos(angle_rot) - y * np.sin(angle_rot)
            yr = x * np.sin(angle_rot) + y * np.cos(angle_rot)
            points.append([xr, yr, 15.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    # Seed 10: Diamond arrangement
    points = [[0.0, 0.0, 0.0]]
    for k in range(4):
        angle = np.radians(90 * k + 45)
        points.append([L * np.cos(angle), L * np.sin(angle), 0.0])
    for k in range(7):
        angle = np.radians(360.0 * k / 7.0 + 20)
        r = L * 1.8
        points.append([r * np.cos(angle), r * np.sin(angle), 0.0])
    pts = np.array(points)
    center = np.mean(pts[:, :2], axis=0)
    pts[:, 0] -= center[0]
    pts[:, 1] -= center[1]
    seeds.append(pts)
    
    return seeds

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 615.0
    
    seeds = _generate_seeds()
    
    best_inner = None
    best_s = float('inf')
    best_outer_angle = 0.0
    
    for seed in seeds:
        for angle_deg in [0, 30, 15, 45, 60, 10, 20, 50, 25, 55]:
            s, is_valid = _compute_s_and_overlap_fast(seed, angle_deg)
            if is_valid and s < best_s:
                best_s = s
                best_inner = seed.copy()
                best_outer_angle = angle_deg
    
    if best_inner is None:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    # Phase 1: SA with collision resolution
    np.random.seed(42)
    current_inner = best_inner.copy()
    current_s = best_s
    current_outer_angle = best_outer_angle
    
    T_initial = 0.03
    T_min = 0.00003
    n_sa_iterations = 20000
    
    for it in range(n_sa_iterations):
        if time.time() > deadline - 180.0:
            break
        
        T = T_initial * (T_min / T_initial) ** (it / n_sa_iterations)
        new_inner = current_inner.copy()
        new_outer_angle = current_outer_angle
        
        move_type = np.random.rand()
        
        if move_type < 0.45:
            n_perturb = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.08
                new_inner[idx, 1] += np.random.randn() * 0.08
                new_inner[idx, 2] += np.random.randn() * 0.04
            new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=40, push_factor=0.5)
        elif move_type < 0.75:
            n_perturb = np.random.choice([1, 2], p=[0.6, 0.4])
            indices = np.random.choice(12, size=n_perturb, replace=False)
            for idx in indices:
                new_inner[idx, 0] += np.random.randn() * 0.04
                new_inner[idx, 1] += np.random.randn() * 0.04
                new_inner[idx, 2] += np.random.randn() * 0.015
        else:
            idx = np.random.randint(0, 12)
            new_inner[idx, 2] += np.random.randn() * 0.08
            new_inner[idx, 0] += np.random.randn() * 0.025
            new_inner[idx, 1] += np.random.randn() * 0.025
            if np.random.rand() < 0.35:
                new_outer_angle += np.random.randn() * 0.04
        
        new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
        
        if new_valid:
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
            resolved = _resolve_collisions(new_inner, new_outer_angle, max_iters=60, push_factor=0.7)
            resolved_s, resolved_valid = _compute_s_and_overlap_fast(resolved, new_outer_angle)
            if resolved_valid and resolved_s < current_s:
                current_inner = resolved
                current_s = resolved_s
                current_outer_angle = new_outer_angle
                if current_s < best_s:
                    best_inner = current_inner.copy()
                    best_s = current_s
                    best_outer_angle = current_outer_angle
            elif resolved_valid and np.random.rand() < np.exp(-(resolved_s - current_s) / max(T, 1e-10)):
                current_inner = resolved
                current_s = resolved_s
                current_outer_angle = new_outer_angle
    
    # Phase 2: Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - 50.0:
            return 1e10
        inner = params[:36].reshape(12, 3).copy()
        outer_angle = params[36]
        s, is_valid = _compute_s_and_overlap_fast(inner, outer_angle)
        if is_valid:
            return s
        else:
            all_verts, all_normals = _compute_all_verts_normals(inner)
            penalty = 0.0
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.linalg.norm(inner[i, :2] - inner[j, :2])
                    if dist > 2.0 * np.sqrt(3) + 0.1:
                        continue
                    overlap, _, _ = _sat_overlap(all_verts[i], all_normals[i], all_verts[j], all_normals[j])
                    penalty += overlap
            return s + 5.0 * penalty + 2.0
    
    x0 = np.concatenate([best_inner.flatten(), [best_outer_angle]])
    
    for restart in range(8):
        if time.time() > deadline - 60.0:
            break
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + np.random.randn(37) * 0.006
        
        try:
            result = minimize(objective, x_start, method='Nelder-Mead',
                             options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12})
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
    
    # Phase 3: Fine-tuning
    if time.time() < deadline - 80.0:
        np.random.seed(123)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(12000):
            if time.time() > deadline - 80.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.003
            new_inner[idx, 1] += np.random.randn() * 0.003
            new_inner[idx, 2] += np.random.randn() * 0.0015
            if np.random.rand() < 0.3:
                new_outer_angle += np.random.randn() * 0.002
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if not new_valid:
                new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=25, push_factor=0.4)
                new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            
            if new_valid:
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
                    if np.random.rand() < np.exp(-delta / max(0.002, 1e-10)):
                        current_inner = new_inner
                        current_s = new_s
                        current_outer_angle = new_outer_angle
    
    # Phase 4: Ultra-fine polish with smaller steps
    if time.time() < deadline - 30.0:
        np.random.seed(456)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(5000):
            if time.time() > deadline - 30.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.001
            new_inner[idx, 1] += np.random.randn() * 0.001
            new_inner[idx, 2] += np.random.randn() * 0.0005
            if np.random.rand() < 0.4:
                new_outer_angle += np.random.randn() * 0.0008
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if not new_valid:
                new_inner = _resolve_collisions(new_inner, new_outer_angle, max_iters=15, push_factor=0.3)
                new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
    
    # Phase 5: High-precision final polish with very small steps
    if time.time() < deadline - 15.0:
        np.random.seed(789)
        current_inner = best_inner.copy()
        current_s = best_s
        current_outer_angle = best_outer_angle
        
        for it in range(3000):
            if time.time() > deadline - 15.0:
                break
            new_inner = current_inner.copy()
            new_outer_angle = current_outer_angle
            idx = np.random.randint(0, 12)
            new_inner[idx, 0] += np.random.randn() * 0.0003
            new_inner[idx, 1] += np.random.randn() * 0.0003
            new_inner[idx, 2] += np.random.randn() * 0.00015
            if np.random.rand() < 0.5:
                new_outer_angle += np.random.randn() * 0.0002
            
            new_s, new_valid = _compute_s_and_overlap_fast(new_inner, new_outer_angle)
            if new_valid:
                if new_s < current_s:
                    current_inner = new_inner
                    current_s = new_s
                    current_outer_angle = new_outer_angle
                    if current_s < best_s:
                        best_inner = current_inner.copy()
                        best_s = current_s
                        best_outer_angle = current_outer_angle
    
    # Final validity check
    s_final, valid_final = _compute_s_and_overlap_fast(best_inner, best_outer_angle)
    if not valid_final:
        best_inner = seeds[0].copy()
        best_s, _ = _compute_s_and_overlap_fast(best_inner, 0)
        best_outer_angle = 0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
