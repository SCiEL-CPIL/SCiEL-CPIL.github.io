# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_verts(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _outer_verts(s, angle_deg):
    return _hex_verts(0.0, 0.0, s, angle_deg)

def _check_inside(verts, outer_verts):
    """Check if all points in verts are inside the hexagon given by outer_verts (CCW)."""
    for k in range(6):
        p1 = outer_verts[k]
        p2 = outer_verts[(k + 1) % 6]
        edge = p2 - p1
        # Cross product for all points
        cross = np.cross(np.tile(edge, (len(verts), 1)), verts - p1)
        if np.any(cross < -1e-6):
            return False
    return True

def _check_separation(verts_i, verts_j):
    """Check if two hexagons (given as 6x2 arrays) are separated (no overlap)."""
    # Collect normals
    normals = []
    for v in [verts_i, verts_j]:
        for k in range(6):
            p1 = v[k]
            p2 = v[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normals.append(normal / norm_n)
    
    for normal in normals:
        proj_i = np.dot(verts_i, normal)
        proj_j = np.dot(verts_j, normal)
        min_i, max_i = np.min(proj_i), np.max(proj_i)
        min_j, max_j = np.min(proj_j), np.max(proj_j)
        
        if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
            return True  # Separated
    
    return False  # Overlapping

def _compute_s(inner, angle_deg):
    """Compute the minimum side length s of the outer hexagon (centered at origin) that contains all inner hexagons."""
    # We want to find the smallest s such that all inner vertices are inside the outer hexagon of side s.
    # The outer hexagon is defined by 6 inequalities: n_k . v <= s * d_k, where d_k is the distance from center to edge (apothem).
    # For a regular hexagon of side s, the apothem is s * sqrt(3) / 2.
    # The normals n_k are unit vectors pointing outward from the edges.
    
    a_rad = np.radians(angle_deg)
    # Unit outer hexagon vertices
    unit_verts = _hex_verts(0.0, 0.0, 1.0, angle_deg)
    
    # Normals for the unit hexagon
    normals = []
    for k in range(6):
        p1 = unit_verts[k]
        p2 = unit_verts[(k + 1) % 6]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Ensure outward
        mid = (p1 + p2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
        normals.append(normal)
    normals = np.array(normals)
    
    # Gather all inner vertices
    all_verts = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_verts(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # For each normal, find the max projection of inner vertices
    # max_proj_k = max(all_verts @ normal_k)
    # The condition is max_proj_k <= s * apothem_unit
    # apothem_unit = sqrt(3) / 2
    # So s >= max_proj_k / apothem_unit = max_proj_k * 2 / sqrt(3)
    
    projs = all_verts @ normals.T  # shape (72, 6)
    max_projs = np.max(projs, axis=0)  # shape (6,)
    
    apothem = np.sqrt(3) / 2
    s_required = np.max(max_projs / apothem)
    
    return max(s_required, np.sqrt(12))

def _is_valid(inner, angle_deg, s):
    """Full validity check."""
    if not np.all(np.isfinite(inner)) or not np.isfinite(s) or not np.isfinite(angle_deg):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    
    outer_verts = _outer_verts(s, angle_deg)
    
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(verts)
        
        # Check inside outer
        if not _check_inside(verts, outer_verts):
            return False
    
    # Check separation
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
                
    return True

def _objective(params):
    """Objective function for optimization. Returns s + penalty."""
    inner = params[:36].reshape(12, 3)
    angle_deg = params[36]
    
    # Ensure angles are reasonable (normalize to [0, 60) for hexagons, though any angle works)
    # Actually, hexagons have 60 degree symmetry.
    
    s = _compute_s(inner, angle_deg)
    
    # Penalty for invalidity
    if not _is_valid(inner, angle_deg, s):
        # Calculate overlap penalty
        penalty = 0.0
        inner_verts_list = []
        for i in range(12):
            cx, cy, ang = inner[i]
            verts = _hex_verts(cx, cy, 1.0, ang)
            inner_verts_list.append(verts)
            
        outer_verts = _outer_verts(s, angle_deg)
        
        # Containment penalty
        for i in range(12):
            verts = inner_verts_list[i]
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                edge = p2 - p1
                cross = np.cross(np.tile(edge, (len(verts), 1)), verts - p1)
                penalty += np.sum(np.maximum(0, -cross - 1e-6))
                
        # Separation penalty
        for i in range(12):
            for j in range(i + 1, 12):
                v1 = inner_verts_list[i]
                v2 = inner_verts_list[j]
                # Simple distance penalty between centers as a proxy, or proper SAT violation
                # Let's use a simple distance check for speed in penalty
                d = np.linalg.norm(inner[i, :2] - inner[j, :2])
                min_d = 2 * np.sqrt(3) # Minimum distance between centers for non-overlapping unit hexagons (tangent)
                if d < min_d:
                    penalty += (min_d - d) ** 2
                    
        return s + penalty * 10.0
        
    return s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 620.0  # 10s safety margin
    
    best_inner = None
    best_s = float('inf')
    best_angle = 0.0
    
    # 1. Generate initial seeds
    seeds = []
    
    # Seed 1: Hexagonal lattice (3 rings around center)
    # Centers at distance 0, sqrt(3), 2*sqrt(3)
    # Ring 0: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3)
    # Ring 2: 5 hexagons? No, 12 total.
    # Let's use a compact cluster.
    # A known good packing for 12 hexagons is a 4-3-3-2 arrangement or similar.
    # Let's just use a grid-like start and let optimizer work.
    
    # Grid start
    L = 2 * np.sqrt(3) + 0.1
    points = []
    for r in range(3):
        for q in range(4):
            x = (q + r / 2.0) * L
            y = r * np.sqrt(3) / 2 * L
            points.append([x, y, 0.0])
    centers = np.array([p[:2] for p in points])
    center = np.mean(centers, axis=0)
    for p in points:
        p[0] -= center[0]
        p[1] -= center[1]
    seeds.append(np.array(points))
    
    # Seed 2: Random perturbation of grid
    np.random.seed(42)
    rand_seed = seeds[0].copy()
    rand_seed[:, :2] += np.random.randn(12, 2) * 0.5
    seeds.append(rand_seed)
    
    # Evaluate seeds
    for seed in seeds:
        for ang in [0.0, 30.0]:
            s = _compute_s(seed, ang)
            if _is_valid(seed, ang, s):
                if s < best_s:
                    best_s = s
                    best_inner = seed.copy()
                    best_angle = ang
    
    if best_inner is None:
        # Fallback if seeds failed
        best_inner = seeds[0]
        best_s = _compute_s(best_inner, 0.0)
        best_angle = 0.0
        
    # 2. Differential Evolution (Global Optimization)
    # Bounds for x, y: [-10, 10]
    # Bounds for angle: [0, 60]
    # Bounds for outer angle: [0, 60]
    
    bounds = [(-10, 10)] * 24 + [(0, 60)] * 12 + [(0, 60)]
    
    def de_objective(params):
        if time.time() > deadline - 30.0:
            return 1e10
        return _objective(params)
    
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-6,
            popsize=15,
            polish=False  # We will do our own polish
        )
        if de_result.fun < best_s:
            candidate = de_result.x
            cand_inner = candidate[:36].reshape(12, 3)
            cand_angle = candidate[36]
            cand_s = _compute_s(cand_inner, cand_angle)
            if _is_valid(cand_inner, cand_angle, cand_s):
                best_inner = cand_inner
                best_s = cand_s
                best_angle = cand_angle
    except Exception:
        pass
    
    # 3. Local Refinement (Nelder-Mead)
    # Multiple restarts from best found so far
    
    x0 = np.concatenate([best_inner.flatten(), [best_angle]])
    
    try:
        nm_result = minimize(
            _objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if nm_result.fun < best_s:
            candidate = nm_result.x
            cand_inner = candidate[:36].reshape(12, 3)
            cand_angle = candidate[36]
            cand_s = _compute_s(cand_inner, cand_angle)
            if _is_valid(cand_inner, cand_angle, cand_s):
                best_inner = cand_inner
                best_s = cand_s
                best_angle = cand_angle
    except Exception:
        pass
    
    # 4. Final Polish with tighter tolerances
    # Try a few more local optimizations with slight perturbations to escape local minima
    
    np.random.seed(43)
    for i in range(10):
        if time.time() > deadline - 10.0:
            break
            
        x0_pert = np.concatenate([best_inner.flatten(), [best_angle]])
        x0_pert += np.random.randn(37) * 0.01
        
        try:
            nm_result = minimize(
                _objective,
                x0_pert,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            if nm_result.fun < best_s:
                candidate = nm_result.x
                cand_inner = candidate[:36].reshape(12, 3)
                cand_angle = candidate[36]
                cand_s = _compute_s(cand_inner, cand_angle)
                if _is_valid(cand_inner, cand_angle, cand_s):
                    best_inner = cand_inner
                    best_s = cand_s
                    best_angle = cand_angle
        except Exception:
            pass

    # Final validity check
    if not _is_valid(best_inner, best_angle, best_s):
        # Should not happen, but fallback to seed
        best_inner = seeds[0]
        best_s = _compute_s(best_inner, 0.0)
        best_angle = 0.0

    outer_hex_data = np.array([0.0, 0.0, best_angle])
    return best_inner, outer_hex_data, float(best_s)

# EVOLVE-BLOCK-END
