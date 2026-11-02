import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 12
DEADLINE = 148.0  # Safety margin for 150s limit

# Unit hexagon geometry (flat-top)
# Vertices at angles 0, 60, 120, 180, 240, 330
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges (pointing to edge midpoints)
# Midpoints at 30, 90, 150, 210, 270, 330
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])

def get_hex_vertices(centers):
    """
    centers: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(centers)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = centers[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def sat_min_separation(v1, v2):
    """
    Compute minimum separation between two convex polygons using SAT.
    Returns negative value if overlapping.
    """
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    axes = np.vstack([edges1, edges2])
    axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
    
    proj1 = v1 @ axes.T
    proj2 = v2 @ axes.T
    
    sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                    proj2.min(axis=0) - proj1.max(axis=0))
    
    return np.max(sep)

def compute_required_side(centers, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute the minimum outer side length needed to contain all inner hexagons.
    """
    verts = get_hex_vertices(np.array(centers, dtype=float))
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(centers)):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        # Project onto outward normals
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # Apothem = side * sqrt(3)/2  =>  side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def check_validity(centers, outer_center=(0.0, 0.0), outer_side=None, outer_angle=0.0, tol=1e-7):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_required_side(centers, outer_center, outer_angle)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            sep = sat_min_separation(v1, v2)
            if sep < -tol:
                return False
    
    # 2. Check containment
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        
        if np.max(max_proj) > apothem + tol:
            return False
            
    return True

def objective(params, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Objective function for Nelder-Mead.
    Minimizes: max(0, -min_separation) + max(0, max_extent - apothem)^2 * 1000 + max_extent
    We want to minimize the side length, but strictly enforce non-overlap.
    """
    centers = np.array(params).reshape(N, 3)
    verts = get_hex_vertices(centers)
    n = verts.shape[0]
    
    # 1. Overlap Penalty
    min_sep = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            sep = sat_min_separation(verts[i], verts[j])
            if sep < min_sep:
                min_sep = sep
    
    overlap_penalty = 0.0
    if min_sep < 0:
        overlap_penalty = -min_sep * 100.0
    
    # 2. Containment
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(n):
        v = verts[i]
        v_rel = v - np.array(outer_center)
        v_rot = v_rel @ R_o.T
        projs = v_rot @ OUTER_NORMALS.T
        abs_projs = np.abs(projs)
        max_proj = np.max(abs_projs, axis=1)
        max_extent = max(max_extent, np.max(max_proj))
    
    # We want to minimize the side length (proportional to max_extent)
    # But we must ensure validity.
    # If we just minimize max_extent, Nelder-Mead might find a spot with overlaps.
    # The penalty term handles that.
    
    return overlap_penalty + max_extent

def relax_configuration(initial_centers, time_limit=5.0):
    """
    Perform local optimization using Nelder-Mead.
    """
    start_time = time.time()
    x0 = np.array(initial_centers).flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return objective(x)
    
    try:
        res = minimize(
            obj, x0, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = np.array(initial_centers)
        
    return new_centers

def generate_structural_configs():
    """
    Generate diverse initial configurations based on hexagonal lattice symmetries.
    We vary the lattice spacing slightly to find "tighter" packs that might
    allow for a smaller outer boundary due to corner effects.
    """
    configs = []
    
    # Hexagonal lattice basis vectors for side-1 hexagons
    # Distance between centers of adjacent hexagons in a close pack is sqrt(3).
    # We vary the scale factor 's' from 0.95 to 1.05.
    # If s < 1, they overlap initially, but optimization can push them apart
    # into "holes" or tighter arrangements if the lattice allows.
    # However, for a good start, we want valid or near-valid configs.
    
    # Pattern 1: 3-4-3-2 (Compact)
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 2: 2-4-4-2 (Rectangular-ish)
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    
    # Pattern 3: 1-3-4-3-1 (Diamond-ish)
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    
    # Pattern 4: 4-4-4 (Three rows)
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    
    # Pattern 5: 2-3-4-3 (Asymmetric)
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    # Pattern 6: 3-3-3-3 (Staggered rows)
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    
    patterns = [p1, p2, p3, p4, p5, p6]
    
    # Vary lattice spacing and rotation
    spacings = [0.98, 1.0, 1.02]
    rotations = [0.0, 30.0, 15.0]
    
    for p in patterns:
        for s in spacings:
            for rot in rotations:
                centers = []
                a1 = np.array([np.sqrt(3) * s, 0.0])
                a2 = np.array([np.sqrt(3) / 2.0 * s, 1.5 * s])
                
                for i, j in p:
                    pos = i * a1 + j * a2
                    # Center the configuration
                    centers.append([pos[0], pos[1], rot])
                
                # Center of mass correction
                com = np.mean([c[:2] for c in centers], axis=0)
                centered = []
                for c in centers:
                    centered.append([c[0] - com[0], c[1] - com[1], c[2]])
                configs.append(centered)
    
    # Add random perturbations to break symmetry for some
    for i in range(6):
        base = configs[i * 3 + 1] # Pick a middle one
        np.random.seed(100 + i)
        perturbed = []
        for c in base:
            dx = np.random.uniform(-0.2, 0.2)
            dy = np.random.uniform(-0.2, 0.2)
            da = np.random.uniform(-10, 10)
            perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
        configs.append(perturbed)
        
    return configs

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_structural_configs()
    
    # Shuffle to ensure diversity in time usage
    np.random.seed(42)
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        # Phase 1: Local Optimization
        # Give each config a reasonable time budget
        relax_time = min(4.0, DEADLINE - elapsed - 5)
        
        if relax_time < 1.0:
            continue
            
        try:
            relaxed = relax_configuration(config, time_limit=relax_time)
        except Exception:
            relaxed = np.array(config)
            
        # Phase 2: Compute actual score and validate
        req_side = compute_required_side(relaxed)
        
        if check_validity(relaxed, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (relaxed.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            # If invalid, try the original config if it's valid and better
            orig_side = compute_required_side(config)
            if check_validity(config, outer_side=orig_side):
                 if orig_side < best_score:
                     best_score = orig_side
                     best_result = (config, (0.0, 0.0), orig_side, 0.0)

    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = compute_required_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
