"""
Combined solution: Caging formulation via convex hull support functions.
Treats the cluster of 12 hexagons as a single convex polytope (its convex hull).
The outer hexagon containment is reduced to checking that the support function
of the hull in each of the 6 outer-normal directions is <= apothem.
This bypasses per-vertex containment and enables a smooth, differentiable objective
that can be optimized with gradient-based methods. Pairwise non-overlap is still
enforced via a smooth repulsion potential (differentiable), so the whole landscape
is smooth and amenable to L-BFGS-B.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
import itertools

N = 12
DEADLINE = 148.0  # Safety margin before 150s kill

# Unit hexagon vertices (flat-top, side length 1)
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i + 30)), np.sin(np.radians(60 * i + 30))] for i in range(6)
])


def get_vertices(centers):
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


def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    verts = get_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)
    
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ OUTER_NORMALS.T
    abs_projs = np.abs(projs)
    max_extent = np.max(abs_projs)
    
    # apothem = side * sqrt(3)/2
    return max_extent / (np.sqrt(3) / 2)


def check_validity(centers, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    verts = get_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # Check pairwise separation using SAT
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            axes = axes / (norms + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
                return False
    
    # Check containment in outer hexagon
    apothem = outer_side * np.sqrt(3) / 2
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    all_verts = verts.reshape(-1, 2)
    v_rel = all_verts - outer_center
    v_rot = v_rel @ R_o.T
    projs = v_rot @ OUTER_NORMALS.T
    abs_projs = np.abs(projs)
    
    if np.max(abs_projs) > apothem + tol:
        return False
    
    return True


def hull_support_function(centers):
    """
    Compute the support function of the convex hull of all vertices of the
    inner hexagons in the 6 outer-normal directions.
    Returns: (6,) array of support values (max projection in each direction).
    """
    verts = get_vertices(np.array(centers, dtype=float))
    all_verts = verts.reshape(-1, 2)
    
    # For each of the 6 normals, compute max projection
    projs = all_verts @ OUTER_NORMALS.T  # (72, 6)
    support = np.max(projs, axis=0)  # (6,)
    return support


def caging_objective(params, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Caging formulation objective:
    - Smooth repulsion to prevent overlap (differentiable)
    - Minimize max support function value (smooth via log-sum-exp)
    This is fully differentiable.
    """
    centers = np.array(params).reshape(N, 3)
    
    # 1. Smooth repulsion for non-overlap
    # Minimum center distance for non-overlapping unit hexagons is sqrt(3)
    min_dist = np.sqrt(3.0)
    repulsion = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i, :2] - centers[j, :2])
            if d < min_dist * 1.1:
                # Smooth quadratic penalty
                overlap = max(0.0, min_dist - d)
                repulsion += 5000.0 * overlap * overlap
    
    # 2. Caging term: minimize the maximum support function value
    # Use log-sum-exp for smooth max
    support = hull_support_function(centers)
    # Smooth max via log-sum-exp with temperature
    temperature = 0.1
    log_sum_exp = np.log(np.sum(np.exp(support / temperature))) * temperature
    caging_cost = log_sum_exp ** 2  # Minimize squared max extent
    
    total = repulsion + caging_cost
    return total


def caging_gradient(params, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Numerical gradient of the caging objective (finite differences).
    """
    eps = 1e-6
    n_params = len(params)
    grad = np.zeros(n_params)
    f0 = caging_objective(params, outer_center, outer_angle)
    
    for i in range(n_params):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = caging_objective(params_plus, outer_center, outer_angle)
        grad[i] = (f_plus - f0) / eps
        params_plus = params.copy()
    
    return grad


def generate_lattice_configurations():
    """
    Generate diverse, compact initial configurations using hexagonal lattice patterns.
    """
    configs = []
    
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3) / 2.0, 1.5])
    
    def make_config(coords, angle=0.0):
        centers = []
        for i, j in coords:
            pos = i * a1 + j * a2
            centers.append([pos[0], pos[1], angle])
        return centers

    # Pattern 1: 3-4-3-2
    p1 = [
        (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p1))
    
    # Pattern 2: 2-4-4-2
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p2))
    
    # Pattern 3: 1-3-4-3-1
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(make_config(p3))
    
    # Pattern 4: 4-4-4
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(make_config(p4))
    
    # Pattern 5: 2-3-4-3
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p5))
    
    # Pattern 6: 3-3-3-3
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(make_config(p6))
    
    # Pattern 7: Center + ring
    p7 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 2)
    ]
    configs.append(make_config(p7))
    
    # Pattern 8: 3-4-3-2 variant
    p8 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3)
    ]
    configs.append(make_config(p8))
    
    # Add perturbed versions
    for i in range(4):
        base = configs[i % len(configs)]
        np.random.seed(100 + i)
        perturbed = []
        for c in base:
            dx = np.random.uniform(-0.15, 0.15)
            dy = np.random.uniform(-0.15, 0.15)
            da = np.random.uniform(-10, 10)
            perturbed.append([c[0] + dx, c[1] + dy, c[2] + da])
        configs.append(perturbed)
        
    return configs


def optimize_with_caging(initial_centers, time_limit=15.0):
    """
    Optimize the configuration using the caging formulation with L-BFGS-B.
    """
    start_time = time.time()
    x0 = np.array(initial_centers).flatten()
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        return caging_objective(x)
    
    try:
        res = minimize(
            obj, x0, method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-6, 'disp': False,
                     'maxfun': 5000}
        )
        new_centers = res.x.reshape(N, 3)
    except Exception:
        new_centers = np.array(initial_centers)
    
    return new_centers


def optimize_global_rotation(centers, outer_center, num_steps=120):
    """
    Find the best global rotation of the inner arrangement to minimize outer side length.
    """
    best_side = float('inf')
    best_angle = 0.0
    
    angles = np.linspace(0, 60, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = required_outer_side(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def refine_rotation(centers, outer_center, center_angle, num_steps=50, range_deg=5.0):
    """Fine-tune the rotation angle around a given center angle."""
    best_side = float('inf')
    best_angle = center_angle
    
    angles = np.linspace(center_angle - range_deg, center_angle + range_deg, num_steps)
    
    for rot_deg in angles:
        rotated_centers = []
        for c in centers:
            rotated_centers.append([c[0], c[1], c[2] + rot_deg])
        
        side = required_outer_side(rotated_centers, outer_center, 0.0)
        
        if side < best_side:
            best_side = side
            best_angle = rot_deg
    
    return best_side, best_angle


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_lattice_configurations()
    
    # Shuffle for diversity
    np.random.shuffle(configs)
    
    outer_center = np.array([0.0, 0.0])
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
        
        # Phase 1: Caging optimization
        relax_time = min(15.0, DEADLINE - elapsed - 5)
        if relax_time < 3.0:
            continue
        
        try:
            relaxed = optimize_with_caging(config, time_limit=relax_time)
        except Exception:
            relaxed = np.array(config)
        
        # Phase 2: Optimize global rotation
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        rot_time = min(8.0, DEADLINE - elapsed - 2)
        if rot_time > 1.0:
            side_rot, angle_rot = optimize_global_rotation(relaxed, outer_center, num_steps=60)
            # Apply the best rotation
            rotated_relaxed = []
            for c in relaxed:
                rotated_relaxed.append([c[0], c[1], c[2] + angle_rot])
            
            # Fine-tune rotation
            side_refined, angle_refined = refine_rotation(rotated_relaxed, outer_center, angle_rot, num_steps=30, range_deg=3.0)
            rotated_relaxed = []
            for c in relaxed:
                rotated_relaxed.append([c[0], c[1], c[2] + angle_refined])
            
            # Check validity of rotated version
            req_side = required_outer_side(rotated_relaxed, outer_center, 0.0)
            if check_validity(rotated_relaxed, outer_center, req_side, 0.0):
                if req_side < best_score:
                    best_score = req_side
                    best_result = (rotated_relaxed.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            # Just use the unrotated version
            req_side = required_outer_side(relaxed, outer_center, 0.0)
            if check_validity(relaxed, outer_center, req_side, 0.0):
                if req_side < best_score:
                    best_score = req_side
                    best_result = (relaxed.tolist(), (0.0, 0.0), req_side, 0.0)
        
        # Also check the original config
        orig_side = required_outer_side(config, outer_center, 0.0)
        if check_validity(config, outer_center, orig_side, 0.0):
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
    
    required_side = required_outer_side(inner_default, outer_center, 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0


if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
