import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin for 150s limit

# Precompute unit hexagon vertices (pointy-top orientation, side=1)
# Vertices at angles 90, 150, 210, 270, 330, 30
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i + 90)), np.sin(np.radians(60 * i + 90))] for i in range(6)
])

# Outward normals for the 6 edges of a regular hexagon (pointy-top)
# Edge midpoints at 0, 60, 120, 180, 240, 300
OUTER_NORMALS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

# Precompute edge vectors for SAT
EDGE_VECS = np.roll(UNIT_VERTS, -1, axis=0) - UNIT_VERTS
EDGE_NORMS = np.linalg.norm(EDGE_VECS, axis=1, keepdims=True)
EDGE_NORMALS_UNIT = EDGE_VECS / (EDGE_NORMS + 1e-15)
# Rotate edge normals by 90 deg to get outward-facing normals for each edge
def _rot90(v):
    return np.array([-v[1], v[0]])
EDGE_OUTWARD_NORMALS = np.array([_rot90(EDGE_NORMALS_UNIT[i]) for i in range(6)])

# Check orientation: for pointy-top, edge 0 goes from v0 to v1
# v0 = (0,1), v1 = (-sqrt(3)/2, 1/2)
# edge = (-sqrt(3)/2, -1/2), rot90 = (1/2, -sqrt(3)/2) -> points right-down, which is outward for edge between v0 and v1? 
# Actually for a convex polygon, the outward normal of edge from v_i to v_{i+1} is rot90(edge) if vertices are CCW.
# Let's verify: v0=(0,1), v1=(-0.866, 0.5). Edge = v1-v0 = (-0.866, -0.5). rot90 = (0.5, -0.866). 
# The midpoint of edge 0 is (-0.433, 0.75). The vector from center (0,0) to midpoint is (-0.433, 0.75), normalized ~ (-0.5, 0.866).
# rot90(edge) = (0.5, -0.866) which is opposite. So we need -rot90(edge) for CCW vertices.
# Let's just compute outward normals as: (midpoint - center) normalized, but for SAT we need consistent normals.
# For SAT, we just need the edge normals (perpendicular to edges). The sign doesn't matter for separation magnitude.
SAT_AXES = EDGE_NORMALS_UNIT  # 6 axes, each perpendicular to an edge

def get_hex_vertices(params):
    """
    params: (N, 3) array of [x, y, angle_deg]
    Returns: (N, 6, 2) array of vertices
    """
    n = len(params)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        x, y, a_deg = params[i]
        a_rad = np.radians(a_deg)
        c = np.cos(a_rad)
        s = np.sin(a_rad)
        R = np.array([[c, -s], [s, c]])
        verts[i] = UNIT_VERTS @ R.T + np.array([x, y])
    return verts

def compute_containment_side(params, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all hexagons.
    Uses smooth max for differentiability.
    """
    verts = get_hex_vertices(params)
    oa_rad = np.radians(outer_angle)
    c_o = np.cos(-oa_rad)
    s_o = np.sin(-oa_rad)
    R_o = np.array([[c_o, -s_o], [s_o, c_o]])
    
    max_extent = 0.0
    for i in range(len(params)):
        v = verts[i]
        v_rel = v - outer_center
        v_rot = v_rel @ R_o.T
        # Project onto 6 outward normals
        projs = v_rot @ OUTER_NORMALS.T  # (6, 6)
        # For each normal direction, the max projection magnitude
        abs_projs = np.abs(projs)  # (6, 6)
        max_per_normal = np.max(abs_projs, axis=1)  # (6,)
        max_extent = max(max_extent, np.max(max_per_normal))
    
    # Apothem = side * sqrt(3)/2 => side = apothem / (sqrt(3)/2)
    return max_extent / (np.sqrt(3) / 2)

def smooth_max(values, beta=50.0):
    """Differentiable approximation of max."""
    return np.log(np.sum(np.exp(beta * values))) / beta

def objective(params):
    """
    Objective: outer_side + overlap_penalty
    params: (N*3,) flat array
    """
    centers = params.reshape(N, 3)
    verts = get_hex_vertices(centers)
    n = verts.shape[0]
    
    # 1. Containment side length
    req_side = compute_containment_side(centers)
    
    # 2. Overlap penalty using SAT
    min_sep = float('inf')
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            # SAT: project both onto 12 axes (6 from each)
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            norms = np.linalg.norm(axes, axis=1, keepdims=True)
            axes = axes / (norms + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            min_sep = min(min_sep, np.max(sep))
    
    overlap_penalty = 0.0
    if min_sep < 0:
        overlap_penalty = -min_sep * 500.0
    
    return req_side + overlap_penalty

def gradient_objective(params):
    """
    For gradient-based optimization, we need a differentiable objective.
    We'll use a smooth approximation of the overlap penalty.
    """
    centers = params.reshape(N, 3)
    verts = get_hex_vertices(centers)
    n = verts.shape[0]
    
    req_side = compute_containment_side(centers)
    
    # Smooth overlap penalty: sum of squared negative separations
    overlap_penalty = 0.0
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
            
            max_sep = np.max(sep)
            if max_sep < 0:
                overlap_penalty += max_sep * max_sep
    
    return req_side + overlap_penalty * 100.0

def numerical_gradient(func, x, eps=1e-5):
    """Compute numerical gradient."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad

def generate_structural_configs():
    """
    Generate diverse initial configurations based on geometric insights.
    """
    configs = []
    
    # 1. Hexagonal lattice cluster
    a1 = np.array([np.sqrt(3), 0.0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    
    # Pick 12 positions from a hex grid centered at origin
    positions = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            pos = i * a1 + j * a2
            positions.append(pos)
    
    # Sort by distance from center and take 12 closest
    positions.sort(key=lambda p: np.linalg.norm(p))
    selected = positions[:12]
    
    com = np.mean(selected, axis=0)
    config1 = []
    for p in selected:
        config1.append([p[0] - com[0], p[1] - com[1], 0.0])
    configs.append(np.array(config1).flatten())
    
    # 2. Circular arrangement with alternating rotations
    config2 = []
    for i in range(12):
        theta = 2 * np.pi * i / 12
        r = 3.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        angle = 30.0 if i % 2 == 0 else 0.0
        config2.append([x, y, angle])
    configs.append(np.array(config2).flatten())
    
    # 3. Two rings: 6 inner, 6 outer
    config3 = []
    # Inner ring
    for i in range(6):
        theta = 2 * np.pi * i / 6 + np.pi / 6
        r = 1.8
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        config3.append([x, y, 0.0])
    # Outer ring
    for i in range(6):
        theta = 2 * np.pi * i / 6
        r = 4.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        config3.append([x, y, 30.0])
    configs.append(np.array(config3).flatten())
    
    # 4. Random with fixed seed
    np.random.seed(42)
    config4 = []
    for i in range(12):
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        angle = np.random.uniform(0, 60)
        config4.append([x, y, angle])
    com = np.mean([c[:2] for c in config4], axis=0)
    config4 = [[c[0] - com[0], c[1] - com[1], c[2]] for c in config4]
    configs.append(np.array(config4).flatten())
    
    # 5. Grid-like arrangement
    config5 = []
    # 3x4 grid
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * 1.9
            y = (row - 1) * 1.7
            angle = 0.0
            config5.append([x, y, angle])
    configs.append(np.array(config5).flatten())
    
    return configs

def l_bfgs_b_optimize(x0, time_limit=10.0):
    """
    Run L-BFGS-B optimization with numerical gradient.
    """
    start_time = time.time()
    
    # Use a wrapper that checks time
    def time_checking_objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return gradient_objective(x)
    
    try:
        result = minimize(
            time_checking_objective,
            x0,
            method='L-BFGS-B',
            jac=None,  # Use numerical gradient
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5, 'maxls': 50}
        )
        return result.x
    except TimeoutError:
        return x0
    except Exception:
        return x0

def check_validity(centers, outer_side=None, outer_center=(0.0, 0.0), outer_angle=0.0, tol=1e-6):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    """
    if outer_side is None:
        outer_side = compute_containment_side(centers)
        
    verts = get_hex_vertices(np.array(centers, dtype=float))
    n = verts.shape[0]
    
    # 1. Check pairwise separation (SAT)
    for i in range(n):
        v1 = verts[i]
        for j in range(i + 1, n):
            v2 = verts[j]
            edges1 = np.roll(v1, -1, axis=0) - v1
            edges2 = np.roll(v2, -1, axis=0) - v2
            axes = np.vstack([edges1, edges2])
            axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-15)
            
            proj1 = v1 @ axes.T
            proj2 = v2 @ axes.T
            
            sep = np.minimum(proj1.min(axis=0) - proj2.max(axis=0),
                            proj2.min(axis=0) - proj1.max(axis=0))
            
            if np.max(sep) < -tol:
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

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Generate structural initial configs
    configs = generate_structural_configs()
    
    # Add more random configs
    np.random.seed(123)
    for _ in range(8):
        config = []
        for i in range(12):
            x = np.random.uniform(-6, 6)
            y = np.random.uniform(-6, 6)
            angle = np.random.uniform(0, 60)
            config.append([x, y, angle])
        com = np.mean([c[:2] for c in config], axis=0)
        config = [[c[0] - com[0], c[1] - com[1], c[2]] for c in config]
        configs.append(np.array(config).flatten())
    
    # Optimize each config with L-BFGS-B
    for idx, x0 in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
            
        remaining = DEADLINE - elapsed - 5
        if remaining < 3.0:
            break
        
        # Allocate time for this optimization
        opt_time = min(remaining * 0.7, 15.0)
        
        try:
            optimized = l_bfgs_b_optimize(x0, time_limit=opt_time)
        except Exception:
            optimized = x0.copy()
        
        # Validate and update best
        centers = optimized.reshape(N, 3)
        req_side = compute_containment_side(centers)
        
        if check_validity(centers, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (centers.tolist(), (0.0, 0.0), req_side, 0.0)
        
        # Also check the initial config
        init_centers = x0.reshape(N, 3)
        init_side = compute_containment_side(init_centers)
        if check_validity(init_centers, outer_side=init_side):
            if init_side < best_score:
                best_score = init_side
                best_result = (init_centers.tolist(), (0.0, 0.0), init_side, 0.0)
        
        # If we have a good result, do a final refinement
        if best_result is not None and best_score < 6.0:
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 10:
                # Refine the best result
                best_params = np.array(best_result[0]).flatten()
                refined = l_bfgs_b_optimize(best_params, time_limit=min(10.0, DEADLINE - elapsed - 5))
                refined_centers = refined.reshape(N, 3)
                refined_side = compute_containment_side(refined_centers)
                if check_validity(refined_centers, outer_side=refined_side):
                    if refined_side < best_score:
                        best_score = refined_side
                        best_result = (refined_centers.tolist(), (0.0, 0.0), refined_side, 0.0)
    
    if best_result is not None:
        return best_result
    
    # Fallback: Simple circular arrangement
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = compute_containment_side(inner_default)
    return inner_default, (0.0, 0.0), required_side, 0.0

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
