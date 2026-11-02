import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Unit hexagon geometry
# Vertices at angles 0, 60, 120, 180, 240, 300
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

def generate_initial_configurations():
    """
    Generate diverse, compact initial configurations.
    """
    configs = []
    
    # Hexagonal lattice basis vectors
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
    configs.append(np.array(make_config(p1)))
    
    # Pattern 2: 2-4-4-2
    p2 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p2)))
    
    # Pattern 3: 1-3-4-3-1
    p3 = [
        (0, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3),
        (0, 4)
    ]
    configs.append(np.array(make_config(p3)))
    
    # Pattern 4: 4-4-4
    p4 = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2)
    ]
    configs.append(np.array(make_config(p4)))
    
    # Pattern 5: 2-3-4-3
    p5 = [
        (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-2, 2), (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p5)))
    
    # Pattern 6: 3-3-3-3
    p6 = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (-1, 2), (0, 2), (1, 2),
        (-1, 3), (0, 3), (1, 3)
    ]
    configs.append(np.array(make_config(p6)))
    
    # Add random perturbations
    for i in range(3):
        base = configs[i % len(configs)].copy()
        np.random.seed(42 + i)
        perturbed = base + np.random.uniform(-0.2, 0.2, base.shape)
        configs.append(perturbed)
        
    return configs

def voronoi_objective(params):
    """
    Objective function based on Voronoi tessellation and Minkowski sums.
    We model the problem as minimizing the maximum extent of the "inflated" Voronoi cells.
    For each seed (hexagon center), we compute the Voronoi cell (approximated by the
    intersection of half-planes defined by perpendicular bisectors to other seeds).
    Then we compute the support function of the hexagon in the direction of each cell vertex.
    The cost is the maximum over all cell vertices of (distance from origin to cell vertex + support).
    """
    centers = np.array(params).reshape(N, 3)
    coords = centers[:, :2]
    angles = centers[:, 2]
    
    # 1. Compute Voronoi cells (approximated)
    # For each point i, the Voronoi cell is the intersection of half-planes:
    # dot(x - (p_i + p_j)/2, p_j - p_i) <= 0 for all j != i
    # We approximate the cell by finding the vertices of this convex polygon.
    # Since we only need the max extent, we can sample directions or compute vertices.
    
    # To be fast and differentiable-ish (though we use numerical grad in L-BFGS-B),
    # let's compute the vertices of the Voronoi cell for each seed.
    
    total_cost = 0.0
    
    for i in range(N):
        # Collect half-planes for cell i
        half_planes = []
        for j in range(N):
            if i == j:
                continue
            # Vector from i to j
            v = coords[j] - coords[i]
            # Midpoint
            mid = (coords[i] + coords[j]) / 2.0
            # Normal pointing away from j (towards i)
            n = -v
            # Normalize
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            # Plane: dot(n, x) <= dot(n, mid)
            # => dot(n, x) - dot(n, mid) <= 0
            # We store (n, c) where c = dot(n, mid)
            c = np.dot(n, mid)
            half_planes.append((n, c))
        
        if not half_planes:
            continue
            
        # Find vertices of the convex polygon defined by these half-planes.
        # We can do this by intersecting pairs of lines and checking if the intersection
        # satisfies all other half-planes.
        
        vertices = []
        n_planes = len(half_planes)
        
        for k in range(n_planes):
            n1, c1 = half_planes[k]
            for l in range(k + 1, n_planes):
                n2, c2 = half_planes[l]
                
                # Solve:
                # n1 . x = c1
                # n2 . x = c2
                A = np.array([n1, n2])
                b = np.array([c1, c2])
                
                # Check if singular
                det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                if abs(det) < 1e-6:
                    continue
                    
                x_int = np.linalg.solve(A, b)
                
                # Check if this intersection is inside all other half-planes
                valid = True
                for m in range(n_planes):
                    if m == k or m == l:
                        continue
                    nm, cm = half_planes[m]
                    if np.dot(nm, x_int) - cm > 1e-6:
                        valid = False
                        break
                
                if valid:
                    # Check if this vertex is already in the list (avoid duplicates)
                    is_dup = False
                    for v in vertices:
                        if np.linalg.norm(v - x_int) < 1e-5:
                            is_dup = True
                            break
                    if not is_dup:
                        vertices.append(x_int)
        
        if not vertices:
            # If no vertices found (e.g. unbounded or error), fallback to center
            vertices = [coords[i]]
            
        # Now, for each vertex of the Voronoi cell, compute the "Minkowski sum" extent.
        # The extent in direction u (unit vector from origin to vertex) is:
        # |vertex| + support_function(hexagon, u)
        # support_function(hexagon, u) = max(dot(v, u)) for v in hexagon vertices
        
        angle_rad = np.radians(angles[i])
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        hex_verts = UNIT_VERTS @ R.T
        
        for v in vertices:
            dist = np.linalg.norm(v)
            if dist < 1e-6:
                u = np.array([1.0, 0.0])
            else:
                u = v / dist
                
            # Support function
            support = np.max(np.dot(hex_verts, u))
            
            extent = dist + support
            total_cost += extent ** 2 # Sum of squares to make it smooth
            
    return total_cost

def optimize_with_voronoi(initial_params, time_limit=20.0):
    """
    Use L-BFGS-B to minimize the Voronoi objective.
    """
    start_time = time.time()
    
    # Bounds for parameters
    # x, y: [-10, 10]
    # angle: [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
        
    # Objective wrapper
    def obj(p):
        return voronoi_objective(p)
        
    # Run optimization
    try:
        res = minimize(
            obj, 
            initial_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'maxfun': 1000, 'disp': False}
        )
        return res.x
    except Exception:
        return initial_params

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    configs = generate_initial_configurations()
    
    # Shuffle to ensure diversity
    np.random.shuffle(configs)
    
    for i, config in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        # Allocate time for optimization
        relax_time = min(20.0, DEADLINE - elapsed - 5)
        
        if relax_time < 2.0:
            continue
            
        try:
            optimized = optimize_with_voronoi(config, time_limit=relax_time)
        except Exception:
            optimized = config
            
        # Compute actual score and validate
        req_side = compute_required_side(optimized)
        
        if check_validity(optimized, outer_side=req_side):
            if req_side < best_score:
                best_score = req_side
                best_result = (optimized.tolist(), (0.0, 0.0), req_side, 0.0)
        else:
            # If invalid, check original
            orig_side = compute_required_side(config)
            if check_validity(config, outer_side=orig_side):
                if orig_side < best_score:
                    best_score = orig_side
                    best_result = (config.tolist(), (0.0, 0.0), orig_side, 0.0)

    if best_result is not None:
        return best_result
    
    # Fallback
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
