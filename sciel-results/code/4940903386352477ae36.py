import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0

# EVOLVE-BLOCK-START

def hex_vertices_local(angle_rad=0.0):
    """Return 6 vertices of a unit hexagon centered at origin."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(thetas), np.sin(thetas)])

def get_vertices(centers, angles):
    """
    centers: (N, 2) array
    angles: (N,) array in radians
    Returns: (N, 6, 2) array of vertices
    """
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        base = hex_vertices_local(angles[i])
        verts[i] = base + centers[i]
    return verts

def compute_side_length(centers, angles, outer_center, outer_angle_rad):
    """
    Compute the minimum side length of the outer hexagon.
    """
    # Normals of the outer hexagon
    # Outer hexagon vertices are at angles outer_angle + i*pi/3
    # Normals point to midpoints of sides, so angles are outer_angle + pi/6 + i*pi/3
    normal_angles = outer_angle_rad + np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Get all vertices
    verts = get_vertices(centers, angles)
    
    # Relative to outer center
    rel_verts = verts - outer_center
    
    # Project onto normals
    # rel_verts shape (N, 6, 2), normals shape (6, 2)
    # Projections shape (N, 6, 6) -> max over inner vertices (axis 1)
    projs = np.einsum('nij,jk->nik', rel_verts, normals)
    max_projs = projs.max(axis=1) # Shape (N, 6)
    
    # We need the max over all hexagons for each normal
    global_max_projs = max_projs.max(axis=0) # Shape (6,)
    
    # Side length S = 2 / sqrt(3) * max(global_max_projs)
    # Note: The projection must be positive. If all are negative, the hexagon is on the other side?
    # The outer hexagon must contain all.
    # The condition is max_proj <= S * sqrt(3) / 2.
    # So S = 2 / sqrt(3) * max(max_proj, 0)
    max_val = np.max(global_max_projs)
    if max_val < 0:
        max_val = 0.0
    return 2.0 * max_val / SQRT3

def overlap_penalty(centers, min_dist=SQRT3 - 1e-4):
    """
    Quadratic penalty for center distances < min_dist.
    """
    pen = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def objective_smooth(params, beta=50.0):
    """
    params: [x1, y1, a1, ..., xN, yN, aN, ox, oy, oa]
    """
    N_loc = N
    centers = params[:N_loc * 2].reshape(N_loc, 2)
    angles = params[N_loc * 2:N_loc * 3]
    ox, oy, oa = params[N_loc * 3], params[N_loc * 3 + 1], params[N_loc * 3 + 2]
    
    # Compute side length (smooth version not strictly needed if we use exact max, 
    # but smooth helps gradient-free methods? No, Nelder-Mead handles non-smooth.)
    # Let's use exact side length for the objective, it's cheap enough.
    side = compute_side_length(centers, angles, np.array([ox, oy]), oa)
    
    # Penalty
    pen = overlap_penalty(centers)
    
    return side + 100.0 * pen

def objective_exact(params):
    """
    Exact objective with hard penalty.
    """
    N_loc = N
    centers = params[:N_loc * 2].reshape(N_loc, 2)
    angles = params[N_loc * 2:N_loc * 3]
    ox, oy, oa = params[N_loc * 3], params[N_loc * 3 + 1], params[N_loc * 3 + 2]
    
    side = compute_side_length(centers, angles, np.array([ox, oy]), oa)
    
    # Check for actual overlaps using SAT? 
    # For speed, we rely on the distance penalty in the smooth phase.
    # Here, if distance < sqrt(3), we add a huge penalty.
    pen = overlap_penalty(centers, min_dist=SQRT3 - 1e-5)
    
    return side + 1000.0 * pen

def check_disjoint_sat(inner, eps=1e-6):
    """
    inner: list of (x, y, angle_deg)
    """
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices_local(math.radians(a)) + np.array([x, y]))
    
    for i in range(n):
        for j in range(i + 1, n):
            # SAT check
            v1 = polys[i]
            v2 = polys[j]
            
            # Axes from v1
            axes = []
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                e = p2 - p1
                nrm = np.array([-e[1], e[0]])
                ln = np.linalg.norm(nrm)
                if ln > 1e-9:
                    axes.append(nrm / ln)
            
            # Axes from v2
            for k in range(6):
                p1 = v2[k]
                p2 = v2[(k + 1) % 6]
                e = p2 - p1
                nrm = np.array([-e[1], e[0]])
                ln = np.linalg.norm(nrm)
                if ln > 1e-9:
                    axes.append(nrm / ln)
            
            for ax in axes:
                proj1 = v1 @ ax
                proj2 = v2 @ ax
                if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                    break # Separated
            else:
                return False # No separating axis found -> overlap
    return True

def generate_initial_configs():
    """
    Generate a few diverse initial configurations.
    """
    configs = []
    
    # Config 1: Flower (1 center, 6 around, 4 more)
    # Center at 0,0
    c = [[0.0, 0.0, 0.0]]
    # 6 neighbors
    for i in range(6):
        a = math.radians(60 * i)
        c.append([SQRT3 * math.cos(a), SQRT3 * math.sin(a), 0.0])
    # 4 more: placed at distance 2*sqrt(3) in directions 30, 150, 270, 30?
    # Let's just put them in the next ring
    for i in [0, 1, 3, 4]: # Pick 4 from 12
        a = math.radians(60 * i + 30) # Offsets
        # Distance to center is 2 * sqrt(3) / sqrt(3) * ... 
        # Axial distance 2 is 2 * 1.5 * ... 
        # Let's just use a grid
        pass
    
    # Simpler: Use a grid
    # Axial coords
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (0, 2), (-2, 1), (-2, -1)
    ]
    s = 1.0 + 1e-5
    centers = []
    for q, r in axial:
        x = 1.5 * s * q
        y = SQRT3 * s * (r + q / 2.0)
        centers.append([x, y, 0.0])
    configs.append(centers)
    
    # Config 2: Perturbed grid
    np.random.seed(42)
    centers2 = [c.copy() for c in centers]
    for i in range(N):
        centers2[i][0] += np.random.normal(0, 0.1)
        centers2[i][1] += np.random.normal(0, 0.1)
        centers2[i][2] = np.random.uniform(-10, 10)
    configs.append(centers2)
    
    # Config 3: Compact cluster
    # Try to fit them in a circle
    centers3 = [[0.0, 0.0, 0.0]]
    for i in range(6):
        a = math.radians(60 * i)
        centers3.append([SQRT3 * math.cos(a), SQRT3 * math.sin(a), 0.0])
    # Add 4 more at radius 2*SQRT3?
    for i in range(4):
        a = math.radians(90 + 90 * i)
        centers3.append([2 * SQRT3 * math.cos(a), 2 * SQRT3 * math.sin(a), 0.0])
    configs.append(centers3)
    
    return configs

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    best_params = None
    best_side = float('inf')
    
    inits = generate_initial_configs()
    
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        # Convert to params array
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(init):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        # Outer center at centroid
        cx = sum(c[0] for c in init) / N
        cy = sum(c[1] for c in init) / N
        p0[N * 3] = cx
        p0[N * 3 + 1] = cy
        p0[N * 3 + 2] = 0.0
        
        # Optimize
        try:
            # Stage 1: Nelder-Mead on smooth objective
            res1 = minimize(objective_smooth, p0, method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            p1 = res1.x
            
            # Stage 2: Nelder-Mead on exact objective
            if time.time() < deadline - 5:
                res2 = minimize(objective_exact, p1, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
                p2 = res2.x
            else:
                p2 = p1
                
            # Validate
            inner = []
            for i in range(N):
                inner.append([float(p2[i * 3]), float(p2[i * 3 + 1]), float(p2[i * 3 + 2])])
            oc = (float(p2[N * 3]), float(p2[N * 3 + 1]))
            oa = float(p2[N * 3 + 2])
            
            if check_disjoint_sat(inner, eps=1e-5):
                side = compute_side_length(
                    np.array([[x, y] for x, y, _ in inner]), 
                    np.array([a for _, _, a in inner]), 
                    np.array(oc), 
                    math.radians(oa)
                )
                if side < best_side:
                    best_side = side
                    best_params = p2.copy()
                    
        except Exception as e:
            pass
    
    # Fallback
    if best_params is None:
        init = inits[0]
        best_params = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(init):
            best_params[i * 3] = x
            best_params[i * 3 + 1] = y
            best_params[i * 3 + 2] = a
        cx = sum(c[0] for c in init) / N
        cy = sum(c[1] for c in init) / N
        best_params[N * 3] = cx
        best_params[N * 3 + 1] = cy
        best_params[N * 3 + 2] = 0.0
        best_side = compute_side_length(
            np.array([[x, y] for x, y, _ in init]), 
            np.array([a for _, _, a in init]), 
            np.array([cx, cy]), 
            0.0
        )
        
    # Final result extraction
    inner = []
    for i in range(N):
        inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    oc = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    oa = float(best_params[N * 3 + 2])
    
    return inner, oc, best_side, oa

# EVOLVE-BLOCK-END
