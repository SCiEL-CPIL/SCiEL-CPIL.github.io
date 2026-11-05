import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + np.cos(theta)
        y = cy + np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _get_outer_normals(angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for i in range(6):
        theta = angle_rad + np.pi/6 + i * np.pi / 3
        normals.append((np.cos(theta), np.sin(theta)))
    return np.array(normals)

def _min_enclosing_hex_side(inner_verts, outer_angle_deg):
    # inner_verts: list of arrays of vertices (N, 6, 2)
    # outer_angle_deg: rotation of outer hex
    normals = _get_outer_normals(outer_angle_deg)
    # Project all vertices onto all normals
    # We need max |dot(v, n)|
    max_proj = 0.0
    for verts in inner_verts:
        # verts shape (6, 2)
        # dots shape (6, 6)
        dots = np.abs(verts @ normals.T)
        max_proj = max(max_proj, np.max(dots))
    
    # S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    return 2 * max_proj / np.sqrt(3)

def _overlap_penalty(centers, min_dist):
    penalty = 0.0
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d)**2
    return penalty

def _compute_S_and_penalty(centers, angles, outer_angle, beta=10.0):
    """
    Compute smooth approximation of the enclosing hex side length and overlap penalty.
    Uses soft-max for the max projection and smooth min for distances.
    """
    n = len(centers)
    normals = _get_outer_normals(outer_angle)
    
    # Compute all vertex positions
    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i] = _hex_vertices(centers[i][0], centers[i][1], angles[i])
    
    # Project all vertices onto all normals: shape (n, 6, 6)
    dots = np.abs(np.einsum('nij,nj->ni', all_verts, normals))  # (n, 6)
    
    # Soft max over all projections
    # max_proj = soft_max(dots.flatten())
    flat_dots = dots.flatten()
    max_val = np.max(flat_dots)
    if max_val > 1e-10:
        exps = np.exp(beta * (flat_dots - max_val))
        soft_max = max_val + np.log(np.sum(exps)) / beta
    else:
        soft_max = 0.0
    
    S = 2 * soft_max / np.sqrt(3)
    
    # Overlap penalty using smooth min
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < np.sqrt(3):
                # Smooth penalty
                pen += (np.sqrt(3) - d)**2
    
    return S, pen

def _coordinate_descent_polish(centers, angles, outer_angle, max_iter=200, tol=1e-8):
    """
    Coordinate descent polish: optimize each center coordinate and outer angle.
    """
    n = len(centers)
    min_dist = np.sqrt(3)
    
    def compute_S(c, o_ang):
        verts_list = []
        for i in range(n):
            verts = _hex_vertices(c[i][0], c[i][1], angles[i])
            verts_list.append(verts)
        return _min_enclosing_hex_side(verts_list, o_ang)
    
    def compute_penalty(c):
        pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < min_dist:
                    pen += (min_dist - d)**2
        return pen
    
    best_S = compute_S(centers, outer_angle)
    best_pen = compute_penalty(centers)
    best_centers = centers.copy()
    best_o_ang = outer_angle
    
    for it in range(max_iter):
        improved = False
        
        # Optimize outer angle
        for _ in range(10):
            for delta in [0.1, -0.1, 0.01, -0.01]:
                new_ang = best_o_ang + delta
                S = compute_S(best_centers, new_ang)
                if S < best_S - tol:
                    best_S = S
                    best_o_ang = new_ang
                    improved = True
        
        # Optimize each center
        for i in range(n):
            for coord in range(2):
                for _ in range(10):
                    for delta in [0.01, -0.01, 0.001, -0.001]:
                        new_c = best_centers.copy()
                        new_c[i, coord] += delta
                        pen = compute_penalty(new_c)
                        if pen > 1e-6:
                            continue
                        S = compute_S(new_c, best_o_ang)
                        if S < best_S - tol:
                            best_S = S
                            best_centers = new_c
                            best_pen = pen
                            improved = True
        
        if not improved:
            break
    
    return best_centers, best_o_ang, best_S

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Initial configuration: 11 hexagons in a hexagonal pattern
    eps = 1e-9
    centers_list = []
    centers_list.append([0.0, 0.0])
    for k in range(6):
        angle = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers_list.append([r * np.cos(angle), r * np.sin(angle)])
        
    centers = np.array(centers_list)
    angles = np.array([30.0] * 11)
    outer_angle = 30.0
    
    # Compute initial S
    verts_list = []
    for i in range(11):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles[i])
        verts_list.append(verts)
    initial_S = _min_enclosing_hex_side(verts_list, outer_angle)
    
    best_S = initial_S
    best_centers = centers.copy()
    best_o_ang = outer_angle
    
    # Try multiple optimization strategies
    strategies = [
        ('nelder_mead', {}),
        ('nelder_mead_tight', {'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}),
    ]
    
    for strat_name, opts in strategies:
        if time.time() > DEADLINE - 20:
            break
            
        x0 = np.concatenate([best_centers.flatten(), [best_o_ang]])
        
        def objective(vars):
            if time.time() > DEADLINE - 5:
                return 1e9
            c = vars[:22].reshape(11, 2)
            o_ang = vars[22]
            
            # Check overlap
            pen = _overlap_penalty(c, np.sqrt(3))
            if pen > 1e-6:
                return 1e6 + pen
            
            # Compute vertices
            verts_list = []
            for i in range(11):
                verts = _hex_vertices(c[i][0], c[i][1], angles[i])
                verts_list.append(verts)
                
            S = _min_enclosing_hex_side(verts_list, o_ang)
            return S
        
        try:
            default_opts = {'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7}
            if opts:
                default_opts.update(opts)
            
            res = minimize(objective, x0, method='Nelder-Mead', options=default_opts)
            if res.fun < best_S:
                best_vars = res.x
                best_centers = best_vars[:22].reshape(11, 2)
                best_o_ang = best_vars[22]
                best_S = res.fun
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < DEADLINE - 10:
        try:
            polished_centers, polished_o_ang, polished_S = _coordinate_descent_polish(
                best_centers, angles, best_o_ang, max_iter=100, tol=1e-9)
            if polished_S < best_S:
                best_centers = polished_centers
                best_o_ang = polished_o_ang
                best_S = polished_S
        except Exception:
            pass
    
    # Final verification and return
    verts_list = []
    for i in range(11):
        verts = _hex_vertices(best_centers[i][0], best_centers[i][1], angles[i])
        verts_list.append(verts)
    side = _min_enclosing_hex_side(verts_list, best_o_ang)
    
    inner = []
    for i in range(11):
        inner.append([best_centers[i][0], best_centers[i][1], angles[i]])
    
    return inner, (0.0, 0.0), side, best_o_ang
