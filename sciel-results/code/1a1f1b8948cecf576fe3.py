"""Combined program: smooth surrogate optimization with robust polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

def hexagon_vertices(center, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    cx, cy = center
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = base @ rot.T * side + np.array([cx, cy])
    return verts

def smooth_min(values, beta=50.0):
    """Soft-min using log-sum-exp."""
    values = np.array(values)
    if len(values) == 0:
        return 0.0
    m = np.min(values)
    if m < -100:
        return m
    shifted = values - m
    exp_vals = np.exp(-beta * shifted)
    return m - np.log(np.sum(exp_vals)) / beta

def compute_clearances(centers, outer_center, outer_angle, outer_side):
    """Compute all clearance values for inner-inner pairs and inner-outer."""
    n = len(centers)
    pair_clearances = []
    outer_clearances = []
    
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    apothem = outer_side * np.sqrt(3) / 2
    
    # Precompute vertices for all inner hexagons
    inner_verts = []
    for c in centers:
        v = hexagon_vertices((c[0], c[1]), 1.0, c[2])
        inner_verts.append(v)
    
    # Precompute edge normals for each hexagon
    inner_normals = []
    for v in inner_verts:
        normals = []
        for k in range(6):
            edge = v[(k + 1) % 6] - v[k]
            elen = np.linalg.norm(edge)
            if elen > 1e-12:
                normals.append(np.array([-edge[1], edge[0]]) / elen)
        inner_normals.append(np.array(normals))
    
    # Pairwise separations using SAT with all 12 axes
    for i in range(n):
        for j in range(i + 1, n):
            v1 = inner_verts[i]
            v2 = inner_verts[j]
            axes = np.vstack([inner_normals[i], inner_normals[j]])
            
            min_sep = float('inf')
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if sep < min_sep:
                    min_sep = sep
            
            pair_clearances.append(min_sep)
    
    # Outer clearance: for each inner vertex, compute signed distance to outer boundary
    for i in range(n):
        v = inner_verts[i]
        for vert in v:
            rel = np.array([vert[0] - outer_center[0], vert[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            max_proj = 0.0
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                if d > max_proj:
                    max_proj = d
            
            clearance = apothem - max_proj
            outer_clearances.append(clearance)
    
    min_pair = min(pair_clearances) if pair_clearances else 0.0
    min_outer = min(outer_clearances) if outer_clearances else 0.0
    
    return min_pair, min_outer, pair_clearances, outer_clearances

def required_outer_side(centers, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons."""
    max_extent = 0.0
    oa_rad = np.radians(outer_angle)
    cos_a = np.cos(-oa_rad)
    sin_a = np.sin(-oa_rad)
    
    for (px, py, pa) in centers:
        inner_verts = hexagon_vertices((px, py), 1.0, pa)
        for v in inner_verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            rx = rel[0] * cos_a - rel[1] * sin_a
            ry = rel[0] * sin_a + rel[1] * cos_a
            
            for k in range(6):
                theta = np.radians(60 * k)
                nx = np.cos(theta)
                ny = np.sin(theta)
                d = abs(rx * nx + ry * ny)
                max_extent = max(max_extent, d)
    
    return max_extent / (np.sqrt(3) / 2)

def sat_min_separation(c1, c2):
    """Compute minimum separation between two hexagons using SAT."""
    v1 = hexagon_vertices((c1[0], c1[1]), 1.0, c1[2])
    v2 = hexagon_vertices((c2[0], c2[1]), 1.0, c2[2])
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            n = np.array([-edge[1], edge[0]]) / edge_len
            axes.append(n)
    
    min_sep = float('inf')
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        sep = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        if sep < min_sep:
            min_sep = sep
    
    return min_sep

def check_validity(centers, outer_center, outer_side, outer_angle):
    """Check if packing is valid: all inner hexagons inside outer, no overlaps."""
    actual_side = required_outer_side(centers, outer_center, outer_angle)
    if actual_side > outer_side + 1e-9:
        return False
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            min_pair, _, _, _ = compute_clearances(
                [centers[i], centers[j]], (0,0), 0, 100
            )
            if min_pair < -1e-9:
                return False
    return True

def initial_config_hexagonal():
    """Create a hexagonal lattice initial configuration."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    
    positions = [
        (0, 0),
        (dx, 0), (-dx, 0),
        (dx/2, dy*0.866), (-dx/2, dy*0.866),
        (dx/2, -dy*0.866), (-dx/2, -dy*0.866),
        (3*dx/2, dy*0.866), (-3*dx/2, dy*0.866),
        (3*dx/2, -dy*0.866), (-3*dx/2, -dy*0.866),
        (0, dy*1.732), (0, -dy*1.732)
    ]
    
    for (x, y) in positions:
        centers.append([x, y, 0.0])
    
    return centers

def initial_config_circular():
    """Create a circular arrangement."""
    centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        r = 2.5
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a = np.degrees(theta)
        centers.append([x, y, a])
    return centers

def initial_config_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    centers = []
    for i in range(N):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        a = np.random.uniform(0, 60)
        centers.append([x, y, a])
    return centers

def initial_config_staggered():
    """Staggered grid arrangement."""
    centers = []
    dx = 2.0 * np.sqrt(3)
    dy = 3.0
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            x -= 1.5 * dx
            y -= dy
            centers.append([x, y, 0.0])
    return centers

def objective_smooth(params, outer_center, outer_angle, outer_side, beta=30.0):
    """Smooth objective: penalize negative clearances using soft-min."""
    n = N
    centers = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        centers.append([x, y, a])
    
    min_pair, min_outer, pair_cl, outer_cl = compute_clearances(
        centers, outer_center, outer_angle, outer_side
    )
    
    all_cl = pair_cl + outer_cl
    if len(all_cl) == 0:
        return 0.0
    
    sm = smooth_min(all_cl, beta)
    return -sm

def gradient_objective(params, outer_center, outer_angle, outer_side, beta=30.0):
    """Compute gradient of the smooth objective using finite differences."""
    eps = 1e-6
    grad = np.zeros_like(params)
    f0 = objective_smooth(params, outer_center, outer_angle, outer_side, beta)
    
    for i in range(len(params)):
        params_p = params.copy()
        params_p[i] += eps
        fp = objective_smooth(params_p, outer_center, outer_angle, outer_side, beta)
        grad[i] = (fp - f0) / eps
    
    return grad

def lbfghs_polish(centers, outer_center, outer_angle, outer_side, time_limit=20.0):
    """Polish a configuration using L-BFGS-B with the smooth objective."""
    start_time = time.time()
    
    params = np.zeros(3 * N)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        c = []
        for i in range(N):
            c.append([p[3*i], p[3*i+1], p[3*i+2]])
        min_pair, min_outer, pair_cl, outer_cl = compute_clearances(
            c, outer_center, outer_angle, outer_side
        )
        all_cl = pair_cl + outer_cl
        if len(all_cl) == 0:
            return 0.0
        sm = smooth_min(all_cl, beta=30.0)
        return -sm
    
    def grad(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        eps = 1e-5
        g = np.zeros_like(p)
        f0 = obj(p)
        for i in range(len(p)):
            p_p = p.copy()
            p_p[i] += eps
            fp = obj(p_p)
            g[i] = (fp - f0) / eps
        return g
    
    result = minimize(
        obj, params, jac=grad, method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    
    return new_centers

def high_precision_polish(centers, outer_center, outer_angle, time_limit=10.0):
    """Final high-precision polish: minimize outer side length directly."""
    start_time = time.time()
    
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    params = np.zeros(3 * N + 1)
    for i in range(N):
        params[3*i] = centers[i][0]
        params[3*i+1] = centers[i][1]
        params[3*i+2] = centers[i][2]
    params[3*N] = req_side
    
    def obj(p):
        if time.time() - start_time > time_limit:
            return 1e6
        c = []
        for i in range(N):
            c.append([p[3*i], p[3*i+1], p[3*i+2]])
        side = p[3*N]
        if side < 1.0:
            return 1e6
        
        min_pair, min_outer, pair_cl, outer_cl = compute_clearances(
            c, outer_center, outer_angle, side
        )
        all_cl = pair_cl + outer_cl
        if len(all_cl) == 0:
            return side
        
        sm = smooth_min(all_cl, beta=50.0)
        penalty = max(0, -sm) ** 2 * 100
        return side + penalty
    
    def grad(p):
        if time.time() - start_time > time_limit:
            return np.zeros_like(p)
        eps = 1e-6
        g = np.zeros_like(p)
        f0 = obj(p)
        for i in range(len(p)):
            p_p = p.copy()
            p_p[i] += eps
            fp = obj(p_p)
            g[i] = (fp - f0) / eps
        return g
    
    result = minimize(
        obj, params, jac=grad, method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    new_centers = []
    for i in range(N):
        new_centers.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
    new_side = result.x[3*N]
    
    return new_centers, new_side

def simulated_annealing_smooth(init_centers, time_limit=30.0):
    """Fast simulated annealing using smooth objective."""
    start_time = time.time()
    n = N
    
    centers = [list(c) for c in init_centers]
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    req_side = required_outer_side(centers, outer_center, outer_angle)
    
    def fast_objective(c, oc, oa):
        rs = required_outer_side(c, oc, oa)
        min_sep = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                s = sat_min_separation(c[i], c[j])
                if s < min_sep:
                    min_sep = s
        
        all_cl = [min_sep]
        
        oa_rad = np.radians(oa)
        cos_a = np.cos(-oa_rad)
        sin_a = np.sin(-oa_rad)
        apothem = rs * np.sqrt(3) / 2
        
        min_outer_cl = float('inf')
        for ci in c:
            verts = hexagon_vertices((ci[0], ci[1]), 1.0, ci[2])
            for v in verts:
                rel = np.array([v[0] - oc[0], v[1] - oc[1]])
                rx = rel[0] * cos_a - rel[1] * sin_a
                ry = rel[0] * sin_a + rel[1] * cos_a
                for k in range(6):
                    theta = np.radians(60 * k)
                    d = abs(rx * np.cos(theta) + ry * np.sin(theta))
                    cl = apothem - d
                    if cl < min_outer_cl:
                        min_outer_cl = cl
        
        all_cl.append(min_outer_cl)
        
        beta = 20.0
        sm = smooth_min(all_cl, beta)
        return rs + 50.0 * max(0, -sm)
    
    current_obj = fast_objective(centers, outer_center, outer_angle)
    best_centers = [list(c) for c in centers]
    best_obj = current_obj
    
    T = 1.0
    T_end = 0.01
    cooling = 0.9999
    
    while time.time() - start_time < time_limit:
        new_centers = [list(c) for c in centers]
        i = np.random.randint(n)
        step = T * 0.3
        new_centers[i][0] += np.random.normal(0, step)
        new_centers[i][1] += np.random.normal(0, step)
        new_centers[i][2] = (new_centers[i][2] + np.random.normal(0, step * 20)) % 60
        
        new_obj = fast_objective(new_centers, outer_center, outer_angle)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            centers = new_centers
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_centers = [list(c) for c in centers]
        
        T = max(T_end, T * cooling)
    
    return best_centers

def construct():
    start_time = time.time()
    
    best_result = None
    best_score = float('inf')
    
    configs = [
        ("hexagonal", initial_config_hexagonal()),
        ("circular", initial_config_circular()),
        ("random1", initial_config_random(42)),
        ("random2", initial_config_random(123)),
        ("staggered", initial_config_staggered()),
    ]
    
    for name, config in configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        sa_time = min(20.0, DEADLINE - elapsed - 10)
        if sa_time < 5:
            sa_time = 5
        
        try:
            sa_centers = simulated_annealing_smooth(config, time_limit=sa_time)
        except Exception:
            sa_centers = [list(c) for c in config]
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 8:
            break
        
        req_side = required_outer_side(sa_centers, (0.0, 0.0), 0.0)
        polish_time = min(15.0, DEADLINE - elapsed - 3)
        
        try:
            polished = lbfghs_polish(sa_centers, (0.0, 0.0), 0.0, req_side, time_limit=polish_time)
        except Exception:
            polished = sa_centers
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 3:
            break
        
        final_side = required_outer_side(polished, (0.0, 0.0), 0.0)
        
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                s = sat_min_separation(polished[i], polished[j])
                if s < -1e-9:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and final_side < best_score:
            best_score = final_side
            best_result = (polished, (0.0, 0.0), final_side, 0.0)
    
    if best_result is not None and time.time() - start_time < DEADLINE - 12:
        try:
            polished_centers, polished_side = high_precision_polish(
                best_result[0], (0.0, 0.0), 0.0, time_limit=10.0
            )
            
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    s = sat_min_separation(polished_centers[i], polished_centers[j])
                    if s < -1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and polished_side < best_score:
                best_score = polished_side
                best_result = (polished_centers, (0.0, 0.0), polished_side, 0.0)
        except Exception:
            pass
    
    if best_result is not None:
        return best_result
    
    inner_default = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        inner_default.append([x, y, 0.0])
    
    required_side = required_outer_side(inner_default, (0.0, 0.0), 0.0)
    return inner_default, (0.0, 0.0), required_side, 0.0
