"""Vectorized hexagon packing with improved search and tighter optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def hex_vertices_all(inner):
    """Vectorized: return (N,6,2) array of all inner hexagon vertices."""
    n = len(inner)
    if n == 0:
        return np.empty((0, 6, 2))
    xs = np.array([c[0] for c in inner])
    ys = np.array([c[1] for c in inner])
    angs = np.radians(np.array([c[2] for c in inner]))
    thetas = angs[:, None] + np.arange(6)[None, :] * np.pi / 3.0
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    return np.stack([xs[:, None] + cos_t, ys[:, None] + sin_t], axis=-1)

def outer_normals(outer_angle_deg):
    """Return (6,2) array of outward face normals for outer hexagon."""
    base = math.radians(outer_angle_deg + 30)
    angles = base + np.arange(6) * np.pi / 3.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def compute_side_exact(inner, ocx, ocy, oangle):
    """Exact minimum outer side length."""
    if not inner:
        return 0.0
    normals = outer_normals(oangle)
    verts = hex_vertices_all(inner)  # (N,6,2)
    # Project each vertex onto each normal
    # verts[:, :, 0] * normals[:, 0] summed over axis=1 -> (N,6)
    proj_x = verts[:, :, 0] @ normals[:, 0]  # (N,6)
    proj_y = verts[:, :, 1] @ normals[:, 1]  # (N,6)
    proj = proj_x + proj_y  # (N,6)
    proj -= ocx * normals[:, 0] + ocy * normals[:, 1]  # subtract center
    max_proj = proj.max()
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_side_smooth(inner, ocx, ocy, oangle, beta=60.0):
    """Smooth log-sum-exp surrogate for outer side length."""
    if not inner:
        return 0.0
    normals = outer_normals(oangle)
    verts = hex_vertices_all(inner)
    proj_x = verts[:, :, 0] @ normals[:, 0]
    proj_y = verts[:, :, 1] @ normals[:, 1]
    proj = proj_x + proj_y
    proj -= ocx * normals[:, 0] + ocy * normals[:, 1]
    P = proj.ravel()
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (numpy arrays of (2,) points)."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            ln = np.linalg.norm(n)
            if ln > 1e-12:
                axes.append(n / ln)
    for ax in axes:
        proj1 = v1 @ ax
        proj2 = v2 @ ax
        if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = [hex_vertices(c[0], c[1], 1.0, math.radians(c[2])) for c in inner]
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def overlap_penalty_vec(centers, min_dist=SQRT3 - 0.002):
    """Vectorized quadratic overlap penalty."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    violations = np.maximum(0, min_dist - dists)
    return np.sum(violations ** 2)

def axial_to_cartesian(q, r, s=1.0):
    x = s * (1.5 * q)
    y = s * (SQRT3 * (r + 0.5 * q))
    return x, y

def build_lattice(axial_coords, s=1.0 + 1e-8):
    centers = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r, s)
        centers.append([x, y, 0.0])
    return centers

def rigid_optimize(inner, deadline):
    """Optimize outer hexagon center and angle for fixed inner positions."""
    ox = sum(c[0] for c in inner) / len(inner)
    oy = sum(c[1] for c in inner) / len(inner)
    verts = hex_vertices_all(inner)  # (N,6,2)
    
    def obj(p):
        return compute_side_smooth(inner, p[0], p[1], p[2], beta=50.0)
    
    def obj_exact(p):
        return compute_side_exact(inner, p[0], p[1], p[2])
    
    best_val = float('inf')
    best_p = None
    for ia in range(0, 60, 3):
        if time.time() > deadline - 10:
            break
        p0 = np.array([ox, oy, float(ia)])
        try:
            r1 = minimize(obj, p0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 600})
            r2 = minimize(obj_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 2000})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    if best_p is not None:
        return (float(best_p[0]), float(best_p[1])), float(best_p[2]), best_val
    return (ox, oy), 0.0, compute_side_exact(inner, (ox, oy), 0.0)

def joint_objective_smooth(params, beta=40.0):
    """Smooth joint objective: outer side + overlap penalty."""
    il = [[params[i*3], params[i*3+1], params[i*3+2]] for i in range(N)]
    centers = params[:N*3].reshape(N, 2)
    ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
    side = compute_side_smooth(il, ocx, ocy, oa, beta=beta)
    pen = overlap_penalty_vec(centers)
    return side + 800.0 * pen

def joint_objective_exact(params):
    """Exact joint objective."""
    il = [[params[i*3], params[i*3+1], params[i*3+2]] for i in range(N)]
    centers = params[:N*3].reshape(N, 2)
    ocx, ocy, oa = params[N*3], params[N*3+1], params[N*3+2]
    side = compute_side_exact(il, ocx, ocy, oa)
    pen = overlap_penalty_vec(centers)
    return side + 5000.0 * pen

def joint_optimize(init_inner, init_oc, init_oa, deadline):
    """Joint optimization via simulated annealing + Nelder-Mead polish."""
    p0 = np.zeros(N*3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i*3] = x
        p0[i*3+1] = y
        p0[i*3+2] = a
    p0[N*3] = init_oc[0]
    p0[N*3+1] = init_oc[1]
    p0[N*3+2] = init_oa
    
    current = p0.copy()
    current_val = joint_objective_smooth(current)
    best_params = current.copy()
    best_val = current_val
    
    T = 0.8
    T_min = 0.001
    cool = 0.99985
    max_iter = 12000
    np.random.seed(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 25:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.55:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx*3 + dim] += np.random.normal(0, 0.2)
        elif r < 0.75:
            idx = np.random.randint(0, N)
            nb[idx*3 + 2] += np.random.normal(0, 12.0)
        elif r < 0.85:
            dim = np.random.randint(0, 2)
            nb[N*3 + dim] += np.random.normal(0, 0.12)
        else:
            nb[N*3 + 2] += np.random.normal(0, 3.5)
        
        nb_val = joint_objective_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val:
                best_params = current.copy()
                best_val = current_val
        T *= cool
        if T < T_min:
            break
    
    # Nelder-Mead polish from best
    if time.time() < deadline - 15:
        try:
            res = minimize(joint_objective_exact, best_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000})
            best_params = res.x
            best_val = joint_objective_exact(best_params)
        except Exception:
            pass
    
    il = [[float(best_params[i*3]), float(best_params[i*3+1]), float(best_params[i*3+2])] for i in range(N)]
    oc = (float(best_params[N*3]), float(best_params[N*3+1]))
    oa = float(best_params[N*3+2])
    return il, oc, oa, best_params

def final_polish(params, deadline):
    """High-precision final polish."""
    def obj(p):
        il = [[p[i*3], p[i*3+1], p[i*3+2]] for i in range(N)]
        centers = p[:N*3].reshape(N, 2)
        ocx, ocy, oa = p[N*3], p[N*3+1], p[N*3+2]
        side = compute_side_exact(il, ocx, ocy, oa)
        pen = overlap_penalty_vec(centers)
        return side + 8000.0 * pen
    try:
        res = minimize(obj, params, method='Nelder-Mead',
                       options={'maxiter': 6000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 10000})
        return res.x
    except Exception:
        return params

def construct():
    t0 = time.time()
    deadline = t0 + 143.0

    # Candidate topologies
    axial1 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(1,1),(-2,1),(-1,-1)]
    axial2 = [(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,1)]
    axial3 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    axial4 = [(-2,-1),(-1,-1),(0,-1),(1,-1),(2,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1)]
    axial5 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(-2,1),(-2,-1)]
    axial6 = [(0,0),(1,0),(1,1),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(2,1),(-2,1)]
    axial7 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(2,0),(-2,0),(-2,1)]
    axial8 = [(0,0),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(2,-1),(2,0),(2,1)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8]
    
    best_side = float('inf')
    best_inner = None
    best_oc = None
    best_oa = 0.0
    best_params = None
    
    # Phase 1: Rigid optimization for each config
    for axial in configs:
        if time.time() > deadline - 55:
            break
        init_inner = build_lattice(axial)
        oc, oa, side_val = rigid_optimize(init_inner, deadline)
        if side_val < best_side:
            best_side = side_val
            best_inner = [list(c) for c in init_inner]
            best_oc = oc
            best_oa = oa
            p0 = np.zeros(N*3+3)
            for i, (x,y,a) in enumerate(init_inner):
                p0[i*3]=x; p0[i*3+1]=y; p0[i*3+2]=a
            p0[N*3]=oc[0]; p0[N*3+1]=oc[1]; p0[N*3+2]=oa
            best_params = p0.copy()
    
    if best_inner is None:
        best_inner = build_lattice(configs[0])
        best_oc = (0,0); best_oa = 0.0
        best_side = compute_side_exact(best_inner, best_oc, best_oa)
        best_params = np.zeros(N*3+3)
    
    # Phase 2: Joint optimization from top configs
    if time.time() < deadline - 40:
        # Try joint opt from best rigid config
        ji, joc, joa, jparams = joint_optimize(best_inner, best_oc, best_oa, deadline)
        if check_disjoint(ji, eps=1e-5):
            js = compute_side_exact(ji, joc, joa)
            if js < best_side:
                best_side = js; best_inner = ji; best_oc = joc; best_oa = joa; best_params = jparams
        else:
            ti = [[float(jparams[i*3]),float(jparams[i*3+1]),float(jparams[i*3+2])] for i in range(N)]
            toc = (float(jparams[N*3]), float(jparams[N*3+1]))
            toa = float(jparams[N*3+2])
            if check_disjoint(ti, eps=1e-4):
                ts = compute_side_exact(ti, toc, toa)
                if ts < best_side:
                    best_side = ts; best_inner = ti; best_oc = toc; best_oa = toa; best_params = jparams
    
    # Phase 3: Polish
    if time.time() < deadline - 8 and best_params is not None:
        polished = final_polish(best_params, deadline)
        pi = [[float(polished[i*3]),float(polished[i*3+1]),float(polished[i*3+2])] for i in range(N)]
        poc = (float(polished[N*3]), float(polished[N*3+1]))
        poa = float(polished[N*3+2])
        if check_disjoint(pi, eps=1e-5):
            ps = compute_side_exact(pi, poc, poa)
            if ps < best_side:
                best_side = ps; best_inner = pi; best_oc = poc; best_oa = poa
    
    # Fallback
    if not check_disjoint(best_inner, eps=1e-5):
        best_inner = build_lattice(configs[0])
        cx = sum(c[0] for c in best_inner)/len(best_inner)
        cy = sum(c[1] for c in best_inner)/len(best_inner)
        best_oc = (cx, cy); best_oa = 0.0
        best_side = compute_side_exact(best_inner, best_oc, best_oa)
    
    return best_inner, best_oc, best_side, best_oa

# EVOLVE-BLOCK-END
