import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def _norm_scale(points):
    pts = points - np.mean(points, axis=0)
    norm = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if norm < 1e-15:
        return pts
    return pts / norm

def _soft_minmax(sq_dists, tau):
    """Return (soft_max, soft_min) via log-sum-exp (numerically stable)."""
    sq_dists = np.maximum(sq_dists, 1e-18)
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    a = tau * (sq_dists - m_max)
    a = np.clip(a, -50, 50)
    soft_max = m_max + np.log(np.sum(np.exp(a))) / tau
    b = -tau * (sq_dists - m_min)
    b = np.clip(b, -50, 50)
    soft_min = m_min - np.log(np.sum(np.exp(b))) / tau
    soft_min = max(soft_min, 1e-18)
    return soft_max, soft_min

def objective_log_ratio(x, tau=20.0):
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    soft_max, soft_min = _soft_minmax(sq_dists, tau)
    return np.log(soft_max) - np.log(soft_min)

def objective_epigraph(x, tau=20.0, mu=5.0):
    points = x[:N*D].reshape(N, D)
    t = x[N*D]
    sq_dists = get_pairwise_sq_dists(points)
    soft_max, soft_min = _soft_minmax(sq_dists, tau)
    viol = soft_max - t * soft_min
    penalty = np.maximum(0.0, viol) ** 2
    return t + mu * penalty

# ---------- Symmetry-based parameterization ----------
def _rot_matrix(angle, axis):
    """Rotation matrix from angle and axis."""
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    elif axis == 1:
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    else:
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _tetra_vertices(r):
    """Regular tetrahedron vertices at radius r."""
    return r * np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ]) / np.sqrt(3.0)

def _octa_vertices(r):
    """Regular octahedron vertices at radius r."""
    return r * np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ])

def _params_to_points(params):
    """
    params = [r1, r2, r3, r4, r5, 
              ax1, ay1, az1,  # rotation angles for tetra 1
              ax2, ay2, az2,  # rotation angles for tetra 2
              ax3, ay3, az3]  # rotation angles for octahedron
    Returns 14 points: 4 (tetra1) + 4 (tetra2) + 6 (octahedron)
    """
    r1, r2, r3, r4, r5 = params[0], params[1], params[2], params[3], params[4]
    rot1 = _rot_matrix(params[5], 0) @ _rot_matrix(params[6], 1) @ _rot_matrix(params[7], 2)
    rot2 = _rot_matrix(params[8], 0) @ _rot_matrix(params[9], 1) @ _rot_matrix(params[10], 2)
    rot3 = _rot_matrix(params[11], 0) @ _rot_matrix(params[12], 1) @ _rot_matrix(params[13], 2)
    
    t1 = _tetra_vertices(r1) @ rot1.T
    t2 = _tetra_vertices(r2) @ rot2.T
    o = _octa_vertices(r3) @ rot3.T
    
    pts = np.vstack([t1, t2, o])
    return _norm_scale(pts)

def _sym_obj(params, tau=50.0):
    pts = _params_to_points(params)
    sq_dists = get_pairwise_sq_dists(pts)
    soft_max, soft_min = _soft_minmax(sq_dists, tau)
    return np.log(soft_max) - np.log(soft_min)

# ---------- Standard initializations ----------
def cuboctahedron_init():
    octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    cubo = np.array([[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],[1,0,1],[1,0,-1]], float)
    pts = np.vstack([octa, cubo])
    extra = np.array([[0.3,0.3,0.3],[-0.3,-0.3,-0.3]], float)
    pts = np.vstack([pts, extra])
    return _norm_scale(pts)

def icosahedron_init():
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi],[0, 1, -phi],[0, -1, phi],[0, -1, -phi],
        [1, phi, 0],[1, -phi, 0],[-1, phi, 0],[-1, -phi, 0],
        [phi, 0, 1],[phi, 0, -1],[-phi, 0, 1],[-phi, 0, -1]
    ], float)
    extra = np.array([[0,0,2.2],[0,0,-2.2]], float)
    pts = np.vstack([ico, extra])
    return _norm_scale(pts)

def fibonacci_sphere(n_points):
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - y**2))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def grid_init():
    pts = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                pts.append([x-1, y-1, z-1])
    pts = np.array(pts, float)
    pts = pts[:14]
    return _norm_scale(pts)

def random_sphere(seed):
    r = np.random.default_rng(seed)
    pts = r.normal(size=(N, D))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-12)
    return pts

def random_ball(seed, scale=1.5):
    r = np.random.default_rng(seed)
    pts = r.normal(size=(N, D)) * scale
    return _norm_scale(pts)

def construct():
    start_time = time.time()
    DEADLINE = 148.0
    def remaining():
        return DEADLINE - (time.time() - start_time)

    best_score = np.inf
    best_points = None

    def consider(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, float)
        if pts.shape != (N, D) or not np.all(np.isfinite(pts)):
            return
        s = raw_score(pts)
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_points = pts.copy()

    # ===== Phase 1: Symmetry-based optimization (9D parameter space) =====
    # params: [r1, r2, r3, r4, r5, ax1, ay1, az1, ax2, ay2, az2, ax3, ay3, az3]
    # r4, r5 are unused but kept for compatibility; use r1, r2, r3 for radii
    bounds_sym = [
        (0.3, 2.0),  # r1: tetra1 radius
        (0.3, 2.0),  # r2: tetra2 radius  
        (0.3, 2.0),  # r3: octahedron radius
        (0.3, 2.0),  # r4 (unused)
        (0.3, 2.0),  # r5 (unused)
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),  # rot1
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),  # rot2
        (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi),  # rot3
    ]
    
    if remaining() > 30:
        try:
            # Differential evolution in 14D param space
            res_de = differential_evolution(
                _sym_obj, bounds_sym,
                seed=42, maxiter=80, popsize=15,
                tol=1e-8, mutation=(0.3, 1.2), recombination=0.9,
                polish=False, init='sobol'
            )
            if res_de.success or np.isfinite(res_de.fun):
                pts_sym = _params_to_points(res_de.x)
                consider(pts_sym)
                
                # Local polish on full 42D from symmetry result
                if remaining() > 20:
                    x_curr = pts_sym.flatten().copy()
                    for tau in [100.0, 500.0]:
                        if remaining() < 5.0:
                            break
                        obj = partial(objective_log_ratio, tau=tau)
                        res = minimize(
                            obj, x_curr, method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11}
                        )
                        if res.x is not None and np.isfinite(res.fun):
                            x_curr = res.x
                    if remaining() > 5.0:
                        obj_nm = partial(objective_log_ratio, tau=800.0)
                        res_nm = minimize(
                            obj_nm, x_curr, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                        )
                        if res_nm.x is not None and np.isfinite(res_nm.fun):
                            x_curr = res_nm.x
                    consider(x_curr.reshape(N, D))
        except Exception:
            pass

    # ===== Phase 2: Standard multi-start optimization =====
    inits = []
    inits.append(cuboctahedron_init())
    inits.append(icosahedron_init())
    inits.append(fibonacci_sphere(N))
    inits.append(grid_init())
    for s in range(6):
        inits.append(random_sphere(s))
    for s in range(6):
        inits.append(random_ball(100 + s))
    for base in inits[:4]:
        for s in range(2):
            r = np.random.default_rng(500 + s)
            inits.append(_norm_scale(base + r.normal(size=(N, D)) * 0.15))

    tau_schedule = [8.0, 30.0, 120.0, 400.0]
    maxiter_schedule = [150, 200, 250, 300]

    for i, init_x in enumerate(inits):
        if remaining() < 6.0:
            break
        x_curr = init_x.flatten().copy()
        try:
            for idx, tau in enumerate(tau_schedule):
                if remaining() < 4.0:
                    break
                obj = partial(objective_log_ratio, tau=tau)
                res = minimize(
                    obj, x_curr, method='L-BFGS-B',
                    options={'maxiter': maxiter_schedule[idx],
                             'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
                )
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            if remaining() > 5.0:
                x_epi = np.concatenate([x_curr, [1.2]])
                obj_epi = partial(objective_epigraph, tau=200.0, mu=10.0)
                res_epi = minimize(
                    obj_epi, x_epi, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                if res_epi.x is not None and np.isfinite(res_epi.fun):
                    x_curr = res_epi.x[:N*D]
            if remaining() > 3.0:
                obj_nm = partial(objective_log_ratio, tau=800.0)
                res_nm = minimize(
                    obj_nm, x_curr, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res_nm.x is not None and np.isfinite(res_nm.fun):
                    x_curr = res_nm.x
            consider(x_curr.reshape(N, D))
        except Exception:
            pass

    if best_points is None:
        for init_x in inits:
            consider(init_x)
        if best_points is None:
            best_points = fibonacci_sphere(N)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq = get_pairwise_sq_dists(pts)
    print(f"Score: {np.max(sq)/np.min(sq)}")
