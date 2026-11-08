import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 145.0  # Safety margin before 150s kill

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas."""
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    i_idx = np.array([c[0] for c in idx])
    j_idx = np.array([c[1] for c in idx])
    k_idx = np.array([c[2] for c in idx])
    v1 = pts[j_idx] - pts[i_idx]
    v2 = pts[k_idx] - pts[i_idx]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _soft_min(areas, tau=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = np.min(areas)
    shifted = areas - m
    shifted = np.clip(shifted, 0, 10.0 / tau)
    return m - (1.0 / tau) * np.log(np.sum(np.exp(-tau * shifted)))

def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    ma = np.min(_triangle_areas(pts))
    return ma / ha

def _normalize(pts):
    """Center and scale to unit hull area."""
    pts = pts - np.mean(pts, axis=0)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return pts
    return pts / np.sqrt(ha)

def _edge_params_to_points(params):
    """Convert edge length and angle params to 2D points.
    params: shape (2N,) - first N are edge lengths, next N are edge angles.
    Edge i goes from point i to point i+1 (mod N).
    The polygon closes automatically if we fix the first point at origin
    and the last edge angle is determined, but we parameterize all N angles
    and let the closure constraint be handled by adjusting.
    
    Actually, for a closed convex polygon, we have N edge vectors e_i = L_i * (cos a_i, sin a_i).
    The closure constraint is sum(e_i) = 0, which gives 2 constraints.
    So we really have 2N - 2 free parameters. But let's just use 2N params
    and during optimization, if the polygon doesn't close, we add a penalty
    or adjust. A better approach: use N-1 edge angles and N edge lengths,
    and the last edge angle is determined by closure. But that's complex.
    
    Alternative: use the same radial/angular parametrization but optimize with DE.
    Let's stick with the radial/angular but use DE for the global search.
    """
    angles_base = np.linspace(0, 2 * np.pi, N, endpoint=False)
    d_angles = params[:N]
    d_radii = params[N:]
    angles = angles_base + d_angles
    radii = 1.0 + d_radii
    radii = np.maximum(radii, 0.1)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _neg_score_param(params):
    """Negative score for parametrized points (for minimization)."""
    pts = _edge_params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    return -_min_triangle_area(pts)

def _min_triangle_area(pts):
    return np.min(_triangle_areas(pts))

def _smooth_objective_param(params, tau=50.0):
    """Smooth surrogate for parametrized points."""
    pts = _edge_params_to_points(params)
    pts = _normalize(pts)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    areas = _triangle_areas(pts)
    m = np.min(areas)
    shifted = areas - m
    shifted = np.clip(shifted, 0, 10.0 / tau)
    sum_exp = np.sum(np.exp(-tau * shifted))
    return -m + (1.0 / tau) * np.log(sum_exp)

def _local_refine(pts, max_iters=300, base_step=0.01):
    """Local refinement: perturb points involved in worst triangles."""
    pts = pts.copy()
    current = _score(pts)
    n = len(pts)
    idx = list(itertools.combinations(range(n), 3))
    
    for it in range(max_iters):
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        worst_tris = np.where(areas - min_a < 1e-10)[0]
        worst_pts = set()
        for ti in worst_tris[:30]:
            for p in idx[ti]:
                worst_pts.add(p)
        
        improved = False
        step = base_step * (0.7 ** (it // 50))
        for pi in sorted(worst_pts):
            # Try multiple step sizes in random directions
            for _ in range(8):
                dx = np.random.uniform(-step, step)
                dy = np.random.uniform(-step, step)
                new_pts = pts.copy()
                new_pts[pi, 0] += dx
                new_pts[pi, 1] += dy
                s = _score(new_pts)
                if s > current + 1e-12:
                    pts = new_pts
                    current = s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try all points with smaller steps
            for pi in range(n):
                for _ in range(4):
                    dx = np.random.uniform(-step * 0.5, step * 0.5)
                    dy = np.random.uniform(-step * 0.5, step * 0.5)
                    new_pts = pts.copy()
                    new_pts[pi, 0] += dx
                    new_pts[pi, 1] += dy
                    s = _score(new_pts)
                    if s > current + 1e-12:
                        pts = new_pts
                        current = s
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    return pts, current

def _local_refine_param(params, max_iters=200, base_step=0.01):
    """Local refinement in parameter space."""
    current = -_min_triangle_area(_edge_params_to_points(params))
    for it in range(max_iters):
        pts = _edge_params_to_points(params)
        pts = _normalize(pts)
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        worst_tris = np.where(areas - min_a < 1e-10)[0]
        worst_pts = set()
        idx = list(itertools.combinations(range(N), 3))
        for ti in worst_tris[:30]:
            for p in idx[ti]:
                worst_pts.add(p)
        
        improved = False
        step = base_step * (0.7 ** (it // 50))
        for pi in sorted(worst_pts):
            for _ in range(6):
                d_params = np.random.uniform(-step, step, size=2)
                new_params = params.copy()
                new_params[pi] += d_params[0]
                new_params[N + pi] += d_params[1]
                new_pts = _edge_params_to_points(new_params)
                new_pts = _normalize(new_pts)
                s = _score(new_pts)
                if s > -current + 1e-12:
                    params = new_params
                    current = -s
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return params, -current

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    best_score = -1.0
    best_points = None
    
    # Initial configurations
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(x0.copy())
    
    # 2. Perturbed regular 14-gon
    for scale in [0.02, 0.05, 0.1, 0.15]:
        configs.append(x0 + scale * np.random.randn(N, 2))
    
    # 3. Two concentric rings
    ang1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    outer = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = ang1 + np.pi / 7
    inner = 0.55 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    configs.append(np.vstack([outer, inner]))
    
    # 4. Three rings
    ang1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1 = np.column_stack([np.cos(ang1), np.sin(ang1)])
    ang2 = ang1 + np.pi / 5
    r2 = 0.65 * np.column_stack([np.cos(ang2), np.sin(ang2)])
    ang3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    r3 = 0.35 * np.column_stack([np.cos(ang3), np.sin(ang3)])
    configs.append(np.vstack([r1, r2, r3]))
    
    # 5. Random in unit disk
    for _ in range(4):
        r = np.sqrt(np.random.rand(N))
        th = np.random.rand(N) * 2 * np.pi
        configs.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    
    # First pass: local refine each initial config
    for x_init in configs:
        if time.time() > deadline:
            break
        try:
            p = _normalize(x_init)
            s = _score(p)
            if s > best_score:
                best_score = s
                best_points = p.copy()
            
            if time.time() < deadline - 20:
                p_r, s_r = _local_refine(p, max_iters=100, base_step=0.01)
                if s_r > best_score:
                    best_score = s_r
                    best_points = p_r.copy()
        except Exception:
            pass
    
    # Differential evolution in parameter space
    if time.time() < deadline - 30:
        try:
            # Bounds: angles in [-0.3, 0.3], radii in [-0.3, 0.3]
            bounds = [(-0.3, 0.3)] * N + [(-0.3, 0.3)] * N
            
            def de_objective(params):
                pts = _edge_params_to_points(params)
                pts = _normalize(pts)
                ha = _hull_area(pts)
                if ha < 1e-10:
                    return 1e6
                return -_min_triangle_area(pts)
            
            # Use multiple seeds for DE
            for seed in range(4):
                if time.time() > deadline - 20:
                    break
                try:
                    result = differential_evolution(
                        de_objective, bounds,
                        seed=seed,
                        maxiter=25,
                        popsize=12,
                        tol=1e-8,
                        mutation=(0.3, 0.8),
                        recombination=0.8,
                        polish=False,
                        init='sobol'
                    )
                    
                    p_de = _edge_params_to_points(result.x)
                    p_de = _normalize(p_de)
                    s_de = _score(p_de)
                    
                    if s_de > best_score:
                        best_score = s_de
                        best_points = p_de.copy()
                    
                    # Refine the DE result in param space
                    if time.time() < deadline - 15:
                        params_r, s_r = _local_refine_param(result.x, max_iters=100, base_step=0.005)
                        pts_r = _edge_params_to_points(params_r)
                        pts_r = _normalize(pts_r)
                        s_r2 = _score(pts_r)
                        if s_r2 > best_score:
                            best_score = s_r2
                            best_points = pts_r.copy()
                except Exception:
                    pass
        except Exception:
            pass
    
    # Nelder-Mead polish on best
    if best_points is not None and time.time() < deadline - 8:
        try:
            def neg_obj(flat):
                pts = flat.reshape(-1, 2)
                pts = _normalize(pts)
                ha = _hull_area(pts)
                if ha < 1e-10:
                    return 1e6
                return -_min_triangle_area(pts)
            
            res = minimize(neg_obj, best_points.flatten(),
                          method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            p_nm = res.x.reshape(-1, 2)
            p_nm = _normalize(p_nm)
            s_nm = _score(p_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_points = p_nm.copy()
        except Exception:
            pass
    
    # Final local refinement
    if best_points is not None and time.time() < deadline - 3:
        try:
            p_final, s_final = _local_refine(best_points, max_iters=200, base_step=0.003)
            if s_final > best_score:
                best_points = p_final
                best_score = s_final
        except Exception:
            pass
    
    if best_points is None:
        best_points = _normalize(x0.copy())
    
    return best_points
