import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 140.0  # Safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    n = len(pts)
    i = np.repeat(np.repeat(np.arange(n), n), n)
    j = np.repeat(np.tile(np.arange(n), n), n)
    k = np.tile(np.tile(np.arange(n), n), 1)
    # Keep only i < j < k
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    v1 = pts[j] - pts[i]
    v2 = pts[k] - pts[i]
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    ma = np.min(_triangle_areas(pts))
    return ma / ha

def _fourier_points(coeffs, angles, n_harmonics):
    """
    Generate points from Fourier series of boundary radius.
    coeffs: array of shape (2*(n_harmonics+1),) = [r0, a1, b1, a2, b2, ...]
    angles: array of shape (N,) - angular positions
    Returns points of shape (N, 2).
    """
    r0 = coeffs[0]
    thetas = angles
    radii = np.full(len(thetas), r0)
    for m in range(1, n_harmonics + 1):
        a = coeffs[2 * m]
        b = coeffs[2 * m + 1]
        radii += a * np.cos(m * thetas) + b * np.sin(m * thetas)
    # Ensure positive radius
    radii = np.maximum(radii, 0.1)
    x = radii * np.cos(thetas)
    y = radii * np.sin(thetas)
    return np.column_stack([x, y])

def _normalize(pts):
    """Center and scale to unit hull area."""
    pts = pts - np.mean(pts, axis=0)
    ha = _hull_area(pts)
    if ha < 1e-10:
        return pts
    return pts / np.sqrt(ha)

def _objective_fourier(params, n_harmonics):
    """
    Objective for differential evolution.
    params: [r0, a1, b1, ..., aM, bM, theta1, ..., thetaN]
    Returns negative min triangle area (to minimize).
    """
    n_coeff = 2 * (n_harmonics + 1)
    coeffs = params[:n_coeff]
    angles = params[n_coeff:]
    
    # Sort angles for well-distributed points
    angles_sorted = np.sort(angles)
    
    pts = _fourier_points(coeffs, angles_sorted, n_harmonics)
    
    # Check for degeneracy
    if np.any(np.abs(pts) > 100):
        return 1e6
    
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    min_a = np.min(areas)
    
    return -min_a

def _refine_local(pts, max_iters=50, base_step=0.003):
    """Local refinement by perturbing points in worst triangles."""
    pts = pts.copy()
    current = _score(pts)
    n = len(pts)
    
    for it in range(max_iters):
        if _remaining() < 10:
            break
        areas = _triangle_areas(pts)
        min_a = np.min(areas)
        worst_tris = np.where(areas < min_a * 1.005)[0]
        
        # Get point indices from worst triangles
        i = np.repeat(np.repeat(np.arange(n), n), n)
        j = np.repeat(np.tile(np.arange(n), n), n)
        k = np.tile(np.tile(np.arange(n), n), 1)
        mask = (i < j) & (j < k)
        i, j, k = i[mask], j[mask], k[mask]
        
        worst_pts = set()
        for ti in worst_tris[:8]:
            worst_pts.add(int(i[ti]))
            worst_pts.add(int(j[ti]))
            worst_pts.add(int(k[ti]))
        
        improved = False
        step = base_step * (0.8 ** (it // 20))
        for pi in sorted(worst_pts):
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[pi, 0] += dx
                    new_pts[pi, 1] += dy
                    s = _score(new_pts)
                    if s > current + 1e-13:
                        pts = new_pts
                        current = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return pts, current

def construct():
    start_time = time.time()
    best_score = -1.0
    best_points = None
    
    N_HARMONICS = 3  # Number of Fourier harmonics
    
    # Initial angle configurations
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Coefficient ranges for differential evolution
    # r0 should be positive, harmonics small
    r0_bounds = [(1.0, 3.0)]
    for m in range(1, N_HARMONICS + 1):
        r0_bounds += [(-0.3, 0.3), (-0.3, 0.3)]
    angle_bounds = [(0, 2 * np.pi)] * N
    
    bounds = r0_bounds + angle_bounds
    
    # Multiple DE runs with different strategies
    de_configs = [
        {'popsize': 20, 'maxiter': 80, 'tol': 1e-8, 'mutation': (0.5, 1.0), 'recombination': 0.8, 'seed': 42},
        {'popsize': 15, 'maxiter': 60, 'tol': 1e-8, 'mutation': (0.3, 0.8), 'recombination': 0.7, 'seed': 123},
        {'popsize': 25, 'maxiter': 50, 'tol': 1e-8, 'mutation': (0.5, 1.0), 'recombination': 0.9, 'seed': 7},
    ]
    
    for de_cfg in de_configs:
        if _remaining() < 20:
            break
        
        try:
            result = differential_evolution(
                _objective_fourier,
                bounds,
                args=(N_HARMONICS,),
                maxiter=de_cfg['maxiter'],
                popsize=de_cfg['popsize'],
                tol=de_cfg['tol'],
                mutation=de_cfg['mutation'],
                recombination=de_cfg['recombination'],
                seed=de_cfg['seed'],
                polish=False,
                init='latinhypercube',
                update_interval=1
            )
            
            if _remaining() < 15:
                break
            
            best_params = result.x
            n_coeff = 2 * (N_HARMONICS + 1)
            coeffs = best_params[:n_coeff]
            angles = np.sort(best_params[n_coeff:])
            
            pts = _fourier_points(coeffs, angles, N_HARMONICS)
            pts = _normalize(pts)
            
            # Local refinement
            if _remaining() > 25:
                pts, s = _refine_local(pts, max_iters=40, base_step=0.003)
            else:
                s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # Also try direct optimization on sorted angular positions on a circle
    if _remaining() > 30:
        def _circle_obj(theta_flat):
            thetas = np.sort(theta_flat)
            pts = np.column_stack([np.cos(thetas), np.sin(thetas)])
            ha = _hull_area(pts)
            if ha < 1e-10:
                return 1e6
            areas = _triangle_areas(pts)
            return -np.min(areas)
        
        try:
            result_c = differential_evolution(
                _circle_obj,
                [(0, 2 * np.pi)] * N,
                maxiter=40,
                popsize=15,
                tol=1e-8,
                seed=99,
                polish=False,
                init='latinhypercube'
            )
            
            if _remaining() > 15:
                thetas = np.sort(result_c.x)
                pts_c = np.column_stack([np.cos(thetas), np.sin(thetas)])
                pts_c = _normalize(pts_c)
                
                if _remaining() > 20:
                    pts_c, sc = _refine_local(pts_c, max_iters=30, base_step=0.002)
                else:
                    sc = _score(pts_c)
                
                if sc > best_score:
                    best_score = sc
                    best_points = pts_c.copy()
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on the best found
    if best_points is not None and _remaining() > 10:
        try:
            def _neg_score(flat):
                p = flat.reshape(-1, 2)
                p = _normalize(p)
                ha = _hull_area(p)
                if ha < 1e-10:
                    return 1e6
                return -np.min(_triangle_areas(p))
            
            res_final = minimize(_neg_score, best_points.flatten(),
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            
            pf = res_final.x.reshape(-1, 2)
            pf = _normalize(pf)
            sf = _score(pf)
            
            if sf > best_score:
                best_points = pf
                best_score = sf
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
