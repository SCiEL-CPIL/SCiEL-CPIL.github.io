import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14
N_HALF = 7

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=50.0):
    """Soft-min objective: differentiable approximation of -min_area/hull_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

# --- Centrally symmetric parameterization ---
# For N=14, we use 7 independent vertex pairs (v_i, -v_i).
# Parameterize by 7 angles and 7 radii: v_i = r_i * (cos(theta_i), sin(theta_i))

def _sym_params_to_points(params):
    """Generate 14 points from 7 angles and 7 radii (centrally symmetric)."""
    angles = params[:7]
    radii = params[7:]
    radii = np.maximum(radii, 1e-8)
    half = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return np.vstack([half, -half])

def _objective_sym(params):
    """Hard objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_sym_soft(params, beta=150.0):
    """Soft objective for centrally symmetric parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _sym_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _generate_sym_inits():
    """Generate initial parameter vectors for centrally symmetric optimization."""
    inits = []
    
    # Regular heptagon-based (7 equally spaced angles)
    base_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)]))
    
    # Rotated versions
    for rot in [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.3]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]))
    
    # Non-uniform angle patterns with varying radii
    rng = np.random.default_rng(42)
    for seed in range(25):
        rng = np.random.default_rng(seed * 11 + 5)
        angles = np.sort(rng.uniform(0, 2*np.pi, 7))
        radii = 1.0 + rng.normal(0, 0.15, 7)
        radii = np.maximum(radii, 0.5)
        inits.append(np.concatenate([angles, radii]))
    
    # Pattern: alternating radii
    for r_pattern in [
        [1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0],
        [1.0, 0.7, 0.9, 0.7, 1.0, 0.7, 0.9],
        [1.2, 0.9, 1.1, 0.9, 1.2, 0.9, 1.1],
    ]:
        a = base_angles + 0.2
        inits.append(np.concatenate([a, np.array(r_pattern)]))
    
    # Jittered regular
    for js in [0.05, 0.1, 0.2]:
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js*0.5, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = -res.fun
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft(points_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min objective for smoother landscape."""
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # Two concentric rings (7+7) with various radii and offsets
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        initializations.append(pts)
    
    # Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        initializations.append(pts)
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (1.1, 0.9)]:
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        initializations.append(np.column_stack([rx*np.cos(a), ry*np.sin(a)]))
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)

    # 2. Refine initializations
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
            
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        if time.time() < deadline - 25.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Powell', maxiter=500)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 3. Symmetric optimization (KEY STRUCTURAL CHANGE)
    # Exploit central symmetry: optimize 7 points, reflect to get 14
    if time.time() < deadline - 30.0:
        sym_inits = _generate_sym_inits()
        
        for i, flat_init in enumerate(sym_inits):
            if time.time() > deadline - 25.0:
                break
            
            # Nelder-Mead on symmetric parameters
            try:
                res_sym = minimize(_objective_sym, flat_init, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res_sym.fun < 0:
                    pts14 = _sym_params_to_points(res_sym.x)
                    cand_score = -res_sym.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass
            
            # Powell on symmetric parameters
            if time.time() < deadline - 20.0:
                try:
                    res_sym_pow = minimize(_objective_sym, flat_init, method='Powell',
                                           options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res_sym_pow.fun < 0:
                        pts14 = _sym_params_to_points(res_sym_pow.x)
                        cand_score = -res_sym_pow.fun
                        update_best(pts14, cand_score)
                except Exception:
                    pass

    # 4. Differential Evolution on symmetric space (14D instead of 28D)
    if time.time() < deadline - 20.0:
        def de_sym_objective(flat):
            if np.any(~np.isfinite(flat)):
                return 1.0
            pts = _sym_params_to_points(flat)
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            bounds = [(-np.pi, np.pi)] * 7 + [(0.1, 2.0)] * 7
            res_de = differential_evolution(
                de_sym_objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                pts14 = _sym_params_to_points(res_de.x)
                cand_score = -res_de.fun
                update_best(pts14, cand_score)
        except Exception:
            pass

    # 5. Final polish on the global best
    if best_points is not None and time.time() < deadline - 10.0:
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 10.0:
                break
            cand_pts, cand_score = refine(best_points, method, maxiter=1000)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
        
        # Soft refinement
        if time.time() < deadline - 8.0:
            cand_pts, cand_score = refine_soft(best_points, 'L-BFGS-B', maxiter=2000, beta=50.0)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # 6. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            noise = rng2.normal(0, 0.1, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
