import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

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

def _objective_soft_multi_beta(flat_points, betas=None):
    """Multi-scale soft-min: average over multiple beta values for robust landscape."""
    if betas is None:
        betas = [20.0, 50.0, 100.0, 200.0]
    vals = []
    for b in betas:
        vals.append(_objective_soft(flat_points, beta=b))
    return np.mean(vals)

# --- Hexagonal lattice parameterization ---
# Parameterize 14 points as a subset of a 2D lattice defined by basis vectors e1, e2.
# Points are: origin + integer combinations of e1, e2 within a small region.
# We optimize: e1 (2D), e2 (2D), and 14 offset vectors (28D) = 32D total.
# But we can reduce: fix the lattice structure, optimize which 14 lattice points to use.
# Better: parameterize as lattice basis + 14 relative positions from a reference point.

# Standard hexagonal lattice basis
def _hex_lattice_basis(scale=1.0, angle=0.0):
    """Return basis vectors for a hexagonal lattice."""
    e1 = np.array([scale, 0.0])
    e2 = np.array([scale * np.cos(angle + np.pi/3), scale * np.sin(angle + np.pi/3)])
    return e1, e2

def _lattice_points(e1, e2, n_range=(-2, 3)):
    """Generate lattice points from basis vectors within a range."""
    points = []
    for i in range(n_range[0], n_range[1]):
        for j in range(n_range[0], n_range[1]):
            points.append(i * e1 + j * e2)
    return np.array(points)

def _select_best_subset(points, k, ref_point=None):
    """Select k points from a larger set to maximize minimum pairwise distance."""
    if len(points) <= k:
        return points
    if ref_point is None:
        ref_point = np.mean(points, axis=0)
    
    # Greedy selection: start with point farthest from center, then add farthest from existing
    selected = [ref_point]
    remaining = list(range(len(points)))
    
    # Find point farthest from center
    dists = np.linalg.norm(points - ref_point, axis=1)
    best_idx = np.argmax(dists)
    selected.append(points[best_idx])
    remaining.remove(best_idx)
    
    while len(selected) < k and remaining:
        # Find point in remaining that maximizes min distance to selected
        min_dists = np.min([np.linalg.norm(points[i] - np.array(selected), axis=1) 
                           for i in remaining], axis=0)
        best_idx = remaining[np.argmax(min_dists)]
        selected.append(points[best_idx])
        remaining.remove(best_idx)
    
    return np.array(selected)

def _lattice_params_to_points(params):
    """
    Convert lattice parameters to 14 points.
    params: [e1x, e1y, e2x, e2y, offsets...] where offsets are 14*2=28 values
    Total: 32 parameters
    """
    e1 = np.array([params[0], params[1]])
    e2 = np.array([params[2], params[3]])
    offsets = params[4:].reshape(14, 2)
    
    # Generate lattice points
    lattice_pts = _lattice_points(e1, e2, n_range=(-2, 3))
    
    # Select 14 points closest to the offset targets
    selected = []
    used = set()
    for target in offsets:
        # Find closest lattice point not already used
        dists = np.linalg.norm(lattice_pts - target, axis=1)
        for idx in np.argsort(dists):
            if idx not in used:
                selected.append(lattice_pts[idx])
                used.add(idx)
                break
    
    return np.array(selected)

def _objective_lattice(params):
    """Hard objective for lattice parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _lattice_params_to_points(params)
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_lattice_soft(params, beta=100.0):
    """Soft objective for lattice parameters."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _lattice_params_to_points(params)
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

def _generate_lattice_inits():
    """Generate initial parameter vectors for lattice optimization."""
    inits = []
    
    # Standard hexagonal lattice
    e1_std = np.array([1.0, 0.0])
    e2_std = np.array([0.5, np.sqrt(3)/2])
    
    # Generate lattice points and select 14
    lattice_pts = _lattice_points(e1_std, e2_std, n_range=(-2, 3))
    
    # Try different subsets
    rng = np.random.default_rng(42)
    for trial in range(20):
        # Random subset selection
        idx = rng.choice(len(lattice_pts), size=14, replace=False)
        selected = lattice_pts[idx]
        
        # Compute offsets from center
        center = np.mean(selected, axis=0)
        offsets = (selected - center).flatten()
        
        inits.append(np.concatenate([e1_std, e2_std, offsets]))
    
    # Regular hexagonal pattern: center + 6 around + 7 more
    center = np.array([0.0, 0.0])
    ring1 = []
    for i in range(6):
        angle = i * np.pi / 3
        ring1.append(e1_std * np.cos(angle) + e2_std * np.sin(angle))
    
    # Add second ring
    ring2 = []
    for i in range(7):
        angle = i * 2 * np.pi / 7
        ring2.append(2.0 * (e1_std * np.cos(angle) + e2_std * np.sin(angle)))
    
    selected = np.vstack([center.reshape(1,2), np.array(ring1), np.array(ring2)])
    center_sel = np.mean(selected, axis=0)
    offsets = (selected - center_sel).flatten()
    inits.append(np.concatenate([e1_std, e2_std, offsets]))
    
    # Triangular lattice variations
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        e1 = scale * np.array([1.0, 0.0])
        e2 = scale * np.array([0.5, np.sqrt(3)/2])
        lattice_pts = _lattice_points(e1, e2, n_range=(-2, 3))
        idx = rng.choice(len(lattice_pts), size=14, replace=False)
        selected = lattice_pts[idx]
        center = np.mean(selected, axis=0)
        offsets = (selected - center).flatten()
        inits.append(np.concatenate([e1, e2, offsets]))
    
    # Square lattice
    for scale in [0.9, 1.0, 1.1]:
        e1 = scale * np.array([1.0, 0.0])
        e2 = scale * np.array([0.0, 1.0])
        lattice_pts = _lattice_points(e1, e2, n_range=(-2, 3))
        idx = rng.choice(len(lattice_pts), size=14, replace=False)
        selected = lattice_pts[idx]
        center = np.mean(selected, axis=0)
        offsets = (selected - center).flatten()
        inits.append(np.concatenate([e1, e2, offsets]))
    
    # Skewed hexagonal
    for skew in [0.1, 0.2, 0.3]:
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.5 + skew, np.sqrt(3)/2])
        lattice_pts = _lattice_points(e1, e2, n_range=(-2, 3))
        idx = rng.choice(len(lattice_pts), size=14, replace=False)
        selected = lattice_pts[idx]
        center = np.mean(selected, axis=0)
        offsets = (selected - center).flatten()
        inits.append(np.concatenate([e1, e2, offsets]))
    
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

    # 3. Lattice-based optimization (KEY STRUCTURAL CHANGE)
    # Optimize lattice basis vectors and point selection
    if time.time() < deadline - 30.0:
        lattice_inits = _generate_lattice_inits()
        
        for i, flat_init in enumerate(lattice_inits):
            if time.time() > deadline - 25.0:
                break
            
            # Nelder-Mead on lattice parameters
            try:
                res_lat = minimize(_objective_lattice, flat_init, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                if res_lat.fun < 0:
                    pts14 = _lattice_params_to_points(res_lat.x)
                    cand_score = -res_lat.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass
            
            # Powell on lattice parameters
            if time.time() < deadline - 20.0:
                try:
                    res_lat_pow = minimize(_objective_lattice, flat_init, method='Powell',
                                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9})
                    if res_lat_pow.fun < 0:
                        pts14 = _lattice_params_to_points(res_lat_pow.x)
                        cand_score = -res_lat_pow.fun
                        update_best(pts14, cand_score)
                except Exception:
                    pass
            
            # Soft lattice refinement
            if time.time() < deadline - 18.0:
                try:
                    res_lat_soft = minimize(lambda p: _objective_lattice_soft(p, beta=150.0), flat_init, method='L-BFGS-B',
                                            options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10})
                    pts14s = _lattice_params_to_points(res_lat_soft.x)
                    cand_scores = _score(pts14s)
                    update_best(pts14s, cand_scores)
                except Exception:
                    pass

    # 4. Differential Evolution on lattice space
    if time.time() < deadline - 25.0:
        def de_lattice_objective(flat):
            if np.any(~np.isfinite(flat)):
                return 1.0
            pts = _lattice_params_to_points(flat)
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            # Bounds: e1, e2 in [-2, 2]^2, offsets in [-3, 3]^2
            bounds = [(-2.0, 2.0)] * 4 + [(-3.0, 3.0)] * 28
            res_de = differential_evolution(
                de_lattice_objective, 
                bounds, 
                seed=42, 
                maxiter=50, 
                popsize=12, 
                tol=1e-5, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                pts14 = _lattice_params_to_points(res_de.x)
                cand_score = -res_de.fun
                update_best(pts14, cand_score)
        except Exception:
            pass
        
        # Second DE run with different seed if time allows
        if time.time() < deadline - 15.0:
            try:
                res_de2 = differential_evolution(
                    de_lattice_objective, 
                    bounds, 
                    seed=123, 
                    maxiter=30, 
                    popsize=10, 
                    tol=1e-5, 
                    mutation=(0.3, 0.8), 
                    recombination=0.8,
                    polish=True,
                    init='sobol'
                )
                if res_de2.success or res_de2.fun < 0:
                    pts14 = _lattice_params_to_points(res_de2.x)
                    cand_score = -res_de2.fun
                    update_best(pts14, cand_score)
            except Exception:
                pass

    # 5. Final polish on the global best - multi-stage
    if best_points is not None and time.time() < deadline - 15.0:
        # Stage 1: Multi-beta soft refinement
        if time.time() < deadline - 12.0:
            try:
                center = np.mean(best_points, axis=0)
                pts_norm = best_points - center
                max_norm = np.max(np.linalg.norm(pts_norm, axis=1))
                if max_norm > 0:
                    pts_norm = pts_norm / max_norm
                res_multi = minimize(lambda x: _objective_soft_multi_beta(x), pts_norm.flatten(),
                                     method='L-BFGS-B',
                                     options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12})
                cand_pts = res_multi.x.reshape(N, 2)
                cand_score = _score(cand_pts)
                update_best(cand_pts, cand_score)
            except Exception:
                pass
        
        # Stage 2: Hard Nelder-Mead with tight tolerance
        if time.time() < deadline - 10.0:
            cand_pts, cand_score = refine(best_points, 'Nelder-Mead', maxiter=8000, tol=1e-10)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
        
        # Stage 3: Powell with tight tolerance
        if time.time() < deadline - 8.0:
            cand_pts, cand_score = refine(best_points, 'Powell', maxiter=5000, tol=1e-10)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # 6. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 12.0:
        rng2 = np.random.default_rng(42)
        for k in range(10):
            if time.time() > deadline - 7.0:
                break
            noise = rng2.normal(0, 0.04 + 0.015*k, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=1200)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
            
            # Also try soft refinement on perturbed
            if time.time() < deadline - 6.0:
                cand_pts2, cand_score2 = refine_soft(perturbed, method='L-BFGS-B', maxiter=2000, beta=100.0)
                if cand_pts2 is not None:
                    update_best(cand_pts2, cand_score2)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
