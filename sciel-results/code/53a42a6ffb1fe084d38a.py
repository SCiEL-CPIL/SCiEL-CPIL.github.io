import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
_tri_indices = np.array(list(combinations(range(N), 3)))
_n_tri = len(_tri_indices)

def _calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    bary: (N, 2) array where column 0 is u, column 1 is v, w = 1 - u - v.
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[_tri_indices[:, 0]]
    vi = v[_tri_indices[:, 0]]
    wi = w[_tri_indices[:, 0]]
    
    uj = u[_tri_indices[:, 1]]
    vj = v[_tri_indices[:, 1]]
    wj = w[_tri_indices[:, 1]]
    
    uk = u[_tri_indices[:, 2]]
    vk = v[_tri_indices[:, 2]]
    wk = w[_tri_indices[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def _clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    bary: (N, 2) array
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def _bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def _min_area_bary(bary):
    """Compute min area from barycentric coordinates (u, v)."""
    areas = _calc_areas_bary(bary)
    return float(np.min(areas))

def _soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = _clamp_bary(bary)
    
    areas = _calc_areas_bary(bary)
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def _delaunay_smooth_obj(bary_flat, lam=1.0):
    """
    Smooth surrogate: minimize sum of squared deviations of Delaunay triangle areas
    from the minimum area. This creates a differentiable landscape that encourages
    uniform small-triangle areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = _clamp_bary(bary)
    cart = _bary_to_cart(bary)
    
    try:
        tri = Delaunay(cart)
        simplices = tri.simplices
        
        if len(simplices) == 0:
            return _hard_min_obj(bary_flat)
        
        areas = np.zeros(len(simplices))
        for i, (a, b, c) in enumerate(simplices):
            ab = cart[b] - cart[a]
            ac = cart[c] - cart[a]
            areas[i] = 0.5 * abs(ab[0]*ac[1] - ab[1]*ac[0])
        
        target = np.min(areas)
        deviations = areas - target
        obj = lam * np.sum(deviations**2) - target
        
        return obj
    except Exception:
        return _hard_min_obj(bary_flat)

def _hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = _clamp_bary(bary)
    areas = _calc_areas_bary(bary)
    return -np.min(areas)

def _init_hexagonal_bary():
    """
    Hexagonal lattice initialization in barycentric coordinates (u, v).
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
            val = -_hard_min_obj(bary.flatten())
            if val > best_val:
                best_val = val
                best_bary = bary.copy()
    
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
    
    return best_bary

def _init_structured_bary():
    """
    Structured initialization in barycentric coordinates (u, v).
    """
    bary_pts = []
    # Vertices
    bary_pts.append(np.array([1.0, 0.0]))
    bary_pts.append(np.array([0.0, 1.0]))
    bary_pts.append(np.array([0.0, 0.0]))
    
    # Edge midpoints
    bary_pts.append(np.array([0.5, 0.5]))
    bary_pts.append(np.array([0.0, 0.5]))
    bary_pts.append(np.array([0.5, 0.0]))
    
    # Centroid
    bary_pts.append(np.array([1/3, 1/3]))
    
    # 4 more points in symmetric positions
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
        du = 0.12 * np.cos(angle)
        dv = 0.12 * np.sin(angle)
        pt = np.array([1/3 + du, 1/3 + dv])
        pt = _clamp_bary(pt.reshape(1, 2))[0]
        bary_pts.append(pt)
        if len(bary_pts) == N:
            break
    
    return np.array(bary_pts)

def _init_random_bary(seed):
    """Random initialization in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def _init_perturbed_bary(base_bary, rng, scale=0.1):
    """Perturb a base barycentric configuration."""
    perturbation = rng.normal(0, scale, size=base_bary.shape)
    candidate = base_bary + perturbation
    return _clamp_bary(candidate)

def _simulated_annealing(bary_init, deadline, time_left_fn, rng):
    """
    Simulated annealing to escape local minima.
    Works in barycentric space with adaptive step size.
    """
    current = bary_init.copy()
    current_score = _min_area_bary(current)
    best = current.copy()
    best_score = current_score
    
    T_start = 0.08
    T_end = 1e-9
    n_iter = 5000
    cooling = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    
    T = T_start
    for i in range(n_iter):
        if time_left_fn() < 15.0:
            break
        
        step_scale = 0.03 + 0.1 * (T / T_start)
        
        perturbation = rng.normal(0, step_scale, size=current.shape)
        candidate = current + perturbation
        candidate = _clamp_bary(candidate)
        
        candidate_score = _min_area_bary(candidate)
        delta = candidate_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best_score = current_score
                best = current.copy()
        
        T *= cooling
    
    return best, best_score

def _optimize_bary(bary_init, deadline, time_left_fn, rng):
    """
    Optimize from a given barycentric initialization using multi-stage refinement.
    Returns (best_bary, best_score).
    """
    best_bary = bary_init.copy()
    best_score = _min_area_bary(best_bary)
    
    bounds = [(0, 1)] * (N * 2)
    
    # Stage 1: L-BFGS-B with soft-min objective
    betas = [20.0, 50.0, 150.0, 400.0]
    current = best_bary.copy()
    
    for beta in betas:
        if time_left_fn() < 10.0:
            break
        
        try:
            res = minimize(
                _soft_min_obj,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            opt_bary = _clamp_bary(res.x.reshape(-1, 2))
            score = _min_area_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary.copy()
                current = opt_bary.copy()
        except Exception:
            pass
    
    # Stage 2: Delaunay smooth objective refinement
    if time_left_fn() > 20.0:
        try:
            res = minimize(
                _delaunay_smooth_obj,
                best_bary.flatten(),
                args=(1.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            del_bary = _clamp_bary(res.x.reshape(-1, 2))
            del_score = _min_area_bary(del_bary)
            
            if del_score > best_score:
                best_score = del_score
                best_bary = del_bary.copy()
        except Exception:
            pass
    
    # Stage 3: Simulated Annealing to escape local minima
    if time_left_fn() > 25.0:
        try:
            sa_bary, sa_score = _simulated_annealing(best_bary, deadline, time_left_fn, rng)
            if sa_score > best_score:
                best_score = sa_score
                best_bary = sa_bary.copy()
        except Exception:
            pass
    
    # Stage 4: Nelder-Mead on hard objective
    if time_left_fn() > 8.0:
        try:
            res = minimize(
                _hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            nm_bary = _clamp_bary(res.x.reshape(-1, 2))
            nm_score = _min_area_bary(nm_bary)
            
            if nm_score > best_score:
                best_score = nm_score
                best_bary = nm_bary.copy()
        except Exception:
            pass
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -1.0
    
    # Generate multiple diverse initializations
    inits = []
    
    # Hexagonal initialization
    inits.append(_init_hexagonal_bary())
    
    # Structured initialization
    inits.append(_init_structured_bary())
    
    # Several random initializations
    for seed in [1, 7, 13, 23, 37, 41, 53, 67, 79, 97]:
        if time_left() < 25.0:
            break
        inits.append(_init_random_bary(seed))
    
    # Optimize from each initialization
    for i, init_bary in enumerate(inits):
        if time_left() < 15.0:
            break
        
        try:
            bary_opt, score = _optimize_bary(init_bary, deadline, time_left, rng)
            if score > best_score:
                best_score = score
                best_bary = bary_opt.copy()
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = _init_structured_bary()
        best_score = _min_area_bary(best_bary)
    
    # Final polish on the best solution
    if time_left() > 20.0:
        try:
            # SLSQP with constraints
            def _constrained_objective(flat_bary):
                bary = flat_bary.reshape(-1, 2)
                return -_min_area_bary(bary)
            
            constraints = {'type': 'eq', 'fun': lambda x: x.reshape(-1, 2)[:, 0] + x.reshape(-1, 2)[:, 1] - 1.0}
            bounds = [(0, 1)] * (N * 2)
            
            res = minimize(
                _constrained_objective,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 800, 'ftol': 1e-15}
            )
            
            if res.success:
                slsqp_bary = _clamp_bary(res.x.reshape(-1, 2))
                slsqp_score = _min_area_bary(slsqp_bary)
                if slsqp_score > best_score:
                    best_score = slsqp_score
                    best_bary = slsqp_bary.copy()
        except Exception:
            pass
    
    # Final high-precision Nelder-Mead polish
    if time_left() > 8.0:
        try:
            res = minimize(
                _hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            final_bary = _clamp_bary(res.x.reshape(-1, 2))
            final_score = _min_area_bary(final_bary)
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary.copy()
        except Exception:
            pass
    
    # Convert to cartesian for output
    best_cart = _bary_to_cart(_clamp_bary(best_bary))
    
    return best_cart

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_bary(_clamp_bary(pts.reshape(N, 2)))}")
