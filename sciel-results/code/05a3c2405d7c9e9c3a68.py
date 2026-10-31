import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute all triangle index combinations
_tri_indices = np.array(list(combinations(range(N), 3)))
_n_tri = len(_tri_indices)

def _bary_to_cart(bary):
    """Convert barycentric coords (a,b,c) to cartesian. bary shape (n,3) with a+b+c=1."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert cartesian points to barycentric coords."""
    x = pts[:, 0]
    y = pts[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _clamp_bary(bary):
    """Clamp barycentric coords to be non-negative and sum to 1."""
    bary = np.maximum(bary, 0.0)
    s = bary.sum(axis=1, keepdims=True)
    s = np.maximum(s, 1e-15)
    bary = bary / s
    return bary

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from cartesian points."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Exact minimum area from cartesian points."""
    return float(np.min(_all_areas_cart(pts)))

def _min_area_bary(bary):
    """Compute min area from barycentric coordinates."""
    cart = _bary_to_cart(_clamp_bary(bary))
    return _min_area_cart(cart)

def _soft_min_area_bary(bary, beta=100.0):
    """Smooth approximation of min area in barycentric coords."""
    cart = _bary_to_cart(_clamp_bary(bary))
    areas = _all_areas_cart(cart)
    # Use log-sum-exp for numerical stability
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _objective_bary(flat_bary, beta=100.0):
    """Objective to minimize: negative soft-min area, working in barycentric space."""
    bary = flat_bary.reshape(-1, 3)
    return -_soft_min_area_bary(bary, beta)

def _objective_hard_bary(flat_bary):
    """Objective to minimize: negative exact min area, working in barycentric space."""
    bary = flat_bary.reshape(-1, 3)
    return -_min_area_bary(bary)

def _init_hexagonal_bary():
    """
    Hexagonal lattice initialization in barycentric coordinates.
    Uses a regular grid pattern that approximates optimal packing.
    """
    bary_pts = []
    
    # Create a hexagonal-like pattern using barycentric coordinates
    # Start with vertices and edge midpoints
    bary_pts.append(np.array([1.0, 0.0, 0.0]))
    bary_pts.append(np.array([0.0, 1.0, 0.0]))
    bary_pts.append(np.array([0.0, 0.0, 1.0]))
    
    # Edge midpoints
    bary_pts.append(np.array([0.5, 0.5, 0.0]))
    bary_pts.append(np.array([0.0, 0.5, 0.5]))
    bary_pts.append(np.array([0.5, 0.0, 0.5]))
    
    # Interior points in hexagonal pattern
    # Use barycentric coordinates for a 3x3 grid-like pattern
    interior = [
        [1/3, 1/3, 1/3],  # centroid
        [0.4, 0.3, 0.3],
        [0.3, 0.4, 0.3],
        [0.3, 0.3, 0.4],
        [0.45, 0.25, 0.3],
        [0.25, 0.45, 0.3],
        [0.25, 0.3, 0.45],
    ]
    for pt in interior:
        pt = np.array(pt)
        pt = _clamp_bary(pt.reshape(1, 3))[0]
        bary_pts.append(pt)
    
    return np.array(bary_pts)

def _init_structured_bary():
    """
    Structured initialization in barycentric coordinates.
    Places points at vertices, edge midpoints, centroid, and symmetric positions.
    """
    bary_pts = []
    # Vertices
    bary_pts.append(np.array([1.0, 0.0, 0.0]))
    bary_pts.append(np.array([0.0, 1.0, 0.0]))
    bary_pts.append(np.array([0.0, 0.0, 1.0]))
    
    # Edge midpoints
    bary_pts.append(np.array([0.5, 0.5, 0.0]))
    bary_pts.append(np.array([0.0, 0.5, 0.5]))
    bary_pts.append(np.array([0.5, 0.0, 0.5]))
    
    # Centroid
    bary_pts.append(np.array([1/3, 1/3, 1/3]))
    
    # 4 more points in symmetric positions around centroid
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.12
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3]:
        da = r * np.cos(angle)
        db = r * np.sin(angle)
        dc = -da - db
        pt = np.array([cx + da, cy + db, cz + dc])
        pt = _clamp_bary(pt.reshape(1, 3))[0]
        bary_pts.append(pt)
        if len(bary_pts) == N:
            break
    
    return np.array(bary_pts)

def _init_random_bary(seed):
    """Random initialization in barycentric coordinates."""
    rng = np.random.default_rng(seed)
    bary_pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        pt = np.array([1 - r1 - r2, r1, r2])
        bary_pts.append(pt)
    return np.array(bary_pts)

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
    
    # Temperature schedule
    T_start = 0.08
    T_end = 1e-9
    n_iter = 5000
    cooling = (T_end / T_start) ** (1.0 / max(n_iter, 1))
    
    T = T_start
    for i in range(n_iter):
        if time_left_fn() < 15.0:
            break
        
        # Adaptive step size based on temperature
        step_scale = 0.03 + 0.1 * (T / T_start)
        
        # Generate perturbation
        perturbation = rng.normal(0, step_scale, size=current.shape)
        candidate = current + perturbation
        candidate = _clamp_bary(candidate)
        
        candidate_score = _min_area_bary(candidate)
        delta = candidate_score - current_score
        
        # Accept if better, or with probability exp(delta/T)
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
    
    # Stage 1: L-BFGS-B with increasing beta
    betas = [20.0, 50.0, 150.0, 400.0]
    current = best_bary.copy()
    
    for beta in betas:
        if time_left_fn() < 10.0:
            break
        
        try:
            res = minimize(
                _objective_bary,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            opt_bary = _clamp_bary(res.x.reshape(-1, 3))
            score = _min_area_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary.copy()
                current = opt_bary.copy()
        except Exception:
            pass
    
    # Stage 2: Simulated Annealing to escape local minima
    if time_left_fn() > 25.0:
        try:
            sa_bary, sa_score = _simulated_annealing(best_bary, deadline, time_left_fn, rng)
            if sa_score > best_score:
                best_score = sa_score
                best_bary = sa_bary.copy()
        except Exception:
            pass
    
    # Stage 3: Nelder-Mead on hard objective
    if time_left_fn() > 8.0:
        try:
            res = minimize(
                _objective_hard_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            nm_bary = _clamp_bary(res.x.reshape(-1, 3))
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
    
    # Final polish on the best solution with more iterations
    if time_left() > 20.0:
        try:
            # Try SLSQP with constraints
            def _constrained_objective(flat_bary):
                bary = flat_bary.reshape(-1, 3)
                return -_min_area_bary(bary)
            
            # Equality constraint: sum of each row = 1
            def _sum_constraint(flat_bary):
                bary = flat_bary.reshape(-1, 3)
                return bary.sum(axis=1) - 1.0
            
            constraints = {'type': 'eq', 'fun': _sum_constraint}
            bounds = [(0, 1)] * (N * 3)
            
            res = minimize(
                _constrained_objective,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 800, 'ftol': 1e-15}
            )
            
            if res.success:
                slsqp_bary = _clamp_bary(res.x.reshape(-1, 3))
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
                _objective_hard_bary,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True}
            )
            final_bary = _clamp_bary(res.x.reshape(-1, 3))
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
    print(f"Best score: {_min_area_cart(pts)}")
