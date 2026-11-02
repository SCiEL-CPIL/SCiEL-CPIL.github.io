import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute the 165 combinations
_indices = list(itertools.combinations(range(N), 3))
_idx_arr = np.array(_indices)

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    i = _idx_arr[:, 0]
    j = _idx_arr[:, 1]
    k = _idx_arr[:, 2]
    
    xi, yi = points[i, 0], points[i, 1]
    xj, yj = points[j, 0], points[j, 1]
    xk, yk = points[k, 0], points[k, 1]
    
    # Area = 0.5 * |xi(yj-yk) + xj(yk-yi) + xk(yi-yj)|
    det = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
    areas = 0.5 * np.abs(det)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _project_to_triangle(points):
    """Project points that are slightly outside the triangle back inside."""
    # Barycentric projection is safer
    # Convert to barycentric
    bary = np.empty((N, 2))
    for m in range(N):
        x = points[m, 0]
        y = points[m, 1]
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
        
    # Clip barycentric coords to valid range [0, 1] and u+v <= 1
    u = np.clip(bary[:, 0], 0.0, None)
    v = np.clip(bary[:, 1], 0.0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
        
    # Convert back to Cartesian
    pts = np.empty((N, 2))
    pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
    pts[:, 1] = (1.0 - u - v) * H
    return pts

def _objective(params):
    """
    Objective function for optimization: negative of min area.
    params is a flat array of 22 coordinates (x0,y0,x1,y1,...,x10,y10).
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points)
    return -_min_area(points)

def _soft_min_area(points, beta=50.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    """
    areas = _compute_areas(points)
    min_a = np.min(areas)
    # Use log-sum-exp trick for numerical stability
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

def _objective_smooth(params, beta=50.0):
    """
    Smooth objective function for optimization: negative of soft min area.
    """
    points = params.reshape(N, 2)
    points = _project_to_triangle(points)
    return -_soft_min_area(points, beta)

def _initial_guess_lattice():
    """
    Generate a lattice-based initial guess.
    For 11 points, a 4x4 grid has 10 points. Add 1 more near center.
    Or use a 3x3x3... no, triangle.
    Rows: 1, 2, 3, 4, 1? Total 11.
    """
    pts = []
    # 4 rows of lattice
    for i in range(4):
        for j in range(4 - i):
            # Barycentric coords for lattice
            # u, v such that u+v <= 1
            # Row i (from top? or bottom?)
            # Let's map to barycentric directly
            # Standard lattice:
            # u = (4-i-1)/4 ? No.
            # Let's use Cartesian and convert.
            # Side length 4 units.
            x = j + 0.5 * i
            y = i * np.sqrt(3) / 2
            pts.append([x/4.0, y/(4.0*H)]) # Normalize to unit triangle
            
    # Add 11th point in the center of the "gap"
    # The 10 points are at vertices and edges of the 4-subtriangles.
    # Center of the whole triangle is a good bet.
    pts.append([0.5, H/3.0]) # Centroid
    
    pts = np.array(pts)
    return pts

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    # Convert to cart
    pts = np.empty((N, 2))
    pts[:, 0] = bary[:, 1] * 1.0 + (1.0 - bary[:, 0] - bary[:, 1]) * 0.5
    pts[:, 1] = (1.0 - bary[:, 0] - bary[:, 1]) * H
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    def _update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # Initial guesses
    # 1. Lattice
    init_pts = _initial_guess_lattice()
    init_pts = _project_to_triangle(init_pts)
    _update_best(init_pts, _min_area(init_pts))
    
    # 2. Random
    rand_pts = _random_valid_barycentric(rng)
    _update_best(rand_pts, _min_area(rand_pts))
    
    # Phase 1: Basin Hopping on exact min area
    # This is good for non-smooth objectives
    x0 = best_points.flatten()
    
    def _local_min(x):
        # Use Nelder-Mead to minimize -min_area
        # NM might struggle with non-smoothness, but it's the standard local step for BH
        res = minimize(
            _objective, 
            x, 
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        return res
        
    def _kick(x):
        # Perturb in barycentric space
        pts = x.reshape(N, 2)
        # Convert to bary
        bary = np.empty((N, 2))
        for m in range(N):
            xx = pts[m, 0]
            yy = pts[m, 1]
            bary[m, 0] = 1.0 - xx - yy / (2.0 * H)
            bary[m, 1] = xx - yy / (2.0 * H)
            
        # Add noise
        noise = rng.normal(0, 0.05, bary.shape)
        bary += noise
        
        # Project back
        u = np.clip(bary[:, 0], 0.0, None)
        v = np.clip(bary[:, 1], 0.0, None)
        s = u + v
        mask = s > 1.0
        if np.any(mask):
            u[mask] /= s[mask]
            v[mask] /= s[mask]
            
        new_pts = np.empty((N, 2))
        new_pts[:, 0] = v * 1.0 + (1.0 - u - v) * 0.5
        new_pts[:, 1] = (1.0 - u - v) * H
        
        return new_pts.flatten()

    # Run Basin Hopping
    # We need to check time frequently
    bh_results = []
    current_time = time.time()
    
    # Estimate iterations: NM takes ~5-10ms per iteration? 
    # Let's run for a fixed number of iterations and check time
    max_bh_iters = 50
    
    for i in range(max_bh_iters):
        if time.time() > deadline - 20:
            break
            
        try:
            bh_res = basinhopping(
                _objective, 
                x0, 
                minimizer=_local_min, 
                minimizer_kwargs={'options': {'maxiter': 200}},
                kick=_kick,
                niter=10, # Inner iterations
                T=1.0,
                stepsize=0.05,
                accept_test=lambda x, y: x.fun <= y.fun, # Only accept if better? No, BH accepts worse with prob exp(-delta/T). 
                # Default accept is fine.
                disp=False
            )
            
            new_pts = bh_res.x.reshape(N, 2)
            new_pts = _project_to_triangle(new_pts)
            score = _min_area(new_pts)
            _update_best(new_pts, score)
            x0 = new_pts.flatten()
            
        except Exception as e:
            pass

    # Phase 2: Refine with Nelder-Mead on smooth objective
    # This helps to "polish" the solution by navigating the soft landscape
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(
                _objective_smooth, 
                x0, 
                method='Nelder-Mead',
                args=(100.0,), # Higher beta for sharper min
                options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_triangle(new_pts)
            score = _min_area(new_pts)
            _update_best(new_pts, score)
        except Exception:
            pass

    # Phase 3: More Basin Hopping from the polished result
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        for i in range(20):
            if time.time() > deadline - 5:
                break
            try:
                bh_res = basinhopping(
                    _objective, 
                    x0, 
                    minimizer=_local_min, 
                    minimizer_kwargs={'options': {'maxiter': 100}},
                    kick=_kick,
                    niter=5,
                    T=0.5,
                    stepsize=0.02,
                    disp=False
                )
                new_pts = bh_res.x.reshape(N, 2)
                new_pts = _project_to_triangle(new_pts)
                score = _min_area(new_pts)
                _update_best(new_pts, score)
                x0 = new_pts.flatten()
            except Exception:
                pass

    # Final check
    if best_points is None:
        best_points = _initial_guess_lattice()
        
    return _project_to_triangle(best_points)
