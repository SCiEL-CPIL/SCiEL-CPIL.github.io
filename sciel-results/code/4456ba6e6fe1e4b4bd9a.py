"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    return _min_triangle_area_impl(pts)

# Pre-compute indices for N=14
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area_impl(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta=100.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -logsumexp(-beta * areas) / beta
    # For stability, subtract max
    m = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = (m - np.log(np.sum(np.exp(-beta * (areas - m)))))/beta
    return soft_min

def _soft_score(pts, beta=100.0):
    """Smooth approximation of score."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    soft_min = _soft_min_area(pts, beta)
    return soft_min / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    best_score = -np.inf
    best_pts = None
    
    # Helper to update best
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s
    
    # --- Fourier-based parameterization of a distorted regular 14-gon ---
    # Points are on a curve: r(theta) = 1 + sum_{k=1}^{K} (a_k cos(k*theta) + b_k sin(k*theta))
    # theta = 2*pi*i/N + sum_{k=1}^{K} (c_k cos(k*theta0_i) + d_k sin(k*theta0_i))
    # This gives 4*K parameters. We use K=3 for 12 parameters, then polish in full space.
    
    K_FOURIER = 3
    N_PARAMS = 4 * K_FOURIER  # a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2,d3
    
    def fourier_to_points(params):
        """Convert Fourier coefficients to 14 points."""
        a = params[0:K_FOURIER]
        b = params[K_FOURIER:2*K_FOURIER]
        c = params[2*K_FOURIER:3*K_FOURIER]
        d = params[3*K_FOURIER:4*K_FOURIER]
        
        theta0 = 2 * np.pi * np.arange(N) / N
        points = np.zeros((N, 2))
        
        for i in range(N):
            t0 = theta0[i]
            # Angular perturbation
            dtheta = 0.0
            for k in range(1, K_FOURIER + 1):
                dtheta += c[k-1] * np.cos(k * t0) + d[k-1] * np.sin(k * t0)
            theta = t0 + dtheta
            
            # Radial perturbation
            r = 1.0
            for k in range(1, K_FOURIER + 1):
                r += a[k-1] * np.cos(k * t0) + b[k-1] * np.sin(k * t0)
            
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
        
        return points
    
    def fourier_objective(params):
        pts = fourier_to_points(params)
        # Use exact score for Fourier optimization (it's fast enough)
        return -_score(pts)
    
    # Initial best from regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(regular_pts)
    
    # Optimize in Fourier space
    if time.time() < deadline - 30:
        # Start from zero Fourier coefficients (regular polygon)
        x0 = np.zeros(N_PARAMS)
        
        # Try multiple random initializations in Fourier space
        rng = np.random.RandomState(42)
        fourier_inits = [x0]
        for _ in range(10):
            fourier_inits.append(rng.normal(0, 0.1, size=N_PARAMS))
        
        bounds_f = [(-0.3, 0.3)] * N_PARAMS
        
        for fi, f_init in enumerate(fourier_inits):
            if time.time() > deadline - 30:
                break
            
            try:
                res = minimize(fourier_objective, f_init, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                pts = fourier_to_points(res.x)
                update_best(pts)
            except Exception:
                pass
        
        # Differential evolution in Fourier space
        if time.time() < deadline - 25:
            try:
                res_de = differential_evolution(
                    fourier_objective,
                    bounds_f,
                    seed=42,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    polish=True,
                    init='sobol'
                )
                pts = fourier_to_points(res_de.x)
                update_best(pts)
            except Exception:
                pass
    
    # --- Full-space local optimization from best Fourier result ---
    if best_pts is not None and time.time() < deadline - 10:
        def objective_full(x):
            return -_score(x.reshape(-1, 2))
        
        # Polish with Nelder-Mead
        try:
            res = minimize(objective_full, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            update_best(res.x.reshape(-1, 2))
        except Exception:
            pass
        
        # Powell
        if time.time() < deadline - 5:
            try:
                res_powell = minimize(objective_full, best_pts.flatten(), method='Powell',
                                     options={'maxiter': 5000, 'xtol': 1e-12, 'ftol': 1e-14})
                update_best(res_powell.x.reshape(-1, 2))
            except Exception:
                pass
    
    # --- Try additional ring-based initializations and optimize ---
    if time.time() < deadline - 15:
        # Two rings: 7 inner, 7 outer
        inner_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        outer_angles = inner_angles + (2 * np.pi / 14)
        inner_pts = np.column_stack([0.5 * np.cos(inner_angles), 0.5 * np.sin(inner_angles)])
        outer_pts = np.column_stack([1.0 * np.cos(outer_angles), 1.0 * np.sin(outer_angles)])
        two_ring_pts = np.vstack([inner_pts, outer_pts])
        
        def objective_full(x):
            return -_score(x.reshape(-1, 2))
        
        try:
            res = minimize(objective_full, two_ring_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            update_best(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    # --- Final polish with tighter tolerances ---
    if best_pts is not None and time.time() < deadline - 2:
        def objective_full(x):
            return -_score(x.reshape(-1, 2))
        
        try:
            res = minimize(objective_full, best_pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            update_best(res.x.reshape(-1, 2))
        except Exception:
            pass
    
    if best_pts is None:
        return regular_pts
        
    return best_pts
# EVOLVE-BLOCK-END
