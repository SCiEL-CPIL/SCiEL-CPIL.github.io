"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _min_triangle_area_with_grad(points):
    """
    Compute min triangle area and its subgradient w.r.t. points.
    Returns (min_area, grad) where grad has shape (N, 2).
    The gradient is the subgradient of -min_area (for minimization).
    """
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1  # (T, 2)
    d2 = p3 - p1  # (T, 2)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # (T,)
    areas = 0.5 * np.abs(cross)
    
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    
    # Subgradient: for the minimizing triangle (i,j,k),
    # d(area)/d(p_i) = -0.5 * (d1 x d2) * sign(cross) / |cross| * ... 
    # Actually, area = 0.5 * |cross|. 
    # If cross > 0: d(area)/d(p_k) = 0.5 * (d1_y, -d1_x) ... wait.
    # cross = (p2-p1)_x * (p3-p1)_y - (p2-p1)_y * (p3-p1)_x
    # d(cross)/d(p3_x) = (p2-p1)_y, d(cross)/d(p3_y) = -(p2-p1)_x
    # d(cross)/d(p2_x) = -(p3-p1)_y, d(cross)/d(p2_y) = (p3-p1)_x
    # d(cross)/d(p1_x) = (p3-p2)_y, d(cross)/d(p1_y) = -(p3-p2)_x
    
    # We want to MINIMIZE -min_area. So we want the gradient of -area.
    # grad(-area) = -grad(area).
    # For the minimizing triangle, if cross > 0:
    #   grad(area) w.r.t. p1 = -0.5 * (d(cross)/d(p1_x), d(cross)/d(p1_y)) ... 
    #   Actually area = 0.5 * cross (if cross>0).
    #   d(area)/d(p1) = -0.5 * ( (p3-p2)_y, -(p3-p2)_x )  -- let's re-derive carefully.
    
    # Let's just use the standard formula for the gradient of the cross product magnitude.
    # For triangle (a,b,c), area = 0.5 * |(b-a) x (c-a)|
    # If (b-a) x (c-a) > 0:
    #   d(area)/d(a) = -0.5 * ( (c-b)_y, -(c-b)_x )
    #   d(area)/d(b) =  0.5 * ( (c-a)_y, -(c-a)_x )  -- wait, this is getting messy.
    
    # Simpler approach: Use the fact that for the minimizing triangle (i,j,k),
    # we can compute the gradient numerically or use the analytic formula.
    # Let's use the analytic formula properly.
    
    # For triangle with vertices A, B, C:
    # area = 0.5 * |(B-A) x (C-A)|
    # Let u = B-A, v = C-A
    # cross = u_x * v_y - u_y * v_x
    # d(cross)/d(A) = d(cross)/d(A_x), d(cross)/d(A_y)
    # u = B-A => du/dA = -I
    # v = C-A => dv/dA = -I
    # d(cross)/dA_x = (du_x/dA_x)*v_y + u_x*(dv_y/dA_x) - (du_y/dA_x)*v_x - u_y*(dv_x/dA_x)
    # du_x/dA_x = -1, du_y/dA_x = 0
    # dv_y/dA_x = 0, dv_x/dA_x = -1
    # d(cross)/dA_x = (-1)*v_y + 0 - 0 - u_y*(-1) = -v_y + u_y = -(v_y - u_y) = -(C_y - A_y - B_y + A_y) = B_y - C_y
    # d(cross)/dA_y = (-1)*v_x*0 + u_x*(0) - (0)*v_x - u_y*(-1) ... let me redo this.
    
    # Actually, it's easier to just note that the gradient of the area w.r.t. vertex i
    # is 0.5 * (perpendicular to the opposite edge).
    
    # For now, let's just use a simple numerical gradient or skip the gradient and use Nelder-Mead.
    # The key insight is to use differential_evolution for global search, then refine with Nelder-Mead.
    
    return min_area, np.zeros_like(points)


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area with penalty."""
    points = flat.reshape(N, 2)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside the triangle
    eps = 1e-8
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -y - eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
    
    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _structured_inits():
    """Generate diverse structured initial configurations."""
    inits = []
    
    # Symmetric configuration near corners and center
    base_bary = np.array([
        [0.02, 0.02, 0.96],
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.45, 0.02, 0.53],
        [0.02, 0.45, 0.53],
        [0.53, 0.45, 0.02],
        [0.33, 0.33, 0.34],
        [0.25, 0.55, 0.20],
        [0.55, 0.20, 0.25],
        [0.20, 0.25, 0.55],
        [0.45, 0.25, 0.30],
    ])
    inits.append(_bary_to_cart(base_bary))
    
    # Three rows
    row_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.35, 0.05, 0.60],
        [0.65, 0.05, 0.30],
        [0.95, 0.05, 0.00],
        [0.15, 0.35, 0.50],
        [0.45, 0.35, 0.20],
        [0.75, 0.35, -0.10],  # Will be clipped
        [0.25, 0.65, 0.10],
        [0.55, 0.65, -0.20],
        [0.35, 0.55, 0.10],
        [0.65, 0.55, -0.20],
    ])
    row_bary[:, 2] = 1 - row_bary[:, 0] - row_bary[:, 1]
    inits.append(_bary_to_cart(row_bary))
    
    # Concentric pattern
    cent_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.40, 0.10, 0.50],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.50, 0.05, 0.45],
        [0.20, 0.55, 0.25],
        [0.55, 0.25, 0.20],
        [0.25, 0.20, 0.55],
        [0.45, 0.30, 0.25],
    ])
    inits.append(_bary_to_cart(cent_bary))
    
    # Grid-based
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    inits.append(_bary_to_cart(grid_bary))
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Global search with differential evolution
    if time.time() < deadline - 60:
        try:
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(flat):
                points = flat.reshape(N, 2)
                points = _clip_to_triangle(points)
                x = points[:, 0]
                y = points[:, 1]
                penalty = 0.0
                penalty += np.sum(np.maximum(0, -y)) * 100
                penalty += np.sum(np.maximum(0, y - S3 * x)) * 100
                penalty += np.sum(np.maximum(0, y - S3 * (1 - x))) * 100
                min_area = _min_triangle_area_fast(points)
                return -min_area + penalty
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                updating='immediate'
            )
            
            if de_result.fun < 0:
                points_de = _clip_to_triangle(de_result.x.reshape(N, 2))
                score_de = _min_triangle_area_fast(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_points = points_de.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    inits = _structured_inits()
    
    if best_points is not None:
        inits.append(best_points.copy())
    
    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(5):
        random_bary = rng.dirichlet([1, 1, 1], size=N)
        random_cart = _bary_to_cart(random_bary)
        inits.append(random_cart)
    
    for init_pts in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            flat_init = init_pts.flatten()
            
            res = minimize(
                _objective_cartesian,
                flat_init,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res.fun < 0:
                points_opt = _clip_to_triangle(res.x.reshape(N, 2))
                score_opt = _min_triangle_area_fast(points_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_points = points_opt.copy()
                    
        except Exception:
            continue
    
    # Phase 3: Final refinement if time permits
    if best_points is not None and time.time() < deadline - 5:
        try:
            flat_best = best_points.flatten()
            res_final = minimize(
                _objective_cartesian,
                flat_best,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-11,
                    'fatol': 1e-13,
                    'adaptive': True
                }
            )
            
            if res_final.fun < 0:
                points_final = _clip_to_triangle(res_final.x.reshape(N, 2))
                score_final = _min_triangle_area_fast(points_final)
                
                if score_final > best_score:
                    best_score = score_final
                    best_points = points_final.copy()
                    
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_points is None or not np.all(np.isfinite(best_points)):
        fallback_bary = np.array([
            [0.02, 0.02, 0.96],
            [0.96, 0.02, 0.02],
            [0.02, 0.96, 0.02],
            [0.45, 0.02, 0.53],
            [0.02, 0.45, 0.53],
            [0.53, 0.45, 0.02],
            [0.33, 0.33, 0.34],
            [0.25, 0.55, 0.20],
            [0.55, 0.20, 0.25],
            [0.20, 0.25, 0.55],
            [0.45, 0.25, 0.30],
        ])
        best_points = _bary_to_cart(fallback_bary)
    
    # Ensure validity
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
