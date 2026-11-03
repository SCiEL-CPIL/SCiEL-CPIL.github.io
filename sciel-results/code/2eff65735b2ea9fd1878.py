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


def _st_to_bary(st):
    """
    Convert (s, t) parameters in [0,1] to barycentric coordinates (u, v, w).
    u = (1-s)(1-t)
    v = s(1-t)
    w = t
    """
    s = st[:, 0]
    t = st[:, 1]
    u = (1 - s) * (1 - t)
    v = s * (1 - t)
    w = t
    return np.column_stack([u, v, w])


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = bary[:, 2]
    x = v + 0.5 * w
    y = H * w
    return np.column_stack([x, y])


def _st_to_cart(st):
    """Convert (s, t) parameters directly to Cartesian."""
    bary = _st_to_bary(st)
    return _bary_to_cart(bary)


def _objective_st(flat):
    """
    Objective function for optimization in (s, t) space.
    Minimizes negative min triangle area.
    No penalties needed as (s, t) in [0,1] maps to inside triangle.
    """
    st = flat.reshape(N, 2)
    # Ensure bounds are respected (DE and NM should handle this, but just in case)
    st = np.clip(st, 0, 1)
    points = _st_to_cart(st)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _generate_symmetric_inits():
    """
    Generate initial configurations in (s, t) space.
    We generate Cartesian points, then invert the mapping to get (s, t).
    Inversion:
    x = v + 0.5w
    y = H*w  => w = y/H
    v = x - 0.5w
    u = 1 - v - w
    
    From u, v, w to s, t:
    w = t => t = w
    v = s(1-t) => s = v / (1-t) = v / (1-w)
    u = (1-s)(1-t) ... check consistency.
    """
    inits_st = []
    
    # Helper to convert Cart to ST
    def cart_to_st(points):
        points = np.clip(points, 0, 1) # Rough clip, will refine
        w = points[:, 1] / H
        # w must be <= 1
        w = np.clip(w, 0, 1)
        v = points[:, 0] - 0.5 * w
        # v must be >= 0
        v = np.clip(v, 0, 1)
        u = 1 - v - w
        # u must be >= 0
        # If u < 0, point is outside. We need to project.
        # For initialization, we'll just use valid points.
        
        t = w
        # s = v / (1 - t)
        denom = 1 - t
        # Handle t=1 case (apex)
        s = np.zeros_like(v)
        mask = denom > 1e-9
        s[mask] = v[mask] / denom[mask]
        # If t is close to 1, v should be 0. s can be anything, set to 0.5.
        s[~mask] = 0.5
        
        s = np.clip(s, 0, 1)
        t = np.clip(t, 0, 1)
        
        return np.column_stack([s, t])

    # Config 1: Symmetric grid
    # Points roughly evenly distributed
    pts = np.array([
        [0.5, 0.05],
        [0.2, 0.1], [0.8, 0.1],
        [0.5, 0.2],
        [0.3, 0.3], [0.7, 0.3],
        [0.5, 0.4],
        [0.4, 0.5], [0.6, 0.5],
        [0.5, 0.6],
        [0.5, 0.7]
    ])
    inits_st.append(cart_to_st(pts))
    
    # Config 2: Clustered near corners
    pts = np.array([
        [0.05, 0.05], [0.95, 0.05], [0.5, H - 0.05],
        [0.3, 0.1], [0.7, 0.1], [0.5, 0.2],
        [0.2, 0.3], [0.8, 0.3], [0.5, 0.4],
        [0.4, 0.5], [0.6, 0.5]
    ])
    inits_st.append(cart_to_st(pts))
    
    # Config 3: Random but valid
    rng = np.random.RandomState(42)
    for _ in range(5):
        # Sample random barycentric coords
        bary = rng.dirichlet([1, 1, 1], size=N)
        pts = _bary_to_cart(bary)
        inits_st.append(cart_to_st(pts))
        
    return inits_st


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_st = None
    best_score = -np.inf
    
    # Phase 1: Global search with Differential Evolution in (s, t) space
    # Bounds are [0, 1] for each s and t
    bounds = [(0, 1), (0, 1)] * N
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective_st,
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,
                updating='immediate'
            )
            
            if de_result.fun < 0:
                st_de = de_result.x.reshape(N, 2)
                points_de = _st_to_cart(st_de)
                score_de = _min_triangle_area_fast(points_de)
                if score_de > best_score:
                    best_score = score_de
                    best_st = st_de.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    inits = _generate_symmetric_inits()
    
    if best_st is not None:
        inits.append(best_st.copy())
    
    for init_st in inits:
        if time.time() > deadline - 15:
            break
        
        try:
            flat_init = init_st.flatten()
            
            res = minimize(
                _objective_st,
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
                st_opt = res.x.reshape(N, 2)
                # Clip to [0,1] just in case NM steps out
                st_opt = np.clip(st_opt, 0, 1)
                points_opt = _st_to_cart(st_opt)
                score_opt = _min_triangle_area_fast(points_opt)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_st = st_opt.copy()
                    
        except Exception:
            continue
    
    # Phase 3: Final refinement if time permits
    if best_st is not None and time.time() < deadline - 5:
        try:
            flat_best = best_st.flatten()
            res_final = minimize(
                _objective_st,
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
                st_final = res_final.x.reshape(N, 2)
                st_final = np.clip(st_final, 0, 1)
                points_final = _st_to_cart(st_final)
                score_final = _min_triangle_area_fast(points_final)
                
                if score_final > best_score:
                    best_score = score_final
                    best_st = st_final.copy()
                    
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_st is None or not np.all(np.isfinite(best_st)):
        # Use a simple symmetric configuration
        # 1 center, 3 on medians at 1/3, 3 on medians at 2/3, 1... wait, 11 points.
        # Just use the first init
        best_st = inits[0] if inits else np.full((N, 2), 0.5)
    
    # Convert best (s, t) to Cartesian points
    best_points = _st_to_cart(best_st)
    
    # Ensure validity (should be inside by construction, but clip just in case)
    # The mapping guarantees inside, but floating point might cause slight issues?
    # The formula x = v + 0.5w, y = H*w with u,v,w >= 0, sum=1 guarantees inside.
    # So no clipping needed, but we can do a safety check.
    
    return best_points
# EVOLVE-BLOCK-END
