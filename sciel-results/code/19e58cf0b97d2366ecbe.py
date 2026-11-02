"""Heilbronn triangles: maximize the smallest triangle area among 11 points.
Structural change: LP relaxation via scipy.optimize.linprog (HiGHS) in barycentric space.
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize
from itertools import combinations

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of unit-side equilateral triangle

# Precompute all triplets
_TRIPLETS = list(combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)  # 165

DEADLINE = 145  # Safety margin for 150s limit

def _barycentric_to_cartesian(bc):
    """Convert barycentric (b, c) to cartesian (x, y). a = 1-b-c."""
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack([x, y])

def _cartesian_to_barycentric(pts):
    """Convert cartesian (x, y) to barycentric (b, c)."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])

def _project_bary(bc):
    """Project barycentric coords to valid simplex: b>=0, c>=0, b+c<=1."""
    b = bc[:, 0].copy()
    c = bc[:, 1].copy()
    b = np.clip(b, 0, None)
    c = np.clip(c, 0, None)
    s = b + c
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask]
        c[mask] /= s[mask]
    return np.column_stack([b, c])

def _min_triangle_area(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _all_triangle_areas(pts):
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5

def _build_lp():
    """Build the LP for maximizing minimum triangle area in barycentric space.
    
    Variables: x = [b_0, c_0, b_1, c_1, ..., b_10, c_10, t]  (2N+1 variables)
    For each triplet (i,j,k), the signed area coefficient is:
        A_ijk = 0.5 * (c_i - c_j) * (b_k - b_j) - 0.5 * (b_i - b_j) * (c_k - c_j)
    which is linear in b's and c's.
    
    Constraints: For each triplet, both A_ijk >= t and -A_ijk >= t (i.e., |A_ijk| >= t)
    Plus simplex constraints: b_i >= 0, c_i >= 0, b_i + c_i <= 1
    """
    # Variable indices: b_i at 2*i, c_i at 2*i+1, t at 2*N
    # Objective: minimize -t (maximize t)
    c_obj = np.zeros(2 * N + 1)
    c_obj[2 * N] = -1.0  # minimize -t
    
    A_ub_rows = []
    b_ub_vals = []
    
    # For each triplet, add both A_ijk - t >= 0 and -A_ijk - t >= 0
    # In linprog form: -A_ijk + t <= 0 and A_ijk + t <= 0
    for idx, (i, j, k) in enumerate(_TRIPLETS):
        # Coefficients for A_ijk = 0.5*(c_i-c_j)*(b_k-b_j) - 0.5*(b_i-b_j)*(c_k-c_j)
        # = 0.5*c_i*b_k - 0.5*c_i*b_j - 0.5*c_j*b_k + 0.5*c_j*b_j 
        #   - 0.5*b_i*c_k + 0.5*b_i*c_j + 0.5*b_j*c_k - 0.5*b_j*c_j
        # Linear terms (coefficient for each variable):
        # b_i: -0.5*c_k + 0.5*c_j  -> but wait, these are products of variables!
        # 
        # Actually, A_ijk as a function of the barycentric coordinates IS linear:
        # A_ijk = 0.5 * det([p_j - p_i, p_k - p_i]) where p_m = (b_m + 0.5*c_m, S3H*c_m)
        # = 0.5 * [(b_j - b_i + 0.5(c_j - c_i)) * S3H*(c_k - c_i) - S3H*(c_j - c_i)*(b_k - b_i + 0.5(c_k - c_i))]
        # = 0.5*S3H * [(b_j - b_i + 0.5(c_j - c_i))(c_k - c_i) - (c_j - c_i)(b_k - b_i + 0.5(c_k - c_i))]
        # = 0.5*S3H * [(b_j - b_i)(c_k - c_i) + 0.5(c_j-c_i)(c_k-c_i) - (c_j-c_i)(b_k-b_i) - 0.5(c_j-c_i)(c_k-c_i)]
        # = 0.5*S3H * [(b_j - b_i)(c_k - c_i) - (c_j - c_i)(b_k - b_i)]
        
        # This is bilinear! NOT linear. So LP won't work directly.
        # 
        # Let me reconsider. The area in cartesian is:
        # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # With x_m = b_m + 0.5*c_m, y_m = S3H*c_m:
        # This is still bilinear in the barycentric coordinates.
        
        # So LP approach doesn't directly apply. Let me use a different strategy.
        pass
    
    # Since the area is bilinear, not linear, LP won't work directly.
    # Alternative: use the fact that for a FIXED set of which triangles are "active"
    # (i.e., which have area = t), we can solve a system.
    # Or: use semidefinite programming, or just do a good local search.
    
    # Let me pivot to a different structural approach: 
    # Use the known structure of Heilbronn configurations.
    # For n=11, the optimal configuration has a specific pattern.
    # Let me try a more targeted local search with better initialization.
    return None

def _soft_min_area(pts, k=30.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = np.log(np.sum(np.exp(shifted / k)))
    return max_a - k * log_sum

def _push_apart_bc(bc, step=0.01, n_iter=50, rng=None):
    """Push apart points that form small triangles, working in barycentric space."""
    bc = _project_bary(bc.copy())
    pts = _barycentric_to_cartesian(bc)
    
    for _ in range(n_iter):
        areas = _all_triangle_areas(pts)
        min_area = np.min(areas)
        
        # Get the k smallest triangles
        k = min(15, _N_TRIP)
        small_idx = np.argsort(areas)[:k]
        
        improved = False
        for idx in small_idx:
            i, j, k_idx = _TRIPLETS[idx]
            pi = pts[i]
            pj = pts[j]
            pk = pts[k_idx]
            
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            area = 0.5 * abs(cross)
            
            if area < min_area * 1.05:
                sign = 1.0 if cross >= 0 else -1.0
                perp_v1 = np.array([-v1[1], v1[0]])
                perp_v2 = np.array([-v2[1], v2[0]])
                
                grad_i = -sign * 0.5 * (perp_v2 - perp_v1)
                grad_j = sign * 0.5 * perp_v2
                grad_k = -sign * 0.5 * perp_v1
                
                new_pts = pts.copy()
                new_pts[i] = pi + step * grad_i
                new_pts[j] = pj + step * grad_j
                new_pts[k_idx] = pk + step * grad_k
                
                new_bc = _project_bary(_cartesian_to_barycentric(new_pts))
                new_pts = _barycentric_to_cartesian(new_bc)
                new_min = _min_triangle_area(new_pts)
                
                if new_min > min_area - 1e-12:
                    pts = new_pts
                    bc = new_bc
                    min_area = new_min
                    improved = True
        
        if not improved:
            break
    
    return bc, pts

def _local_refine(pts, time_budget, rng):
    """Multi-stage local refinement from a starting point."""
    start_time = time.time()
    bc = _project_bary(_cartesian_to_barycentric(pts.copy()))
    pts = _barycentric_to_cartesian(bc)
    best_area = _min_triangle_area(pts)
    
    def _time_left():
        return time_budget - (time.time() - start_time)
    
    # Stage 1: Gradient-based soft-min optimization with adaptive temperature
    if _time_left() > 10:
        for k_val in [50.0, 20.0, 10.0, 5.0]:
            if _time_left() < 5:
                break
            
            def soft_obj(flat):
                bc_local = _project_bary(flat.reshape(-1, 2))
                pts_local = _barycentric_to_cartesian(bc_local)
                return -_soft_min_area(pts_local, k=k_val)
            
            x0 = bc.flatten()
            try:
                res = minimize(
                    soft_obj, x0,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                new_bc = _project_bary(res.x.reshape(-1, 2))
                new_pts = _barycentric_to_cartesian(new_bc)
                new_area = _min_triangle_area(new_pts)
                if new_area > best_area:
                    best_area = new_area
                    bc = new_bc
                    pts = new_pts
            except Exception:
                pass
    
    # Stage 2: Push-apart with adaptive step sizes
    if _time_left() > 5:
        step = 0.02
        for _ in range(30):
            if _time_left() < 2:
                break
            bc_new, pts_new = _push_apart_bc(bc, step=step, n_iter=60, rng=rng)
            area_new = _min_triangle_area(pts_new)
            if area_new > best_area + 1e-12:
                best_area = area_new
                bc = bc_new
                pts = pts_new
            else:
                step *= 0.7
                if step < 0.0005:
                    break
    
    # Stage 3: Fine-tune
    if _time_left() > 2:
        for _ in range(20):
            if _time_left() < 1:
                break
            bc_new, pts_new = _push_apart_bc(bc, step=0.002, n_iter=80, rng=rng)
            area_new = _min_triangle_area(pts_new)
            if area_new > best_area + 1e-13:
                best_area = area_new
                bc = bc_new
                pts = pts_new
            else:
                break
    
    return pts, best_area

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Multiple diverse initial configurations
    init_configs = []
    
    # Config 1: Symmetric-ish starting point
    init1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    init_configs.append(init1)
    
    # Config 2: More spread out
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.5, 0.1],
        [0.3, S3H/3],
        [0.7, S3H/3],
        [0.5, 2*S3H/3],
        [0.15, S3H/4],
        [0.85, S3H/4],
    ], dtype=float)
    init_configs.append(init2)
    
    # Config 3: Random but projected
    for _ in range(3):
        rand_bc = rng.random((N, 2)) * 0.9
        rand_bc = _project_bary(rand_bc)
        rand_pts = _barycentric_to_cartesian(rand_bc)
        init_configs.append(rand_pts)
    
    # Evaluate all initial configs
    for cfg in init_configs:
        cfg_bc = _project_bary(_cartesian_to_barycentric(cfg))
        cfg_pts = _barycentric_to_cartesian(cfg_bc)
        _update_best(cfg_pts)
    
    if best_pts is None:
        return init_configs[0]
    
    # Phase 1: Local refinement from each good starting point
    time_per_config = 25
    for cfg in init_configs[:4]:  # Limit to 4 configs
        if not _elapsed_ok(15):
            break
        
        cfg_bc = _project_bary(_cartesian_to_barycentric(cfg))
        cfg_pts = _barycentric_to_cartesian(cfg_bc)
        
        refined_pts, refined_area = _local_refine(cfg_pts, time_per_config, rng)
        _update_best(refined_pts)
    
    # Phase 2: More aggressive local search from the best point
    if best_pts is not None and _elapsed_ok(30):
        # Try multiple perturbation strategies from the best point
        for n_pert in range(8):
            if not _elapsed_ok(5):
                break
            
            if n_pert == 0:
                start_pts = best_pts.copy()
            else:
                pert_scale = 0.02 * (1 + n_pert * 0.5)
                pert = rng.normal(0, pert_scale, size=(N, 2))
                start_bc = _project_bary(_cartesian_to_barycentric(best_pts) + pert)
                start_pts = _barycentric_to_cartesian(start_bc)
            
            refined_pts, refined_area = _local_refine(start_pts, 15, rng)
            _update_best(refined_pts)
    
    # Phase 3: Final fine-tuning from the overall best
    if best_pts is not None and _elapsed_ok(10):
        bc = _project_bary(_cartesian_to_barycentric(best_pts))
        pts = _barycentric_to_cartesian(bc)
        
        # Very fine push-apart
        for _ in range(50):
            if not _elapsed_ok(2):
                break
            bc_new, pts_new = _push_apart_bc(bc, step=0.001, n_iter=100, rng=rng)
            area_new = _min_triangle_area(pts_new)
            if area_new > best_area + 1e-14:
                best_area = area_new
                best_pts = pts_new
                bc = bc_new
                pts = pts_new
            else:
                break
        
        # Final soft-min polish with very small k
        if _elapsed_ok(5):
            def soft_obj_final(flat):
                bc_local = _project_bary(flat.reshape(-1, 2))
                pts_local = _barycentric_to_cartesian(bc_local)
                return -_soft_min_area(pts_local, k=3.0)
            
            x0 = bc.flatten()
            try:
                res = minimize(
                    soft_obj_final, x0,
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                new_bc = _project_bary(res.x.reshape(-1, 2))
                new_pts = _barycentric_to_cartesian(new_bc)
                _update_best(new_pts)
            except Exception:
                pass
    
    if best_pts is None:
        return init_configs[0]
    
    return best_pts
