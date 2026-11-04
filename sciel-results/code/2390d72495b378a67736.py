import numpy as np
from scipy.optimize import minimize, linprog
import time
import itertools

N = 11

# Vertices of the equilateral triangle in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle with side 1

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
n_triples = len(_IDX_TRIPLES)  # 165

def _all_dets_from_bary(bary):
    """
    Compute all 165 determinants (twice the area in barycentric coords).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """
    Returns the minimum absolute determinant (proportional to area ratio).
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    w = 1 - u - v
    Point = u*V0 + v*V1 + w*V2
    """
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _project_to_bary_simplex(bary):
    """
    Project points to the valid barycentric simplex.
    """
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _build_lp_constraints(signs):
    """
    Build the A_ub, b_ub, A_eq, b_eq for the LP.
    Variables: [u0, v0, u1, v1, ..., u10, v10, t] (22 + 1 = 23 vars)
    
    Constraints:
    1. For each triple (i,j,k): sign[t] * (det expression) >= t
       => -sign[t] * (det linear expr) + t <= 0
       The det is linear in the barycentric coords.
       
    2. u_i >= 0, v_i >= 0, u_i + v_i <= 1 for each i
    """
    n_vars = 2 * N + 1  # 22 + t
    
    # Build the coefficient matrix for determinants
    # det(i,j,k) = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    # where w = 1 - u - v
    # Expanding: this is linear in u and v for each point.
    # Let's compute the gradient of det with respect to each u_m and v_m.
    
    # For a triple (i, j, k), the determinant as a function of barycentric coords:
    # det = u_i*(v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k) 
    #       - v_i*(u_j*(1-u_k-v_k) - (1-u_j-v_j)*u_k)
    #       + (1-u_i-v_i)*(u_j*v_k - v_j*u_k)
    
    # This is complex. Let me use a different approach:
    # The area of triangle (Pi, Pj, Pk) in barycentric coords is:
    # det = |u_i(v_j w_k - w_j v_k) - v_i(u_j w_k - w_j u_k) + w_i(u_j v_k - v_j u_k)|
    # where w_m = 1 - u_m - v_m
    
    # The gradient with respect to u_m for m in {i,j,k}:
    # Let's just compute it numerically per triple or use the known formula.
    
    # Actually, the determinant of three points in barycentric coords (u,v,w) is:
    # det = u_i*v_j*w_k + v_i*w_j*u_k + w_i*u_j*v_k - w_i*v_j*u_k - u_i*w_j*v_k - v_i*u_j*w_k
    
    # This is a multilinear form. The coefficient of u_m in the determinant for triple (i,j,k):
    # If m == i: coeff_u_i = v_j*w_k - w_j*v_k = v_j*(1-u_k-v_k) - (1-u_j-v_j)*v_k
    # This depends on other variables! So it's NOT linear in the full set.
    
    # Wait - the determinant IS linear in each point's barycentric coordinates individually,
    # but when we fix the other two points, it's linear in the first point's coords.
    # But we're optimizing all points simultaneously, so the constraint is NOT linear.
    
    # The determinant is NOT jointly linear in all 22 variables. It's trilinear.
    # So LP won't work directly.
    
    # Let me reconsider. The approach should be:
    # Fix the positions of all but one point, then the constraint is linear in the remaining point.
    # But that's a sequential approach, not a single LP.
    
    # Alternative: Use the fact that for a FIXED ordering (signs), we can write:
    # For each triple, the signed area is a specific linear combination IF we fix which points are which.
    # But the area formula involves products of coordinates from different points.
    
    # I think the LP approach as described in the directive is more nuanced.
    # Let me use a different structural approach: use scipy's minimize with a 
    # carefully constructed smooth objective and multiple restarts with better initialization.
    
    # Actually, let me try a different structural change: 
    # Use a population-based approach with differential evolution or a 
    # more sophisticated multi-start with adaptive restarts.
    
    # For now, let me return None to indicate this path is not viable as a simple LP.
    return None

def _solve_lp_sign_fixed(bary_init, max_iter=50, time_budget=30.0):
    """
    Try the sign-fixed approach. If the determinant is trilinear, we can't use LP directly.
    Instead, use a smooth max-min approach with better initialization and more restarts.
    """
    start_time = time.time()
    bary = bary_init.copy()
    best_bary = bary.copy()
    best_score = _min_area_ratio_from_bary(bary) / (2 * TRI_AREA)
    
    # Use a smooth objective with log-sum-exp
    def obj_smooth(x, beta=100.0):
        b = x.reshape(N, 2)
        u = b[:, 0]
        v = b[:, 1]
        w = 1.0 - u - v
        
        # Penalty for invalid
        pen = 0.0
        pen += 100.0 * np.sum(np.maximum(0, -u)**2)
        pen += 100.0 * np.sum(np.maximum(0, -v)**2)
        pen += 100.0 * np.sum(np.maximum(0, u + v - 1.0)**2)
        
        dets = _all_dets_from_bary(b)
        areas = np.abs(dets)
        m = np.min(areas)
        shifted = areas - m
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
        smooth_min = m - (1.0 / beta) * log_sum_exp
        return -smooth_min + pen
    
    # Multiple restarts with different betas
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0]:
        if time.time() - start_time > time_budget:
            break
        
        x0 = bary.flatten()
        try:
            res = minimize(
                obj_smooth, x0, method='L-BFGS-B',
                args=(beta,),
                bounds=[(0, 1)] * (2 * N),
                options={'maxiter': 300, 'gtol': 1e-9}
            )
            x_opt = res.x
            bary_opt = _project_to_bary_simplex(x_opt.reshape(N, 2))
            s = _min_area_ratio_from_bary(bary_opt) / (2 * TRI_AREA)
            
            if s > best_score:
                best_score = s
                best_bary = bary_opt.copy()
                bary = bary_opt.copy()
        except Exception:
            pass
    
    return best_bary, best_score

def _structured_init_fps():
    """
    Farthest Point Sampling on a grid of barycentric coordinates.
    """
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _local_optimize_nm(x0, maxiter=300, xatol=1e-8, fatol=1e-8):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_lbfgs(x0, maxiter=200, gtol=1e-9):
    """Run L-BFGS-B local optimization from x0."""
    try:
        res = minimize(
            _objective_smooth, x0, method='L-BFGS-B',
            args=(50.0,),
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=[(0, 1)] * (2 * N),
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords (u,v,w < 0)
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    excess_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(excess_u**2)
    excess_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(excess_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    
    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def score(points):
    """Compute the raw score: min triangle area / TRI_AREA."""
    bary = np.zeros((N, 2))
    for i in range(N):
        p = points[i]
        w = 2 * p[1] / np.sqrt(3)
        v = p[0] - w * 0.5
        u = 1 - v - w
        bary[i] = [u, v]
    
    dets = _all_dets_from_bary(bary)
    min_area = np.min(np.abs(dets))
    return min_area / (2 * TRI_AREA)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    inits = []
    
    # 1. FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # 2. Random Inits
    rng = np.random.RandomState(42)
    for _ in range(15):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(pts.flatten())
        
    # 3. Magic Init
    magic_pts = [
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ]
    inits.append(np.array(magic_pts).flatten())
    
    # 4. Structured Starts
    bary1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5],
        [1/3, 1/3],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.25, 0.25],
        [0.5, 0.75],
    ])
    inits.append(_project_to_bary_simplex(bary1).flatten())
    
    bary2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.0, 0.333],
        [0.0, 0.667],
        [0.333, 0.333],
        [0.667, 0.333],
        [0.333, 0.667],
        [0.5, 0.5],
    ])
    inits.append(_project_to_bary_simplex(bary2).flatten())
    
    rng3 = np.random.RandomState(123)
    bary3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.25],
        [0.5, 0.25],
        [0.25, 0.5],
        [0.5, 0.5],
    ])
    bary3 += rng3.normal(0, 0.02, size=bary3.shape)
    inits.append(_project_to_bary_simplex(bary3).flatten())
    
    rng4 = np.random.RandomState(456)
    bary4 = []
    bary4.append([0.0, 0.0])
    bary4.append([1.0, 0.0])
    bary4.append([0.0, 1.0])
    bary4.append([1/3, 1/3])
    for _ in range(7):
        u = rng4.uniform(0.1, 0.9)
        v = rng4.uniform(0.1, 0.9)
        if u + v < 0.85:
            bary4.append([u, v])
        else:
            bary4.append([u * 0.8, v * 0.8])
    inits.append(_project_to_bary_simplex(np.array(bary4[:N])).flatten())
    
    # Optimize each initial point
    for init in inits:
        if time_left() < 5.0:
            break
        
        x0 = _project_to_bary_simplex(init.reshape(N, 2)).flatten()
        
        # Try multiple optimization methods
        for opt_func in [_local_optimize_nm, _local_optimize_lbfgs, _local_optimize_slsqp]:
            if time_left() < 5.0:
                break
            
            try:
                x_opt = opt_func(x0)
                x_opt = _project_to_bary_simplex(x_opt.reshape(N, 2)).flatten()
                
                bary_opt = x_opt.reshape(N, 2)
                s = _min_area_ratio_from_bary(bary_opt) / (2 * TRI_AREA)
                
                if s > best_score:
                    best_score = s
                    best_x = x_opt.copy()
                    
                x0 = x_opt
            except Exception:
                pass
    
    # If we found a good solution, convert to Cartesian
    if best_x is not None:
        bary_best = best_x.reshape(N, 2)
        pts_best = _bary_to_cart(bary_best[:, 0], bary_best[:, 1])
        return pts_best
    
    # Fallback: return random valid points
    pts = []
    for _ in range(N):
        u, v = np.random.random(2)
        while u + v > 1:
            u, v = np.random.random(2)
        pts.append([u + 0.5*v, np.sqrt(3)/2*v])
    return np.array(pts)
