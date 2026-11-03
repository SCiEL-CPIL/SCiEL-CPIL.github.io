# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
SQRT3 = np.sqrt(3.0)
TRI_AREA = 0.5 * SQRT3 / 2.0  # Area of equilateral triangle with side 1

# Precompute all triplets
_indices = list(itertools.combinations(range(N), 3))
_IDX_I = np.array([t[0] for t in _indices])
_IDX_J = np.array([t[1] for t in _indices])
_IDX_K = np.array([t[2] for t in _indices])
N_TRIP = len(_indices)

def _bary_to_cart(u, v):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    Point = u*V0 + v*V1 + w*V2 where w = 1-u-v
    V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = v + 0.5*w = v + 0.5*(1-u-v) = 0.5 + 0.5*v - 0.5*u
    y = w * sqrt(3)/2 = (1-u-v)*sqrt(3)/2
    """
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * SQRT3 / 2.0
    return np.column_stack((x, y))

def _cart_to_bary(points):
    """Convert Cartesian points to barycentric (u, v)."""
    x = points[:, 0]
    y = points[:, 1]
    # y = (1-u-v)*sqrt(3)/2 => 1-u-v = 2y/sqrt(3) => u+v = 1 - 2y/sqrt(3)
    # x = v + 0.5*(1-u-v) = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # => 2x - 1 = 0.5v - 0.5u => v - u = 2*(2x-1) = 4x - 2
    # From u+v = 1 - 2y/sqrt(3) and v-u = 4x-2:
    # v = (1 - 2y/sqrt(3) + 4x - 2)/2 = (4x - 1 - 2y/sqrt(3))/2 = 2x - 0.5 - y/sqrt(3)
    # u = 1 - 2y/sqrt(3) - v = 1 - 2y/sqrt(3) - 2x + 0.5 + y/sqrt(3) = 1.5 - 2x - y/sqrt(3)
    u = 1.5 - 2.0 * x - y / SQRT3
    v = 2.0 * x - 0.5 - y / SQRT3
    return np.column_stack((u, v))

def _all_areas_from_bary(u, v):
    """
    Compute all triangle areas from barycentric coordinates.
    u, v: (N, 2) arrays of barycentric coords
    Returns: array of N_TRIP areas
    """
    # Convert to Cartesian for area computation
    pts = _bary_to_cart(u, v)
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    vec1 = pj - pi
    vec2 = pk - pi
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    return 0.5 * np.abs(cross)

def _all_areas_from_cart(pts):
    """Compute all triangle areas from Cartesian points (N,2)."""
    pi = pts[_IDX_I]
    pj = pts[_IDX_J]
    pk = pts[_IDX_K]
    vec1 = pj - pi
    vec2 = pk - pi
    cross = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
    return 0.5 * np.abs(cross)

def _softmin_neg(areas, beta):
    """
    Return -softmin(areas) where softmin approximates the minimum.
    softmin(A) = -1/beta * log(sum(exp(-beta*A)))
    We want to maximize min(A), so minimize -softmin(A).
    """
    x = -beta * areas
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return 0.0
    log_sum_exp = x_max + np.log(np.sum(np.exp(x - x_max)))
    soft_min_val = -1.0 / beta * log_sum_exp
    return -soft_min_val

def _objective_bary(x_flat, beta):
    """
    Objective for barycentric optimization.
    x_flat: (2*N,) array of (u1,v1,u2,v2,...)
    Constraints: u >= 0, v >= 0, u+v <= 1 for each point.
    """
    u = x_flat[0::2]
    v = x_flat[1::2]
    # Penalty for constraint violation
    w = 1.0 - u - v
    penalty = 0.0
    if np.any(u < -1e-10) or np.any(v < -1e-10) or np.any(w < -1e-10):
        # Heavy penalty
        penalty = 100.0 * (np.sum(np.maximum(0, -u)**2) + 
                          np.sum(np.maximum(0, -v)**2) + 
                          np.sum(np.maximum(0, -w)**2))
    areas = _all_areas_from_bary(u, v)
    return _softmin_neg(areas, beta) + penalty

def _hard_min_area_from_bary(u, v):
    """Compute the true minimum triangle area from barycentric coords."""
    areas = _all_areas_from_bary(u, v)
    return np.min(areas)

def _generate_grid_init(n_per_edge):
    """
    Generate a structured grid initialization in barycentric coordinates.
    For N=11, use a pattern that distributes points well.
    """
    # Use a combination of structured and random
    rng = np.random.RandomState(12345)
    
    # Start with a coarse grid
    # For 11 points, use a pattern like:
    # Corner points, edge midpoints, and interior points
    u = np.zeros(N)
    v = np.zeros(N)
    
    # Place some points at strategic locations
    # Corners (slightly inside)
    eps = 0.05
    u[0], v[0] = eps, eps          # Near V0
    u[1], v[1] = eps, 1-2*eps      # Near V1  
    u[2], v[2] = 1-2*eps, eps      # Near V2
    
    # Edge midpoints (slightly inside)
    u[3], v[3] = 0.5, 0.02         # Mid of V0-V1
    u[4], v[4] = 0.02, 0.5         # Mid of V0-V2
    u[5], v[5] = 0.5, 0.5          # Mid of V1-V2 (center-ish)
    
    # Interior points
    for i in range(6, N):
        # Random interior points
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        u[i] = r1 + 0.05
        v[i] = r2 + 0.05
        # Ensure inside
        if u[i] + v[i] > 0.9:
            scale = 0.9 / (u[i] + v[i])
            u[i] *= scale
            v[i] *= scale
    
    return np.column_stack((u, v))

def _generate_random_init(rng):
    """Generate random initialization inside the triangle."""
    u = np.zeros(N)
    v = np.zeros(N)
    for i in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        u[i] = r1 * 0.8 + 0.1
        v[i] = r2 * 0.8 + 0.1
        if u[i] + v[i] > 0.85:
            scale = 0.85 / (u[i] + v[i])
            u[i] *= scale
            v[i] *= scale
    return np.column_stack((u, v))

def _generate_fps_init(rng, n_points=N):
    """
    Farthest Point Sampling style initialization.
    Start with a random point, then iteratively add the point farthest
    from all existing points (in terms of minimum triangle area contribution).
    """
    u = np.zeros(N)
    v = np.zeros(N)
    
    # Start with 3 corners (slightly inside)
    eps = 0.03
    u[0], v[0] = eps, eps
    u[1], v[1] = eps, 1 - 2*eps
    u[2], v[2] = 1 - 2*eps, eps
    
    pts = _bary_to_cart(u[:3], v[:3])
    
    for i in range(3, N):
        # Try many random candidates, pick the one that maximizes min area
        best_candidate = None
        best_min_area = -1
        
        n_candidates = 200
        for _ in range(n_candidates):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            cu = r1
            cv = r2
            if cu + cv > 0.95:
                scale = 0.95 / (cu + cv)
                cu *= scale
                cv *= scale
            
            # Add candidate and compute min area
            u_trial = np.concatenate([u[:i], [cu]])
            v_trial = np.concatenate([v[:i], [cv]])
            pts_trial = _bary_to_cart(u_trial, v_trial)
            areas = _all_areas_from_cart(pts_trial)
            min_a = np.min(areas)
            
            if min_a > best_min_area:
                best_min_area = min_a
                best_candidate = (cu, cv)
        
        u[i], v[i] = best_candidate
        pts = _bary_to_cart(u[:i+1], v[:i+1])
    
    return np.column_stack((u, v))

def _local_optimize(u0, v0, beta, method='L-BFGS-B', maxiter=200):
    """
    Perform local optimization from initial barycentric coords.
    Returns (u_opt, v_opt, score).
    """
    x0 = np.concatenate([u0, v0])
    
    # Bounds: u >= 0, v >= 0, u+v <= 1
    # For L-BFGS-B, we can use bounds on u and v
    # u in [0, 1], v in [0, 1-u] but that's coupled
    # Use bounds u in [0,1], v in [0,1] and rely on penalty
    bounds = [(0, 1)] * (2 * N)
    
    try:
        if method == 'L-BFGS-B':
            res = minimize(
                _objective_bary, x0, args=(beta,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        elif method == 'Nelder-Mead':
            res = minimize(
                _objective_bary, x0, args=(beta,),
                method='Nelder-Mead',
                options={'maxiter': maxiter * 3, 'xatol': 1e-10, 'fatol': 1e-12}
            )
        elif method == 'SLSQP':
            # SLSQP with explicit constraints
            constraints = []
            for i in range(N):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
            res = minimize(
                _objective_bary, x0, args=(beta,),
                method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
        else:
            res = minimize(
                _objective_bary, x0, args=(beta,),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter}
            )
        
        x_opt = res.x
        u_opt = x_opt[0::2]
        v_opt = x_opt[1::2]
        
        # Clamp to valid region
        u_opt = np.clip(u_opt, 0, 1)
        v_opt = np.clip(v_opt, 0, 1)
        w_opt = 1.0 - u_opt - v_opt
        mask = w_opt < 0
        if np.any(mask):
            # Scale back
            for idx in np.where(mask)[0]:
                s = (u_opt[idx] + v_opt[idx])
                if s > 0:
                    u_opt[idx] *= 1.0 / s
                    v_opt[idx] *= 1.0 / s
                else:
                    u_opt[idx] = 0.33
                    v_opt[idx] = 0.33
        
        score = _hard_min_area_from_bary(u_opt, v_opt)
        return u_opt, v_opt, score, res
    except Exception:
        return u0, v0, _hard_min_area_from_bary(u0, v0), None

def _anneal_refine(u0, v0, betas, methods, maxiter_each=100):
    """
    Multi-stage annealing: start with low beta (smooth), increase to high beta (sharp).
    """
    u_curr, v_curr = u0.copy(), v0.copy()
    best_score = _hard_min_area_from_bary(u_curr, v_curr)
    
    for beta in betas:
        for method in methods:
            u_new, v_new, score, _ = _local_optimize(u_curr, v_curr, beta, method, maxiter_each)
            if score > best_score:
                best_score = score
                u_curr, v_curr = u_new, v_new
    
    return u_curr, v_curr, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer)
    
    rng = np.random.RandomState(42)
    
    best_score = -float('inf')
    best_u = None
    best_v = None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Structured grid
    inits.append(_generate_grid_init(3))
    
    # 2. FPS-style
    inits.append(_generate_fps_init(rng))
    
    # 3. Random initializations
    for _ in range(5):
        inits.append(_generate_random_init(rng))
    
    # 4. Fibonacci-like spiral in barycentric space
    u_fib = np.zeros(N)
    v_fib = np.zeros(N)
    for i in range(N):
        angle = i * 2.399963  # Golden angle
        r = 0.3 * np.sqrt((i + 1) / N)
        # Map to barycentric: center is (1/3, 1/3)
        u_fib[i] = 1/3 + r * np.cos(angle) * 0.5
        v_fib[i] = 1/3 + r * np.sin(angle) * 0.5
        if u_fib[i] < 0:
            u_fib[i] = 0.05
        if v_fib[i] < 0:
            v_fib[i] = 0.05
        if u_fib[i] + v_fib[i] > 0.9:
            s = 0.9 / (u_fib[i] + v_fib[i])
            u_fib[i] *= s
            v_fib[i] *= s
    inits.append(np.column_stack((u_fib, v_fib)))
    
    # Annealing schedule
    betas = [5.0, 15.0, 50.0, 150.0, 500.0, 2000.0]
    methods = ['L-BFGS-B', 'Nelder-Mead']
    
    for init_idx, init_bary in enumerate(inits):
        if time.time() > deadline:
            break
        
        u_init = init_bary[:, 0]
        v_init = init_bary[:, 1]
        
        # Check validity
        w_init = 1.0 - u_init - v_init
        if np.any(u_init < -1e-6) or np.any(v_init < -1e-6) or np.any(w_init < -1e-6):
            # Clamp
            u_init = np.clip(u_init, 0, 1)
            v_init = np.clip(v_init, 0, 1)
            w_check = 1.0 - u_init - v_init
            mask = w_check < 0
            for idx in np.where(mask)[0]:
                s = u_init[idx] + v_init[idx]
                if s > 0:
                    u_init[idx] /= s
                    v_init[idx] /= s
                else:
                    u_init[idx] = 0.33
                    v_init[idx] = 0.33
        
        # Evaluate initial score
        init_score = _hard_min_area_from_bary(u_init, v_init)
        if init_score > best_score:
            best_score = init_score
            best_u = u_init.copy()
            best_v = v_init.copy()
        
        # Anneal and refine
        u_opt, v_opt, opt_score = _anneal_refine(u_init, v_init, betas, methods, maxiter_each=80)
        
        if opt_score > best_score:
            best_score = opt_score
            best_u = u_opt.copy()
            best_v = v_opt.copy()
        
        # If we still have time, do a final sharp refinement
        if time.time() < deadline - 30:
            u_final, v_final, final_score = _local_optimize(
                u_opt, v_opt, beta=5000.0, method='Nelder-Mead', maxiter=300
            )
            if final_score > best_score:
                best_score = final_score
                best_u = u_final.copy()
                best_v = v_final.copy()
    
    # If we still have time, try differential evolution for global search
    if time.time() < deadline - 40:
        try:
            bounds_de = [(0.0, 1.0)] * (2 * N)
            
            def _de_objective(x_flat):
                u = x_flat[0::2]
                v = x_flat[1::2]
                w = 1.0 - u - v
                if np.any(w < -1e-8):
                    return 1.0  # Large penalty
                areas = _all_areas_from_bary(u, v)
                return -np.min(areas)
            
            res_de = differential_evolution(
                _de_objective, bounds_de,
                seed=42, maxiter=50, tol=1e-10,
                popsize=15, mutation=(0.3, 1.0), recombination=0.8
            )
            
            if res_de.success:
                u_de = res_de.x[0::2]
                v_de = res_de.x[1::2]
                w_de = 1.0 - u_de - v_de
                if np.all(w_de >= -1e-8):
                    de_score = _hard_min_area_from_bary(u_de, v_de)
                    if de_score > best_score:
                        best_score = de_score
                        best_u = u_de.copy()
                        best_v = v_de.copy()
        except Exception:
            pass
    
    # Final: if we have time, do one more round of local refinement on best
    if best_u is not None and time.time() < deadline - 10:
        u_final, v_final, final_score = _local_optimize(
            best_u, best_v, beta=10000.0, method='Nelder-Mead', maxiter=500
        )
        if final_score > best_score:
            best_score = final_score
            best_u = u_final.copy()
            best_v = v_final.copy()
    
    # Fallback
    if best_u is None:
        u_fb = np.ones(N) / 3.0
        v_fb = np.ones(N) / 3.0
        # Perturb
        for i in range(N):
            u_fb[i] += rng.randn() * 0.05
            v_fb[i] += rng.randn() * 0.05
        u_fb = np.clip(u_fb, 0.01, 0.98)
        v_fb = np.clip(v_fb, 0.01, 0.98)
        w_fb = 1.0 - u_fb - v_fb
        mask = w_fb < 0
        for idx in np.where(mask)[0]:
            s = u_fb[idx] + v_fb[idx]
            if s > 0:
                u_fb[idx] *= 0.9 / s
                v_fb[idx] *= 0.9 / s
            else:
                u_fb[idx] = 0.33
                v_fb[idx] = 0.33
        best_u, best_v = u_fb, v_fb
    
    # Convert to Cartesian
    points = _bary_to_cart(best_u, best_v)
    
    return points
# EVOLVE-BLOCK-END
