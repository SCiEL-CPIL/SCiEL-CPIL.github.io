import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

# Triangle vertices in cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

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

def _normalize_bary(bary):
    """Ensure barycentric coords are non-negative and sum to 1."""
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
    cart = _bary_to_cart(_normalize_bary(bary))
    return _min_area_cart(cart)

def _init_hexagonal_bary():
    """
    Hexagonal lattice initialization in barycentric coordinates.
    This provides a structured starting point closer to optimal.
    """
    bary_pts = []
    
    # 3 vertices
    bary_pts.append(np.array([1.0, 0.0, 0.0]))
    bary_pts.append(np.array([0.0, 1.0, 0.0]))
    bary_pts.append(np.array([0.0, 0.0, 1.0]))
    
    # 3 edge midpoints
    bary_pts.append(np.array([0.5, 0.5, 0.0]))
    bary_pts.append(np.array([0.0, 0.5, 0.5]))
    bary_pts.append(np.array([0.5, 0.0, 0.5]))
    
    # Centroid
    bary_pts.append(np.array([1/3, 1/3, 1/3]))
    
    # 4 more points arranged in a small hexagonal pattern around centroid
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.12
    for angle in [0, np.pi/3, 2*np.pi/3, np.pi]:
        da = r * np.cos(angle)
        db = r * np.sin(angle)
        dc = -da - db
        pt = np.array([cx + da, cy + db, cz + dc])
        pt = _normalize_bary(pt.reshape(1, 3))[0]
        bary_pts.append(pt)
    
    return np.array(bary_pts)

def _init_perturbed(bary_base, rng, scale=0.05):
    """Create a perturbed version of a base barycentric configuration."""
    bary = bary_base.copy()
    perturbation = rng.normal(0, scale, size=bary.shape)
    bary = bary + perturbation
    return _normalize_bary(bary)

def _objective_alpha(flat_bary, alpha_target=0.0):
    """
    Objective that maximizes the minimum area.
    Returns negative of min area (to minimize).
    """
    bary = flat_bary.reshape(-1, 3)
    cart = _bary_to_cart(_normalize_bary(bary))
    areas = _all_areas_cart(cart)
    return -np.min(areas)

def _objective_smoothed(flat_bary, beta=200.0):
    """
    Smoothed objective using log-sum-exp for better gradient behavior.
    """
    bary = flat_bary.reshape(-1, 3)
    cart = _bary_to_cart(_normalize_bary(bary))
    areas = _all_areas_cart(cart)
    m = np.min(areas)
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _nelder_mead_optimize(bary_init, time_left_fn, max_iter=3000):
    """
    Nelder-Mead optimization on the hard objective.
    """
    start_time = time.time()
    
    def objective(flat_bary):
        if time.time() - start_time > time_left_fn() - 2.0:
            return -1.0  # Signal to stop
        return _objective_alpha(flat_bary)
    
    try:
        res = minimize(
            objective,
            bary_init.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-12,
                'fatol': 1e-14,
                'adaptive': True,
                'initial_simplex': None  # Let scipy choose
            }
        )
        opt_bary = _normalize_bary(res.x.reshape(-1, 3))
        return opt_bary, _min_area_bary(opt_bary)
    except Exception as e:
        return bary_init, _min_area_bary(bary_init)

def _lbfgsb_refine(bary_init, time_left_fn):
    """
    L-BFGS-B refinement with increasing beta for soft-min.
    """
    best_bary = bary_init.copy()
    best_score = _min_area_bary(best_bary)
    current = best_bary.copy()
    
    betas = [100.0, 300.0, 800.0]
    
    for beta in betas:
        if time_left_fn() < 10.0:
            break
        
        try:
            def obj(flat_bary):
                return -_objective_smoothed(flat_bary, beta)
            
            res = minimize(
                obj,
                current.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            opt_bary = _normalize_bary(res.x.reshape(-1, 3))
            score = _min_area_bary(opt_bary)
            
            if score > best_score:
                best_score = score
                best_bary = opt_bary.copy()
                current = opt_bary.copy()
        except Exception:
            pass
    
    return best_bary, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def time_left():
        return deadline - time.time()
    
    rng = np.random.default_rng(42)
    
    best_bary = None
    best_score = -1.0
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal structured init
    hex_bary = _init_hexagonal_bary()
    inits.append(hex_bary)
    
    # Perturbed versions of hexagonal
    for scale in [0.03, 0.08, 0.15]:
        pert = _init_perturbed(hex_bary, rng, scale)
        inits.append(pert)
    
    # Random inits
    for seed in range(5):
        bary_pts = []
        for _ in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            pt = np.array([1 - r1 - r2, r1, r2])
            bary_pts.append(pt)
        inits.append(np.array(bary_pts))
    
    # Evaluate all inits and sort
    scored_inits = []
    for init in inits:
        score = _min_area_bary(init)
        scored_inits.append((init, score))
    
    scored_inits.sort(key=lambda x: x[1], reverse=True)
    
    # Take top candidates and refine
    top_candidates = scored_inits[:5]
    
    for bary_init, init_score in top_candidates:
        if time_left() < 15.0:
            break
        
        # Stage 1: Nelder-Mead on hard objective
        try:
            nm_bary, nm_score = _nelder_mead_optimize(bary_init, time_left, max_iter=2000)
            if nm_score > best_score:
                best_score = nm_score
                best_bary = nm_bary.copy()
        except Exception:
            pass
        
        # Stage 2: L-BFGS-B refinement on the NM result (or init if NM failed)
        if time_left() > 10.0:
            base_for_lbfgs = best_bary if best_bary is not None and best_score > init_score else bary_init
            try:
                lbfgs_bary, lbfgs_score = _lbfgsb_refine(base_for_lbfgs, time_left)
                if lbfgs_score > best_score:
                    best_score = lbfgs_score
                    best_bary = lbfgs_bary.copy()
            except Exception:
                pass
    
    # Final high-precision Nelder-Mead polish
    if best_bary is not None and time_left() > 5.0:
        try:
            final_bary, final_score = _nelder_mead_optimize(best_bary, time_left, max_iter=5000)
            if final_score > best_score:
                best_score = final_score
                best_bary = final_bary.copy()
        except Exception:
            pass
    
    # Fallback
    if best_bary is None:
        best_bary = _init_hexagonal_bary()
        best_score = _min_area_bary(best_bary)
    
    # Convert to cartesian for output
    best_cart = _bary_to_cart(_normalize_bary(best_bary))
    
    return best_cart

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_cart(pts)}")
