"""Circle packing in a unit circle - 50 equal circles."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(pts):
    """Compute the packing score (minimum radius)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    # Container constraint
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    if np.any(dists_center >= 1.0):
        return 0.0
    r_container = 1.0 - np.max(dists_center)
    if r_container <= 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    return min(r_container, r_pair)

def _soft_min_score(pts, beta_pair=8.0, beta_final=20.0):
    """Soft approximation of the packing score using log-sum-exp."""
    if pts is None or len(pts) != N:
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    pts = np.asarray(pts, dtype=np.float64)
    # Container constraint
    dists_center = np.sqrt(np.sum(pts**2, axis=1))
    container_violation = np.maximum(0, dists_center - 1.0)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = dists[dists < np.inf]
    if len(min_dists) == 0:
        return -1e6
    # Soft min of pairwise distances
    m = np.min(min_dists)
    exp_terms = np.exp(-beta_pair * (min_dists - m))
    soft_min_pair = m - np.log(np.sum(exp_terms)) / beta_pair
    # Soft min with container
    r1 = soft_min_pair / 2.0
    r2 = 1.0 - np.max(dists_center)
    m_final = min(r1, r2)
    exp_terms_final = np.exp(-beta_final * np.array([r1, r2]) + beta_final * m_final)
    soft_min_final = m_final - np.log(np.sum(exp_terms_final)) / beta_final
    # Penalty for container violation
    penalty = 100.0 * np.sum(container_violation**2)
    return soft_min_final - penalty

def _negative_soft_score(pts, beta_pair=8.0, beta_final=20.0):
    return -_soft_min_score(pts, beta_pair, beta_final)

def _hex_lattice_init():
    """Generate initial points using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _random_init():
    """Generate random initial points inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _ring_init():
    """Generate initial points on concentric rings matching known optimal structure."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Known structure: rings with approximately 6, 12, 18, 13 points (total 50)
    ring_configs = [
        (6, 0.25),
        (12, 0.45),
        (18, 0.65),
        (13, 0.80),
    ]
    for n_on_ring, ring_radius in ring_configs:
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        ring_pts = np.column_stack([ring_radius * np.cos(angles), ring_radius * np.sin(angles)])
        pts.append(ring_pts)
    pts = np.vstack(pts)
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        extra = N - len(pts)
        angles = np.random.uniform(0, 2 * np.pi, extra)
        radii = np.sqrt(np.random.uniform(0.5, 0.95, extra))
        extra_pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = np.vstack([pts, extra_pts])
    return pts

def _gradient_descent(pts, lr_init=0.01, n_steps=200, deadline=None):
    """Custom gradient descent with adaptive learning rate."""
    pts = pts.copy()
    lr = lr_init
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute soft score and gradient numerically
        eps = 1e-6
        grad = np.zeros_like(pts)
        base_score = _soft_min_score(pts)
        
        for i in range(N):
            for j in range(2):
                pts_plus = pts.copy()
                pts_plus[i, j] += eps
                score_plus = _soft_min_score(pts_plus)
                grad[i, j] = -(score_plus - base_score) / eps
        
        # Update
        pts_new = pts - lr * grad
        
        # Clip to stay inside container
        dists = np.sqrt(np.sum(pts_new**2, axis=1))
        mask = dists > 0.99
        if np.any(mask):
            pts_new[mask] *= 0.99 / dists[mask]
        
        new_score = _score(pts_new)
        if new_score > best_score:
            best_score = new_score
            best_pts = pts_new.copy()
        
        pts = pts_new
        lr *= 0.999
    
    return best_pts

def _optimize_from_start(init_pts, deadline):
    """Try multiple optimization methods from a starting point."""
    best_pts = init_pts.copy()
    best_score = _score(best_pts)
    
    # Method 1: SLSQP with soft objective
    try:
        res = minimize(
            _negative_soft_score,
            init_pts,
            method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-12}
        )
        candidate = res.x
        cand_score = _score(candidate)
        if cand_score > best_score:
            best_score = cand_score
            best_pts = candidate.copy()
    except Exception:
        pass
    
    # Method 2: L-BFGS-B with soft objective
    if time.time() < deadline - 20.0:
        try:
            res2 = minimize(
                _negative_soft_score,
                best_pts,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate2 = res2.x
            cand_score2 = _score(candidate2)
            if cand_score2 > best_score:
                best_score = cand_score2
                best_pts = candidate2.copy()
        except Exception:
            pass
    
    # Method 3: Gradient descent
    if time.time() < deadline - 10.0:
        try:
            gd_pts = _gradient_descent(best_pts, lr_init=0.005, n_steps=100, deadline=deadline)
            gd_score = _score(gd_pts)
            if gd_score > best_score:
                best_score = gd_score
                best_pts = gd_pts.copy()
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin for safety
    
    # Generate multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(8):
        inits.append(_random_init())
    
    best_pts = inits[0].copy()
    best_score = _score(best_pts)
    
    # Try to improve each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        optimized_pts, opt_score = _optimize_from_start(init_pts, deadline)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = optimized_pts.copy()
    
    # Final polish with high precision
    if time.time() < deadline - 5.0 and best_score > 0:
        try:
            res_final = minimize(
                _negative_soft_score,
                best_pts,
                method='SLSQP',
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            candidate_final = res_final.x
            cand_score_final = _score(candidate_final)
            if cand_score_final > best_score:
                best_score = cand_score_final
                best_pts = candidate_final.copy()
        except Exception:
            pass
        
        # Even more refinement
        if time.time() < deadline - 2.0:
            try:
                res_final2 = minimize(
                    _negative_soft_score,
                    best_pts,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                candidate_final2 = res_final2.x
                cand_score_final2 = _score(candidate_final2)
                if cand_score_final2 > best_score:
                    best_score = cand_score_final2
                    best_pts = candidate_final2.copy()
            except Exception:
                pass
    
    # Ensure all points are strictly inside the unit circle
    dists_center = np.sqrt(np.sum(best_pts**2, axis=1))
    mask = dists_center >= 0.9999
    if np.any(mask):
        scale = 0.9999 / dists_center[mask]
        best_pts[mask] *= scale[:, np.newaxis]
    
    return best_pts
