"""Circle packing in a unit disk: 50 equal circles, maximize radius."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

N = 50
DEADLINE = 130.0  # seconds, safety margin before 150s kill


def _score(pts):
    """Compute the radius score for a set of points."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = pdist(pts)
    min_dist = dists.min() if len(dists) > 0 else 0.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_margin = (1.0 - radii).min()
    r = min(min_dist / 2.0, min_margin)
    return r if r > 0 else 0.0


def _soft_objective(x, beta=50.0):
    """Smooth approximation of negative packing radius using log-sum-exp."""
    pts = x.reshape(N, 2)
    # Pairwise distances
    dists = pdist(pts)
    # Container margins
    radii = np.hypot(pts[:, 0], pts[:, 1])
    margins = 1.0 - radii
    
    # We want to maximize min(min_dist/2, min_margin)
    # Equivalently minimize -min(min_dist/2, min_margin)
    # Use soft-min: -softmin(a, b) where softmin approximates min
    
    # All constraints: pairwise distances/2 and margins
    # We want min of all these
    # softmin(x) = -(1/beta) * log(sum(exp(-beta*x_i)))
    all_vals = np.concatenate([dists / 2.0, margins])
    
    # Numerically stable soft-min
    min_val = all_vals.min()
    exp_terms = np.exp(-beta * (all_vals - min_val))
    soft_min = min_val - np.log(exp_terms.sum()) / beta
    
    return -soft_min


def _hard_objective(x):
    """Exact negative packing radius."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def _hex_init(spread=1.0):
    """Generate hexagonal lattice points inside unit disk, take N closest to center."""
    pts = []
    s = 0.22 * spread
    sqrt3 = np.sqrt(3.0)
    for j in range(-8, 9):
        for i in range(-8, 9):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * sqrt3 / 2.0
            if np.hypot(x, y) < 0.95:
                pts.append([x, y])
    pts = np.array(pts)
    d = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(d)[:N]
    return pts[idx]


def _ring_init(ring_radii, ring_counts):
    """Generate points on concentric rings."""
    pts = []
    for r, n in zip(ring_radii, ring_counts):
        if n == 1:
            pts.append([0.0, 0.0])
        elif n > 1:
            angles = 2 * np.pi * np.arange(n) / n
            pts.extend(list(zip(r * np.cos(angles), r * np.sin(angles))))
    return np.array(pts[:N])


def construct():
    t_start = time.time()
    best_score = 0.0
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        if pts is None:
            return
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal lattice with different spreads
    for spread in [0.9, 1.0, 1.1, 1.2]:
        inits.append(_hex_init(spread))
    
    # Ring configurations
    ring_configs = [
        ([0.0, 0.30, 0.55, 0.78, 0.93], [1, 6, 12, 18, 13]),
        ([0.0, 0.35, 0.60, 0.82, 0.95], [1, 6, 12, 18, 13]),
        ([0.0, 0.33, 0.58, 0.80, 0.94], [1, 6, 12, 18, 13]),
        ([0.0, 0.28, 0.52, 0.75, 0.92], [1, 6, 12, 18, 13]),
        ([0.0, 0.38, 0.63, 0.84, 0.96], [1, 6, 12, 18, 13]),
    ]
    for rr, cc in ring_configs:
        inits.append(_ring_init(rr, cc))
    
    # Evaluate all inits
    for init in inits:
        if time.time() - t_start > DEADLINE:
            break
        update_best(init)
    
    # Stage 1: L-BFGS-B with soft objective from best init
    if best_pts is not None and time.time() - t_start < DEADLINE - 20:
        x0 = best_pts.flatten()
        
        # Try multiple beta values for soft objective
        for beta in [30.0, 50.0, 80.0]:
            if time.time() - t_start > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _soft_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                pts = res.x.reshape(N, 2)
                update_best(pts)
                x0 = best_pts.flatten()  # Use updated best
            except Exception:
                pass
    
    # Stage 2: Differential evolution (global search)
    if time.time() - t_start < DEADLINE - 40:
        # Bounds: points must be inside unit circle, so |x|,|y| <= 1
        # But we can use tighter bounds: -0.95 to 0.95 for each coordinate
        bounds = [(-0.95, 0.95)] * (2 * N)
        
        try:
            de_result = differential_evolution(
                _hard_objective,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-6,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                init='sohal',
                updating='immediate',
                seed=42,
            )
            if de_result.success or de_result.fun < 0:
                pts = de_result.x.reshape(N, 2)
                update_best(pts)
        except Exception:
            pass
    
    # Stage 3: Multi-start L-BFGS-B with hard objective from multiple perturbations
    if best_pts is not None and time.time() - t_start < DEADLINE - 15:
        np.random.seed(123)
        
        for attempt in range(8):
            if time.time() - t_start > DEADLINE - 5:
                break
            
            if attempt == 0:
                x_init = best_pts.flatten()
            else:
                # Various perturbation strategies
                if attempt <= 3:
                    pert = np.random.randn(N, 2) * (0.02 + 0.01 * attempt)
                    x_init = best_pts.flatten() + pert.flatten()
                elif attempt <= 5:
                    # Random subset perturbation
                    n_pert = np.random.randint(5, 15)
                    idx = np.random.choice(N, n_pert, replace=False)
                    x_init = best_pts.flatten().copy()
                    x_init[idx * 2: (idx + 1) * 2] += np.random.randn(n_pert, 2) * 0.05
                else:
                    # Larger random restart from hex init
                    x_init = _hex_init(np.random.uniform(0.8, 1.3)).flatten()
            
            try:
                res = minimize(
                    _hard_objective,
                    x_init,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
                )
                pts = res.x.reshape(N, 2)
                update_best(pts)
            except Exception:
                pass
    
    # Stage 4: Final polish with very tight tolerance
    if best_pts is not None and time.time() - t_start < DEADLINE - 3:
        x0 = best_pts.flatten()
        for beta in [100.0, 200.0]:
            if time.time() - t_start > DEADLINE - 1:
                break
            try:
                res = minimize(
                    _soft_objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
                )
                pts = res.x.reshape(N, 2)
                update_best(pts)
                x0 = best_pts.flatten()
            except Exception:
                pass
        
        # One more hard objective pass
        if time.time() - t_start < DEADLINE - 1:
            try:
                res = minimize(
                    _hard_objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
                )
                pts = res.x.reshape(N, 2)
                update_best(pts)
            except Exception:
                pass
    
    if best_pts is None:
        best_pts = _hex_init(1.0)
    
    return best_pts
