"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

# Deadline guard: must finish well before 150s wall-clock.
_DEADLINE = time.time() + 140.0  # 10s safety margin

def _hex_seed(n, spacing):
    """Generate a hexagonal lattice cut to the unit disk, return up to n points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                rows.append((x, y))
    pts = np.array(rows)
    # Sort by radius to prefer central points (better for packing)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    order = np.argsort(rad, kind='stable')
    return pts[order[:n]]


def _soft_min_radius(pts, beta):
    """
    Smooth surrogate for the minimum of:
      - min_{i<j} |c_i - c_j| / 2
      - min_i (1 - |c_i|)
    using log-sum-exp. Returns a smooth approximation of the true radius.
    Higher is better; we maximize this.
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Exclude self: set diagonal to a large value
    np.fill_diagonal(d2, 1e12)
    # Soft-min of pairwise distances / 2
    # logsumexp of -d / (2*beta) then negate
    neg_d_over_2b = -np.sqrt(d2) / (2.0 * beta)
    # Subtract max for numerical stability
    max_val = np.max(neg_d_over_2b)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_d_over_2b - max_val)))
    soft_pair_min = -log_sum_exp * beta  # approximates min d/2

    # Boundary: 1 - |c_i|
    r = np.hypot(pts[:, 0], pts[:, 1])
    boundary = 1.0 - r
    # Soft-min of boundary
    neg_b_over_b = -boundary / beta
    max_b = np.max(neg_b_over_b)
    log_sum_exp_b = max_b + np.log(np.sum(np.exp(neg_b_over_b - max_b)))
    soft_boundary_min = -log_sum_exp_b * beta

    # Combine: soft-min of the two soft-mins
    # We want min(soft_pair_min, soft_boundary_min)
    # Use a soft-min again
    vals = np.array([soft_pair_min, soft_boundary_min])
    neg_vals = -vals / beta
    max_v = np.max(neg_vals)
    log_sum_exp_v = max_v + np.log(np.sum(np.exp(neg_vals - max_v)))
    return -log_sum_exp_v * beta


def _true_radius(pts):
    """Compute the true (nonsmooth) radius for a given set of points."""
    n = len(pts)
    if n < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e12)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = 1.0 - np.max(r)
    return min(min_pair, min_boundary)


def _objective(x, beta):
    """Objective to minimize: negative of soft-min radius."""
    pts = x.reshape(-1, 2)
    return -_soft_min_radius(pts, beta)


def _gradient(x, beta):
    """Numerical gradient via finite differences (central)."""
    n = len(x)
    eps = 1e-6
    g = np.zeros_like(x)
    f0 = _objective(x, beta)
    for i in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (_objective(xp, beta) - _objective(xm, beta)) / (2.0 * eps)
    return g


def _polish(pts, max_iter=200, ftol=1e-12, xtol=1e-14):
    """
    High-precision polish pass: local solve with tight tolerances.
    Uses L-BFGS-B on the true (nonsmooth) objective via a smooth surrogate
    with very small beta, and many iterations.
    """
    n = len(pts)
    x = pts.flatten().copy()
    beta = 1e-4  # very small beta for near-exact min

    # Use L-BFGS-B with tight tolerances
    try:
        res = minimize(
            _objective,
            x,
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'ftol': ftol,
                'gtol': 1e-10,
                'xtol': xtol,
                'maxls': 50,
                'maxcor': 20,
            }
        )
        if res.success or res.fun < _objective(x, beta):
            new_pts = res.x.reshape(-1, 2)
            # Ensure validity
            if np.all(np.isfinite(new_pts)) and np.all(np.hypot(new_pts[:, 0], new_pts[:, 1]) <= 1.0):
                return new_pts
    except Exception:
        pass
    return pts


def _valid(pts):
    """Check if points are valid: finite, inside unit circle, no coincident centers."""
    if not np.all(np.isfinite(pts)):
        return False
    r = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r > 1.0 + 1e-12):
        return False
    # Check no coincident centers
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1e12)
    if np.min(d2) < 1e-16:
        return False
    return True


def construct():
    start_time = time.time()
    
    # Generate a good hexagonal seed
    # Estimate spacing: for N=300 in unit circle, area per circle ~ pi/(N) ~ 0.0105
    # hex spacing ~ sqrt(4*area/(sqrt(3))) ~ sqrt(4*0.0105/1.732) ~ 0.155
    # Try a few spacings to get a good seed
    best_seed = None
    best_seed_radius = -1.0
    
    for spacing in [0.15, 0.155, 0.16, 0.165, 0.17]:
        seed = _hex_seed(N, spacing)
        if len(seed) >= N:
            seed = seed[:N]
            r = _true_radius(seed)
            if r > best_seed_radius:
                best_seed_radius = r
                best_seed = seed.copy()
    
    if best_seed is None:
        # Fallback: uniform grid
        seed = _hex_seed(N, 0.18)
        if len(seed) < N:
            # Fill with random points
            extra = np.random.randn(N - len(seed), 2) * 0.5
            seed = np.vstack([seed, extra])
            seed = seed[:N]
        best_seed = seed
        best_seed_radius = _true_radius(best_seed)
    
    best_pts = best_seed.copy()
    best_radius = best_seed_radius
    
    # L-BFGS-B optimization with soft-min surrogate
    # Multiple restarts with small perturbations
    n_restarts = 3
    beta_initial = 0.01
    beta_final = 1e-3
    
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 30.0:
            break
        
        if restart == 0:
            x0 = best_pts.flatten().copy()
        else:
            # Perturb the best solution
            noise = np.random.randn(N, 2) * 0.005
            x0 = (best_pts + noise).flatten().copy()
        
        # Anneal beta: start large (smooth), decrease (sharper)
        betas = [beta_initial, beta_initial / 3.0, beta_final]
        
        for beta in betas:
            if time.time() > _DEADLINE - 20.0:
                break
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                        'xtol': 1e-12,
                        'maxls': 30,
                        'maxcor': 15,
                    }
                )
                if res.fun < _objective(x0, beta):
                    new_pts = res.x.reshape(-1, 2)
                    if _valid(new_pts):
                        new_r = _true_radius(new_pts)
                        if new_r > best_radius:
                            best_radius = new_r
                            best_pts = new_pts.copy()
                            x0 = new_pts.flatten().copy()
            except Exception:
                pass
        
        # Update x0 for next restart
        x0 = best_pts.flatten().copy()
    
    # High-precision polish pass
    if time.time() < _DEADLINE - 10.0:
        polished = _polish(best_pts, max_iter=300, ftol=1e-13, xtol=1e-15)
        if _valid(polished):
            r = _true_radius(polished)
            if r > best_radius:
                best_radius = r
                best_pts = polished
    
    # Final validity check
    if not _valid(best_pts):
        # Fallback to hex seed
        best_pts = best_seed
        best_radius = best_seed_radius
    
    return best_pts
# EVOLVE-BLOCK-END
