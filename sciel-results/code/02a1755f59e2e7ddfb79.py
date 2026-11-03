"""Packing 50 equal circles in a unit circle using multi-stage optimization."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50


def _pairwise_distances(pts):
    """Compute pairwise distances for (N,2) array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))


def _score(pts):
    """Compute the packing score: min of all pair-dist/2 and boundary-dist."""
    dists = _pairwise_distances(pts)
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - radii)
    return min(min_pair, min_boundary)


def _softmin_logsumexp(x, beta):
    """Numerically stable soft-min: approximates min(x) as beta->inf."""
    neg_beta_x = -beta * x
    max_val = np.max(neg_beta_x)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    return -log_sum_exp / beta


def _smooth_objective(points_flat, beta):
    """Smooth surrogate: minimize negative soft-min of all critical distances."""
    pts = points_flat.reshape(N, 2)
    dists = _pairwise_distances(pts)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    sm = _softmin_logsumexp(scaled, beta)
    return -sm


def _ring_to_points(params):
    """
    Convert ring parameters to point positions.
    params: [r1, r2, r3, r4, p1, p2, p3, p4]
    Ring structure: 1 center + 6 + 12 + 15 + 16 = 50
    """
    r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
    p1, p2, p3, p4 = params[4], params[5], params[6], params[7]
    
    pts = [(0.0, 0.0)]
    
    # Ring 1: 6 points
    for k in range(6):
        angle = 2 * np.pi * k / 6 + p1
        pts.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    # Ring 2: 12 points
    for k in range(12):
        angle = 2 * np.pi * k / 12 + p2
        pts.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    # Ring 3: 15 points
    for k in range(15):
        angle = 2 * np.pi * k / 15 + p3
        pts.append((r3 * np.cos(angle), r3 * np.sin(angle)))
    
    # Ring 4: 16 points
    for k in range(16):
        angle = 2 * np.pi * k / 16 + p4
        pts.append((r4 * np.cos(angle), r4 * np.sin(angle)))
    
    return np.array(pts)


def _hexagonal_init():
    """
    Generate initial points using a hexagonal lattice pattern clipped to unit circle.
    This provides a good starting configuration with near-optimal local packing.
    """
    pts = []
    # Hexagonal lattice spacing
    s = 0.25  # side length of hexagons
    dx = s
    dy = s * np.sqrt(3) / 2
    
    # Generate lattice points
    for i in range(-8, 9):
        for j in range(-8, 9):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            if x**2 + y**2 < 0.95**2:
                pts.append((x, y))
    
    pts = np.array(pts)
    
    # If we have too many, select 50 closest to center
    if len(pts) > N:
        dists = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
    
    # If we have too few, add more points
    while len(pts) < N:
        angle = 2 * np.pi * len(pts) / N
        r = 0.5 + 0.4 * (len(pts) / N)
        pts = np.vstack([pts, [[r * np.cos(angle), r * np.sin(angle)]]])
    
    return pts[:N]


def _ring_init():
    """Generate initial points using concentric rings."""
    params = np.array([0.25, 0.50, 0.75, 0.93, 0.0, np.pi/6, np.pi/10, np.pi/8])
    return _ring_to_points(params)


def _anneal_points(pts, t_start, deadline, max_beta=2000):
    """
    Multi-stage annealing with increasing beta values.
    Returns optimized points.
    """
    current = pts.copy()
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, max_beta]
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        x0 = current.flatten()
        
        def obj_flat(pf, b=beta):
            return _smooth_objective(pf, b)
        
        try:
            result = minimize(
                obj_flat,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            
            if result.fun < obj_flat(x0):
                new_pts = result.x.reshape(N, 2)
                if np.all(np.isfinite(new_pts)):
                    radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                    if np.all(radii < 1.0):
                        current = new_pts
        except Exception:
            pass
    
    return current


def _binary_search_polish(pts, t_start, deadline):
    """
    Binary search on radius to find the maximum feasible radius.
    Uses constraint-based optimization to push points outward.
    """
    if time.time() > deadline - 3.0:
        return pts
    
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Try to increase radius by adjusting positions
    for _ in range(20):
        if time.time() > deadline - 1.0:
            break
        
        # Try to push all points slightly outward
        scale = 1.0 + 0.001
        new_pts = pts * scale
        
        # Clamp to unit circle
        radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            scale_factor = 0.999 / radii[mask]
            new_pts[mask, :] *= scale_factor[:, np.newaxis]
        
        # Optimize with constraints
        x0 = new_pts.flatten()
        
        def obj_constrained(pf):
            p = pf.reshape(N, 2)
            # Penalty for violations
            dists = _pairwise_distances(p)
            iu = np.triu_indices(N, k=1)
            min_pair = np.min(dists[iu])
            radii = np.hypot(p[:, 0], p[:, 1])
            max_r = np.max(radii)
            # We want min_pair >= 2*r and max_r + r <= 1
            # Use a penalty approach
            penalty = 0
            for i in range(N):
                if radii[i] > 0.999:
                    penalty += (radii[i] - 0.999)**2 * 1000
            if min_pair < 2 * best_r:
                penalty += (2 * best_r - min_pair)**2 * 1000
            return penalty + _smooth_objective(pf, 500.0)
        
        try:
            result = minimize(
                obj_constrained,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            new_pts = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                r = _score(new_pts)
                if r > best_r:
                    best_r = r
                    best_pts = new_pts
                    pts = new_pts
        except Exception:
            pass
    
    return best_pts


def _local_optimize(pts, t_start, deadline):
    """
    Local optimization using small perturbations and gradient-based refinement.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    rng = np.random.RandomState(42)
    
    step_size = 0.002
    for iteration in range(500):
        if time.time() > deadline - 1.0:
            break
        
        # Random perturbation
        perturbed = best_pts + rng.uniform(-step_size, step_size, size=(N, 2))
        
        # Clamp to unit circle
        radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            scale_factor = 0.999 / radii[mask]
            perturbed[mask, :] *= scale_factor[:, np.newaxis]
        
        # Quick gradient-based refinement
        x0 = perturbed.flatten()
        
        def obj_refine(pf):
            return _smooth_objective(pf, 1000.0)
        
        try:
            result = minimize(
                obj_refine,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-14}
            )
            refined = result.x.reshape(N, 2)
            if np.all(np.isfinite(refined)):
                r = _score(refined)
                if r > best_score:
                    best_score = r
                    best_pts = refined
                    step_size *= 0.9
                else:
                    step_size *= 0.95
        except Exception:
            pass
    
    return best_pts


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # 5s safety margin
    
    # --- Phase 1: Multi-start initialization ---
    # Try multiple initial configurations and pick the best
    inits = []
    
    # Ring-based initialization
    ring_pts = _ring_init()
    inits.append(ring_pts)
    
    # Hexagonal lattice initialization
    hex_pts = _hexagonal_init()
    inits.append(hex_pts)
    
    # Perturbed ring initialization
    rng = np.random.RandomState(123)
    for seed in [1, 2, 3]:
        rng_local = np.random.RandomState(seed)
        perturbed_ring = ring_pts + rng_local.uniform(-0.02, 0.02, size=(N, 2))
        radii = np.hypot(perturbed_ring[:, 0], perturbed_ring[:, 1])
        mask = radii > 0.999
        if np.any(mask):
            scale_factor = 0.999 / radii[mask]
            perturbed_ring[mask, :] *= scale_factor[:, np.newaxis]
        inits.append(perturbed_ring)
    
    # Select best initial configuration
    best_init = None
    best_init_score = -1
    for init in inits:
        if time.time() > deadline - 100.0:
            break
        s = _score(init)
        if s > best_init_score:
            best_init_score = s
            best_init = init.copy()
    
    if best_init is None:
        best_init = ring_pts.copy()
    
    # --- Phase 2: Annealing from best initial ---
    if time.time() < deadline - 60.0:
        best_init = _anneal_points(best_init, t_start, deadline)
    
    # --- Phase 3: Binary search polish ---
    if time.time() < deadline - 20.0:
        best_init = _binary_search_polish(best_init, t_start, deadline)
    
    # --- Phase 4: Local optimization ---
    if time.time() < deadline - 5.0:
        best_init = _local_optimize(best_init, t_start, deadline)
    
    # Final validation
    radii = np.hypot(best_init[:, 0], best_init[:, 1])
    if np.any(radii > 1.0):
        scale_factor = 0.999 / radii
        best_init *= scale_factor[:, np.newaxis]
    
    return best_init
