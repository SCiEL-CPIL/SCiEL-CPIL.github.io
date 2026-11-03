"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


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


def _ring_initialization():
    """
    Generate initial points using concentric rings.
    Parameters: list of (radius, count, phase) for each ring.
    We'll optimize these parameters via DE.
    """
    # Known good structure for 50 circles:
    # Center: 1
    # Ring 1: ~6 at r~0.25
    # Ring 2: ~12 at r~0.5
    # Ring 3: ~15 at r~0.75
    # Ring 4: ~16 at r~0.95
    # Total: 1+6+12+15+16 = 50
    
    # Parameters to optimize: [r1, r2, r3, r4, phase1, phase2, phase3, phase4, jitter_scale]
    # Plus we can vary ring counts slightly
    
    def ring_params_to_points(params):
        r1, r2, r3, r4 = params[0], params[1], params[2], params[3]
        p1, p2, p3, p4 = params[4], params[5], params[6], params[7]
        jitter = params[8]
        
        pts = [(0.0, 0.0)]  # center
        
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
        
        pts = np.array(pts)
        
        # Add small deterministic jitter based on index (reproducible)
        if jitter > 0:
            rng = np.random.RandomState(42)
            jitter_vecs = rng.uniform(-jitter, jitter, size=(N, 2))
            pts = pts + jitter_vecs
        
        return pts
    
    return ring_params_to_points


def construct():
    t_start = time.time()
    deadline = 135.0  # 15s safety margin before 150s kill
    
    # --- Phase 1: Differential Evolution over ring parameters ---
    ring_params_to_points = _ring_initialization()
    
    # DE bounds: [r1, r2, r3, r4, p1, p2, p3, p4, jitter]
    # r1 in [0.15, 0.4], r2 in [0.4, 0.65], r3 in [0.65, 0.85], r4 in [0.85, 0.98]
    # phases in [0, 2*pi], jitter in [0, 0.02]
    bounds = [
        (0.15, 0.40),   # r1
        (0.35, 0.65),   # r2
        (0.60, 0.85),   # r3
        (0.82, 0.98),   # r4
        (0.0, 2 * np.pi),   # p1
        (0.0, 2 * np.pi),   # p2
        (0.0, 2 * np.pi),   # p3
        (0.0, 2 * np.pi),   # p4
        (0.0, 0.02),     # jitter
    ]
    
    def de_objective(params):
        pts = ring_params_to_points(params)
        # Check validity
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(radii > 0.999):
            return 1e10
        dists = _pairwise_distances(pts)
        iu = np.triu_indices(N, k=1)
        if np.min(dists[iu]) < 1e-6:
            return 1e10
        return -_score(pts)  # minimize negative score
    
    # Run DE with limited time
    de_time_budget = 30.0
    de_start = time.time()
    
    try:
        de_result = differential_evolution(
            de_objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False,
            callback=lambda x, convergence: (
                None if time.time() - de_start < de_time_budget else
                (_ for _ in ()).throw(StopIteration)
            )
        )
    except (StopIteration, Exception):
        # Use a reasonable default if DE times out or fails
        de_result = None
    
    if time.time() > deadline:
        # Return a default ring structure
        default_params = np.array([0.25, 0.50, 0.75, 0.93, 0.0, np.pi/6, np.pi/10, np.pi/8, 0.0])
        return ring_params_to_points(default_params)
    
    if de_result is not None and de_result.fun < 1e9:
        best_params = de_result.x
    else:
        best_params = np.array([0.25, 0.50, 0.75, 0.93, 0.0, np.pi/6, np.pi/10, np.pi/8, 0.0])
    
    initial_points = ring_params_to_points(best_params).copy()
    
    # --- Phase 2: Multi-stage soft-min annealing with L-BFGS-B ---
    # Start with low beta (broad) and increase (sharp)
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0]
    current_points = initial_points.copy()
    
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        
        x0 = current_points.flatten()
        
        def obj_flat(pf, b=beta):
            return _smooth_objective(pf, b)
        
        try:
            result = minimize(
                obj_flat,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < _smooth_objective(x0, beta):
                new_pts = result.x.reshape(N, 2)
                # Validate
                if np.all(np.isfinite(new_pts)):
                    radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                    if np.all(radii < 1.0):
                        current_points = new_pts
        except Exception:
            pass
    
    # --- Phase 3: Final polish with very high beta ---
    if time.time() < deadline - 3.0:
        try:
            x0 = current_points.flatten()
            result = minimize(
                lambda pf: _smooth_objective(pf, 1000.0),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            new_pts = result.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                if np.all(radii < 1.0):
                    current_points = new_pts
        except Exception:
            pass
    
    # --- Phase 4: Local search with small random perturbations ---
    if time.time() < deadline - 2.0:
        best_score = _score(current_points)
        rng = np.random.RandomState(123)
        step_size = 0.005
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            perturbed = current_points + rng.uniform(-step_size, step_size, size=(N, 2))
            # Clamp to stay inside
            radii = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = radii > 0.999
            if np.any(mask):
                scale = 0.999 / radii[mask]
                perturbed[mask, :] *= scale[:, np.newaxis]
            
            s = _score(perturbed)
            if s > best_score:
                current_points = perturbed
                best_score = s
                step_size *= 0.8
            else:
                step_size *= 0.95
    
    return current_points

# EVOLVE-BLOCK-END
