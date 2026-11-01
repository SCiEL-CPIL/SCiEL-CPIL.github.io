"""
Packing 50 equal circles in a unit disk.
Combines the soft-max contact-aware optimization from the inspiration
with the adaptive local search and multi-start structure from the current program.

Key structural change: instead of a generic penalty objective, we use a
soft-max (log-sum-exp) over constraint violations, which focuses gradient
signal on the most violated constraints. We also add a spectral/Laplacian
regularization term that encourages the contact graph to be well-connected,
helping escape local minima where clusters form.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 142.0  # Safety margin for 150s limit


def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)


def _soft_max_objective_and_grad(params, r_target, beta, lam_spectral=0.0):
    """
    Soft-max objective over all constraint violations.
    Violations: boundary (|c_i| - (1-r)) and pairwise (2r - |c_i-c_j|).
    Optionally adds a spectral regularization term.
    """
    n = N
    centers = params.reshape(n, 2)
    
    # Boundary violations
    norms = np.sqrt(np.sum(centers ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    # Pairwise violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_viol = 2.0 * r_target - dists[iu]
    
    all_viol = np.concatenate([cont_viol, pair_viol])
    
    # Soft-max approximation
    max_v = np.max(all_viol)
    if max_v > 50.0 / beta:
        return max_v, np.zeros_like(params)
    
    exp_v = np.exp(beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / beta
    weights = exp_v / sum_exp
    
    grad = np.zeros_like(centers)
    
    # Container gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = centers / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs
    
    # Pairwise gradient
    iw = weights[n:]
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = centers[iu_valid[0]] - centers[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]
        
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)
    
    # Spectral regularization: penalize low algebraic connectivity
    # This encourages the contact graph to be well-connected
    if lam_spectral > 0.0:
        # Build a simple k-nearest neighbor graph
        k = 6
        dist_mat = dists.copy()
        np.fill_diagonal(dist_mat, np.inf)
        knn_idx = np.argsort(dist_mat, axis=1)[:, :k]
        
        # Build adjacency
        adj = np.zeros((n, n))
        for i in range(n):
            for j in knn_idx[i]:
                adj[i, j] = 1.0
        adj = np.maximum(adj, adj.T)
        
        # Laplacian
        degree = adj.sum(axis=1)
        lap = np.diag(degree) - adj
        
        # Compute algebraic connectivity (second smallest eigenvalue)
        try:
            eigvals = np.linalg.eigvalsh(lap)
            alg_conn = eigvals[1] if len(eigvals) > 1 else 0.0
            # Penalize low connectivity
            soft_max_val += lam_spectral * max(0.0, 2.0 - alg_conn)
        except Exception:
            pass
    
    return soft_max_val, grad.reshape(-1)


def _soft_max_optimize(pts_init, r_target, start_beta=50.0, end_beta=10000.0,
                       n_stages=6, max_iter_per_stage=400, lam_spectral=0.0,
                       start_time=0.0):
    """
    Optimize using soft-max objective with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage in range(n_stages):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        
        def obj_grad(x):
            pts = x.reshape(N, 2)
            val, grad = _soft_max_objective_and_grad(pts, r_target, beta, lam_spectral)
            return val, grad
        
        x0 = current_pts.reshape(-1)
        try:
            result = minimize(
                obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_stage, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            current_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    return current_pts


def _check_feasible(pts, r_target, tol=1e-6):
    """Check if configuration is feasible for given radius."""
    val, _ = _soft_max_objective_and_grad(pts.flatten(), r_target, beta=10000.0)
    return val < tol


def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) == 0:
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_initialization(n, seed=0, angle_offsets=None):
    """Generate initial points on concentric rings matching known optimal structure."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points at r=0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points at r=0.5
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 3: 18 points at r=0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 4: remaining points at r=0.9
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points[:n])


def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)


def _get_golden_angle_initialization(n, seed=0):
    """Generate points using golden angle spiral."""
    rng = np.random.default_rng(seed)
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * 0.95
        theta = i * golden_angle + rng.uniform(-0.05, 0.05)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def _adaptive_local_search(centers, start_time, deadline, n_iterations=3000):
    """
    Adaptive local search that focuses on the most constrained pairs.
    Uses larger steps for more violated constraints.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(777)
    
    for iteration in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        # Get current clearances
        n = len(best_centers)
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        i_idx, j_idx = np.triu_indices(n, k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        norms = np.sqrt(np.sum(best_centers * best_centers, axis=1))
        bound_dists = 1.0 - norms
        clearances = np.concatenate([pair_dists, bound_dists])
        min_clearance = np.min(clearances)
        
        # Find the most constrained pair
        min_idx = np.argmin(clearances)
        
        # Determine which constraint is limiting
        n_pairs = N * (N - 1) // 2
        if min_idx < n_pairs:
            # Pair constraint
            i, j = i_idx[min_idx], j_idx[min_idx]
            # Perturb both points
            direction = rng.choice([-1, 1])
            delta = rng.normal(0, 0.005, size=2)
            trial = best_centers.copy()
            trial[i] += direction * delta
            trial[j] -= direction * delta
        else:
            # Boundary constraint
            i = min_idx - n_pairs
            # Push inward
            norm = np.linalg.norm(best_centers[i])
            if norm > 1e-10:
                delta = -0.005 * best_centers[i] / norm * rng.uniform(0.5, 1.5)
                trial = best_centers.copy()
                trial[i] += delta
            else:
                continue
        
        # Ensure inside circle
        norms = np.linalg.norm(trial, axis=1)
        mask = norms > 0.999
        if np.any(mask):
            trial[mask] *= 0.999 / norms[mask]
        
        trial_score = _compute_score(trial)
        
        if trial_score > best_score:
            best_score = trial_score
            best_centers = trial.copy()
    
    return best_centers, best_score


def _separation_pass(centers, n_iter=50, step_size=0.01):
    """
    Iterative pairwise separation: for each pair that's too close,
    push them apart along the line connecting them.
    """
    centers = centers.copy()
    n = len(centers)
    current_score = _compute_score(centers)
    target_min_dist = 2.0 * current_score * 0.95
    
    for _ in range(n_iter):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs
        min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_dist_idx
        if i == j:
            continue
        
        d = dists[i, j]
        if d < target_min_dist:
            # Push apart
            vec = (centers[i] - centers[j]) / d
            push = (target_min_dist - d) * 0.5 * step_size / target_min_dist
            centers[i] += vec * push
            centers[j] -= vec * push
        
        # Keep inside circle
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            centers[mask] *= 0.999 / norms[mask]
    
    return centers


def _bisection_refine(init_pos, start_time, deadline):
    """
    Bisection search for optimal radius using soft-max optimizer.
    """
    init_score = _compute_score(init_pos)
    
    lo, hi = 0.05, 0.15
    best_r_for_init = init_score
    best_pos_for_init = init_pos.copy()
    
    for bisect_iter in range(10):
        if time.time() - start_time > deadline - 5:
            break
        
        mid_r = (lo + hi) / 2.0
        
        # Try to optimize for this radius
        optimized_pos = _soft_max_optimize(
            init_pos, mid_r,
            start_beta=50.0,
            end_beta=10000.0,
            n_stages=5,
            max_iter_per_stage=300,
            lam_spectral=0.01,
            start_time=start_time
        )
        
        # Check feasibility
        if _check_feasible(optimized_pos, mid_r, tol=1e-6):
            actual_score = _compute_score(optimized_pos)
            if actual_score > best_r_for_init:
                best_r_for_init = actual_score
                best_pos_for_init = optimized_pos.copy()
            lo = mid_r
        else:
            hi = mid_r
    
    return best_pos_for_init, best_r_for_init


def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_golden_angle_initialization(N, seed=0),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization using bisection + soft-max
    for init in initializations:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        centers, score = _bisection_refine(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        return initializations[0]
    
    # Adaptive local search
    if time.time() - start_time < DEADLINE - 15:
        best_centers, best_score = _adaptive_local_search(best_centers, start_time, DEADLINE)
    
    # Final separation polish
    if time.time() - start_time < DEADLINE - 5:
        best_centers = _separation_pass(best_centers, n_iter=30, step_size=0.005)
        final_score = _compute_score(best_centers)
        if final_score > best_score:
            best_score = final_score
    
    return best_centers


def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
