"""Packing 50 equal circles in a unit circle using soft-min surrogate + PBD + gradient optimization."""
import numpy as np
import time

N = 50
DEADLINE = 142.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values with numerical stability."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    # Numerically stable log-sum-exp
    shifted = v - m
    max_shifted = np.max(shifted)
    if max_shifted > 700:
        # Avoid overflow
        exp_terms = np.exp(-beta * (shifted - max_shifted))
        return m + max_shifted - (1.0 / beta) * np.log(np.sum(exp_terms))
    else:
        exp_terms = np.exp(-beta * shifted)
        return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    obj = -_soft_min(vals, beta)
    
    # Soft-min weights
    v_min = np.min(vals)
    shifted = vals - v_min
    max_shifted = np.max(shifted)
    if max_shifted > 700:
        exp_vals = np.exp(-beta * (shifted - max_shifted))
        S = np.sum(exp_vals)
        weights = exp_vals / S
    else:
        exp_vals = np.exp(-beta * shifted)
        S = np.sum(exp_vals)
        weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    # Vectorized pair gradient
    pair_weights = weights[:n*(n-1)//2]
    mask = pair_weights > 1e-12
    if np.any(mask):
        pi_idx = i_idx[mask]
        pj_idx = j_idx[mask]
        pw = pair_weights[mask]
        pd = pair_dists[mask]
        valid = pd > 1e-10
        if np.any(valid):
            pi_idx = pi_idx[valid]
            pj_idx = pj_idx[valid]
            pw = pw[valid]
            pd = pd[valid]
            vecs = (centers[pi_idx] - centers[pj_idx]) / pd[:, np.newaxis]
            contrib = pw[:, np.newaxis] * vecs * 0.5
            np.add.at(grad, pi_idx, contrib)
            np.add.at(grad, pj_idx, -contrib)
    
    # Vectorized boundary gradient
    bound_weights = weights[n*(n-1)//2:]
    bm = bound_weights > 1e-12
    if np.any(bm):
        bi = np.where(bm)[0]
        bw = bound_weights[bi]
        bn = norms[bi]
        valid_b = bn > 1e-10
        if np.any(valid_b):
            bi = bi[valid_b]
            bw = bw[valid_b]
            bn = bn[valid_b]
            contrib = bw[:, np.newaxis] * centers[bi] / bn[:, np.newaxis]
            np.add.at(grad, bi, -contrib)
        
    return obj, -grad

def _pbd_relax(centers, target_r, start_time, max_iters=2000, tol=1e-10):
    """
    Position-Based Dynamics relaxation with hard constraint projection.
    Alternates between pairwise overlap resolution and boundary projection.
    """
    current = centers.copy()
    n = len(current)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        prev = current.copy()
        
        # Pairwise overlap resolution
        for i in range(n):
            for j in range(i + 1, n):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 2 * target_r:
                    if d < 1e-12:
                        # Random separation
                        angle = np.random.uniform(0, 2 * np.pi)
                        current[i] += target_r * np.array([np.cos(angle), np.sin(angle)])
                        current[j] -= target_r * np.array([np.cos(angle), np.sin(angle)])
                    else:
                        overlap = 2 * target_r - d
                        direction = diff / d
                        current[i] += 0.5 * overlap * direction
                        current[j] -= 0.5 * overlap * direction
        
        # Boundary projection
        norms = np.linalg.norm(current, axis=1)
        mask = norms > (1.0 - target_r)
        if np.any(mask):
            current[mask] *= (1.0 - target_r) / norms[mask]
        
        # Check convergence
        delta = np.max(np.linalg.norm(current - prev, axis=1))
        if delta < tol:
            break
    
    return current

def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Optimize from a given starting point."""
    try:
        res = minimize(
            _objective_and_gradient, 
            centers.flatten(), 
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return centers

def _get_initial_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.12
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

def _get_ring_initialization(n, num_rings=3):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    
    remaining = n - 1
    points_per_ring = [remaining // num_rings] * num_rings
    for i in range(remaining - sum(points_per_ring)):
        points_per_ring[i % num_rings] += 1
    
    for ring_idx, count in enumerate(points_per_ring):
        if count <= 0:
            continue
        radius = 0.3 + ring_idx * 0.25
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    
    return np.array(centers[:n])

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.85)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_fibonacci_initialization(n):
    """Generate Fibonacci sphere-like initialization in 2D (golden angle spiral)."""
    centers = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        angle = i * golden_angle
        radius = np.sqrt((i + 0.5) / n) * 0.95
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=80, gtol=1e-9)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _final_high_precision_polish(centers, start_time):
    """Final polish with very high beta and tight tolerances for maximum precision."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [2000.0, 5000.0, 10000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=150, gtol=1e-11)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _local_search(centers, start_time, max_iters=40, step_size=0.015):
    """Local search with random perturbations and PBD relaxation."""
    current = centers.copy()
    current_score = _compute_score(current)
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Try random perturbation
        noisy = current + rng.normal(0, step_size, current.shape)
        
        # Keep inside container
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            noisy[mask] *= 0.95 / norms[mask]
        
        # PBD relax first
        target_r = current_score * 0.95
        relaxed = _pbd_relax(noisy, target_r, start_time, max_iters=500)
        
        # Then optimize
        optimized = _optimize_from_start(relaxed, beta=200.0, maxiter=60, gtol=1e-8)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > current_score:
                current_score = r
                current = optimized.copy()
    
    return current, current_score

def _pbd_with_binary_search(centers, start_time):
    """
    Use PBD with binary search on radius to find a feasible packing.
    """
    current = centers.copy()
    
    # Estimate initial radius
    r_low = 0.10
    r_high = 0.15
    
    # First, try to find a good radius with PBD
    best_r = _compute_score(current)
    best_centers = current.copy()
    
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        r_mid = (r_low + r_high) / 2
        relaxed = _pbd_relax(current, r_mid, start_time, max_iters=3000)
        r_achieved = _compute_score(relaxed)
        
        if r_achieved >= r_mid:
            r_low = r_mid
            if r_achieved > best_r:
                best_r = r_achieved
                best_centers = relaxed.copy()
        else:
            r_high = r_mid
    
    return best_centers, best_r

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=555),
        _get_fibonacci_initialization(N),
    ]
    
    # First pass: PBD relax each initialization to get feasible packings
    for init in initializations:
        if time.time() - start_time > DEADLINE - 60:
            break
        
        # Ensure all points are inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
        
        # PBD relax with a target radius
        target_r = 0.12
        relaxed = _pbd_relax(init, target_r, start_time, max_iters=1500)
        
        # Then gradient optimize
        optimized = _optimize_from_start(relaxed, beta=50.0, maxiter=100, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()
    
    # Second pass: PBD binary search on the best solution
    if best_centers is not None and time.time() - start_time < DEADLINE - 40:
        pbd_centers, pbd_score = _pbd_with_binary_search(best_centers, start_time)
        if pbd_score > best_score:
            best_score = pbd_score
            best_centers = pbd_centers
    
    # Third pass: polish the best solution with gradient optimization
    if best_centers is not None and time.time() - start_time < DEADLINE - 30:
        best_centers, best_score = _polish(best_centers, start_time)
    
    # Fourth pass: local search with perturbations and PBD
    if best_centers is not None and time.time() - start_time < DEADLINE - 15:
        best_centers, best_score = _local_search(best_centers, start_time, max_iters=30, step_size=0.012)
    
    # Fifth pass: final high-precision polish
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        best_centers, best_score = _final_high_precision_polish(best_centers, start_time)
    
    if best_centers is None:
        return initializations[0]
        
    return best_centers
# EVOLVE-BLOCK-END
