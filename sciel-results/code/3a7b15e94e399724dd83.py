"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)

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
        return _get_random_initialization(n, seed=42)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_initialization(n, seed=42)
    
    # Sort by distance to center to pick central ones
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        # Rejection sampling to ensure uniform density in area
        while True:
            radius = rng.uniform(0, 1.0)
            if radius <= 0.95:
                break
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization."""
    centers = []
    centers.append([0.0, 0.0])
    
    # Approximate distribution: 1, 6, 12, 12, 9, 10? 
    # Total 50. 
    # Radii approx: 0, 0.25, 0.5, 0.75, 0.9
    rings = [
        (6, 0.25),
        (12, 0.5),
        (12, 0.75),
        (10, 0.9)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])

def _get_perturbed_hex_lattice(n, seed=1):
    """Generate a hexagonal lattice with small random perturbations."""
    base = _get_initial_hex_lattice(n)
    rng = np.random.default_rng(seed)
    perturbed = base + rng.normal(0, 0.01, size=base.shape)
    # Ensure inside
    norms = np.linalg.norm(perturbed, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        perturbed[mask] *= 0.95 / norms[mask]
    return perturbed

def _objective_soft_min(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    m = np.min(vals)
    # Shift to avoid overflow
    vals_shifted = vals - m
    exp_terms = np.exp(-beta * vals_shifted)
    obj = -(m - (1.0 / beta) * np.log(np.sum(exp_terms)))
    
    # Gradient
    grad = np.zeros_like(centers)
    
    exp_vals = exp_terms
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        if w < 1e-8:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-8:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
            
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-8:
            continue
        if norms[i] > 1e-8:
            grad[i] -= w * centers[i] / norms[i]
            
    return obj, -grad

def _polish_subgradient(centers, start_time, max_iters=2000, initial_step=0.005):
    """
    Direct approach: Sub-gradient ascent on the minimum distance.
    Identifies the tightest constraint and pushes points apart.
    Uses a simple adaptive step size.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    step_size = initial_step
    no_improve_count = 0
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        # Determine which constraint is the bottleneck
        # r = min(min_pair_dist/2, min_bound_dist)
        # We want to maximize r.
        
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j: i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                # Move i and j apart
                current[i] += vec * step_size * 0.5
                current[j] -= vec * step_size * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                # Move i towards center
                current[i] -= vec * step_size
        
        # Ensure no points go out of bounds (safety check)
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.99
        if np.any(mask):
            current[mask] *= 0.99 / new_norms[mask]
            
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
            no_improve_count = 0
            # Increase step size slightly if improving
            step_size *= 1.01
        else:
            no_improve_count += 1
            # Decrease step size if not improving
            step_size *= 0.95
            if no_improve_count > 50:
                break

    return best_centers, best_score

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. L-BFGS-B with soft min (beta 50 -> 500)
    2. Sub-gradient polish
    """
    current = centers.copy()
    
    # Stage 1: L-BFGS-B
    betas = [50.0, 200.0, 500.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            res = minimize(
                _objective_soft_min, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if res.success or np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                # Verify validity
                if np.all(np.isfinite(candidate)):
                    current = candidate
        except Exception:
            pass
            
    # Stage 2: Direct Sub-gradient Polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_subgradient(current, start_time, max_iters=5000, initial_step=0.002)
        
    return current

def _nelder_mead_objective(params):
    """Objective for Nelder-Mead: negative of the packing score."""
    centers = params.reshape(N, 2)
    return -_compute_score(centers)

def _optimize_nelder_mead(centers, start_time):
    """
    Optimize using Nelder-Mead method.
    """
    current = centers.copy()
    
    # Ensure points are inside the circle
    norms = np.linalg.norm(current, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        current[mask] *= 0.95 / norms[mask]
    
    try:
        res = minimize(
            _nelder_mead_objective,
            current.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if np.isfinite(res.fun):
            candidate = res.x.reshape(N, 2)
            if np.all(np.isfinite(candidate)):
                # Ensure validity
                norms = np.linalg.norm(candidate, axis=1)
                if np.all(norms <= 1.0):
                    current = candidate
    except Exception:
        pass
    
    # Polish with subgradient
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _polish_subgradient(current, start_time, max_iters=3000, initial_step=0.002)
        
    return current

def _refine_with_high_beta(centers, start_time, max_betas=5):
    """
    Refine using very high beta soft-min to sharpen the minimum.
    This helps escape shallow local minima by focusing gradient on the tightest constraints.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    betas = [1000, 5000, 10000, 50000, 100000][:max_betas]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 5:
            break
        try:
            res = minimize(
                _objective_soft_min, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-8}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    score = _compute_score(candidate)
                    if score > best_score:
                        best_score = score
                        best_centers = candidate.copy()
                        current = candidate
        except Exception:
            pass
            
    return best_centers, best_score

def _perturb_and_retry(centers, start_time, n_perturbations=5):
    """
    Perturb the best solution slightly and re-optimize to escape local minima.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    rng = np.random.default_rng(12345)
    
    for p in range(n_perturbations):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Create a perturbed copy
        perturbed = best_centers.copy()
        noise_scale = 0.01 * (1.0 + p * 0.5)
        noise = rng.normal(0, noise_scale, size=perturbed.shape)
        perturbed += noise
        
        # Ensure validity
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            perturbed[mask] *= 0.95 / norms[mask]
            
        # Optimize the perturbed version
        optimized = _optimize_from_start(perturbed, start_time)
        if np.all(np.isfinite(optimized)):
            score = _compute_score(optimized)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_initial_hex_lattice(N),
        _get_ring_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=777),
        _get_perturbed_hex_lattice(N, seed=1),
        _get_perturbed_hex_lattice(N, seed=2),
    ]
    
    # Optimize each initialization with both methods
    for init in initializations:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        # Try L-BFGS-B optimization
        if time.time() - start_time < DEADLINE - 20:
            optimized_lbfgs = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized_lbfgs)):
                r = _compute_score(optimized_lbfgs)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_lbfgs.copy()
        
        # Try Nelder-Mead optimization
        if time.time() - start_time < DEADLINE - 15:
            optimized_nm = _optimize_nelder_mead(init, start_time)
            if np.all(np.isfinite(optimized_nm)):
                r = _compute_score(optimized_nm)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_nm.copy()
                
    if best_centers is None:
        return initializations[0]
        
    # Final polish on the absolute best
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=10000, initial_step=0.001)
        
    # High-beta refinement to sharpen the minimum
    if time.time() - start_time < DEADLINE - 5:
        refined, refined_score = _refine_with_high_beta(best_centers, start_time, max_betas=5)
        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined
        
    # Perturbation-retry loop to escape local minima
    if time.time() - start_time < DEADLINE - 10:
        perturbed, perturbed_score = _perturb_and_retry(best_centers, start_time, n_perturbations=3)
        if perturbed_score > best_score:
            best_score = perturbed_score
            best_centers = perturbed
            
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 2:
        best_centers, best_score = _polish_subgradient(best_centers, start_time, max_iters=5000, initial_step=0.0005)
        
    return best_centers
# EVOLVE-BLOCK-END
