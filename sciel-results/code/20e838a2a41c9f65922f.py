"""Rewritten program for 50 circles in a circle packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s)

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
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
    r_est = 0.125
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.98:
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
        # Use sqrt for uniform area sampling
        radius = np.sqrt(rng.uniform(0, 0.95**2))
        angle = rng.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _get_ring_initialization(n):
    """Generate ring-based initialization with better spacing."""
    centers = []
    # Center point
    centers.append([0.0, 0.0])
    
    # Try to fill with rings
    # Approximate optimal: 1 + 6 + 12 + 12 + 10 + 9 = 50
    ring_configs = [
        (6, 0.26),
        (12, 0.52),
        (12, 0.78),
        (10, 0.95),
        (10, 0.95)  # backup if needed
    ]
    
    count = 1
    for count_in_ring, radius in ring_configs:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _force_directed_layout(centers, start_time, max_iters=5000):
    """
    Force-directed layout with repulsive potentials and hard constraint projection.
    Uses inverse distance^4 repulsion for strong separation.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    n = len(current)
    learning_rate = 0.001
    decay = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Repulsive forces between pairs (inverse distance^4)
        for i in range(n):
            for j in range(i+1, n):
                diff = current[i] - current[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    # Random perturbation for coincident points
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
                    continue
                
                # Force magnitude: k / dist^4, direction: away from each other
                k = 1.0
                force_mag = k / (dist**4)
                force_vec = diff / dist * force_mag
                
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Attractive force toward center for boundary constraint
        # Actually, we want to push points inward if they're near the boundary
        norms = np.linalg.norm(current, axis=1)
        for i in range(n):
            if norms[i] > 0.9:
                # Push inward
                forces[i] -= current[i] / norms[i] * (norms[i] - 0.9) * 10.0
        
        # Update positions
        current += learning_rate * forces
        
        # Hard constraint projection: ensure all points are inside unit circle
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 0.995
        if np.any(mask):
            current[mask] *= 0.995 / norms[mask]
        
        # Check improvement
        new_score = _compute_score(current)
        if new_score > best_score:
            best_score = new_score
            best_centers = current.copy()
        
        # Decay learning rate
        learning_rate *= decay
        if learning_rate < 1e-8:
            break
    
    return best_centers, best_score

def _soft_min_objective(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    Returns (objective, gradient)
    """
    centers = params.reshape(n, 2)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    m = np.min(vals)
    exp_terms = np.exp(-beta * (vals - m))
    obj = -(m - (1.0 / beta) * np.log(np.sum(exp_terms)))
    
    # Gradient computation
    grad = np.zeros_like(centers)
    
    S = np.sum(exp_terms)
    weights = exp_terms / S
    
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

def _subgradient_polish(centers, start_time, max_iters=10000, base_step=0.003):
    """
    Sub-gradient ascent on the minimum distance with adaptive step size.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    no_improve_count = 0
    step = base_step
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(current, axis=1)
        bound_dists = 1.0 - norms
        
        min_pair_dist = np.min(dists)
        min_bound_dist = np.min(bound_dists)
        
        # Determine bottleneck
        if min_pair_dist / 2.0 < min_bound_dist:
            # Bottleneck is a pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j:
                i, j = j, i
            d = dists[i, j]
            if d > 1e-8:
                vec = (current[i] - current[j]) / d
                current[i] += vec * step * 0.5
                current[j] -= vec * step * 0.5
        else:
            # Bottleneck is boundary
            i = np.argmin(bound_dists)
            norm = norms[i]
            if norm > 1e-8:
                vec = current[i] / norm
                current[i] -= vec * step
        
        # Safety: keep points inside
        new_norms = np.linalg.norm(current, axis=1)
        mask = new_norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / new_norms[mask]
            
        new_score = _compute_score(current)
        if new_score > best_score + 1e-12:
            best_score = new_score
            best_centers = current.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Adaptive step: reduce if stuck
            if no_improve_count % 100 == 0:
                step *= 0.95
                if step < 1e-6:
                    break
    
    return best_centers, best_score

def _optimize_from_start(centers, start_time):
    """
    Multi-stage optimization:
    1. Force-directed layout for initial separation
    2. L-BFGS-B with soft min (increasing beta)
    3. Sub-gradient polish
    """
    current = centers.copy()
    
    # Stage 1: Force-directed layout
    if time.time() - start_time < DEADLINE - 40:
        current, _ = _force_directed_layout(current, start_time, max_iters=3000)
    
    # Stage 2: L-BFGS-B
    betas = [30.0, 100.0, 300.0, 1000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 20:
            break
        try:
            res = minimize(
                _soft_min_objective, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'gtol': 1e-7, 'maxls': 50}
            )
            if np.isfinite(res.fun):
                candidate = res.x.reshape(N, 2)
                if np.all(np.isfinite(candidate)):
                    norms = np.linalg.norm(candidate, axis=1)
                    if np.all(norms <= 1.0):
                        current = candidate
        except Exception:
            pass
            
    # Stage 3: Sub-gradient Polish
    if time.time() - start_time < DEADLINE - 5:
        current, _ = _subgradient_polish(current, start_time, max_iters=15000, base_step=0.002)
        
    return current

def _nelder_mead_objective(params):
    """Objective for Nelder-Mead: negative of the packing score."""
    centers = params.reshape(N, 2)
    return -_compute_score(centers)

def _optimize_nelder_mead(centers, start_time):
    """
    Optimize using Nelder-Mead method with force-directed pre-processing.
    """
    current = centers.copy()
    
    # Ensure points are inside the circle
    norms = np.linalg.norm(current, axis=1)
    mask = norms > 0.95
    if np.any(mask):
        current[mask] *= 0.95 / norms[mask]
    
    # Force-directed pre-processing
    if time.time() - start_time < DEADLINE - 30:
        current, _ = _force_directed_layout(current, start_time, max_iters=2000)
    
    try:
        res = minimize(
            _nelder_mead_objective,
            current.flatten(),
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-9,
                'fatol': 1e-11,
                'adaptive': True
            }
        )
        if np.isfinite(res.fun):
            candidate = res.x.reshape(N, 2)
            if np.all(np.isfinite(candidate)):
                norms = np.linalg.norm(candidate, axis=1)
                if np.all(norms <= 1.0):
                    current = candidate
    except Exception:
        pass
    
    # Polish with subgradient
    if time.time() - start_time < DEADLINE - 3:
        current, _ = _subgradient_polish(current, start_time, max_iters=10000, base_step=0.0015)
        
    return current

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
        _get_random_initialization(N, seed=2024),
    ]
    
    # Optimize each initialization
    for idx, init in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.98 / norms[mask]
            
        # Try L-BFGS-B optimization (primary method)
        if time.time() - start_time < DEADLINE - 20:
            optimized_lbfgs = _optimize_from_start(init, start_time)
            if np.all(np.isfinite(optimized_lbfgs)):
                r = _compute_score(optimized_lbfgs)
                if r > best_score:
                    best_score = r
                    best_centers = optimized_lbfgs.copy()
        
        # Try Nelder-Mead optimization (secondary method for diversity)
        if idx < 4 and time.time() - start_time < DEADLINE - 15:
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
        best_centers, best_score = _subgradient_polish(best_centers, start_time, max_iters=20000, base_step=0.001)
        
    return best_centers
