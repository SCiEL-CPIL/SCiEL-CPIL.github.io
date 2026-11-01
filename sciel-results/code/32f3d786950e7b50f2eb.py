"""Packing 300 circles in a unit circle using simulated annealing with ring initialization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    # Compute distance to boundary
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist, min_boundary)

def _ring_init(n, seed=0):
    """
    Initialize points in concentric rings with optimized angular phases.
    """
    rng = np.random.RandomState(seed)
    points = []
    
    # Estimate ring structure
    # For N points, number of rings ~ sqrt(2*N/sqrt(3))
    num_rings = int(np.sqrt(2.0 * n / np.sqrt(3.0))) + 1
    
    # Distribute points among rings proportional to circumference
    ring_counts = []
    remaining = n
    for k in range(num_rings):
        if k == 0:
            count = 1
        else:
            # Approximate count based on circumference
            count = max(1, int(2 * np.pi * k / (2 * np.sqrt(3.0) * np.sqrt(n))))
        ring_counts.append(count)
    
    # Adjust to match total
    diff = n - sum(ring_counts)
    if diff > 0:
        for i in range(len(ring_counts) - 1, -1, -1):
            add = min(diff, max(1, ring_counts[i] // 3 + 1))
            ring_counts[i] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for i in range(len(ring_counts) - 1, 0, -1):
            if ring_counts[i] > 1:
                remove = min(-diff, ring_counts[i] - 1)
                ring_counts[i] -= remove
                diff += remove
                if diff >= 0:
                    break
    
    # Place points in rings
    for k, count in enumerate(ring_counts):
        if k == 0:
            points.append([0.0, 0.0])
        else:
            # Ring radius: proportional to k
            # For hexagonal packing, ring radius ~ k * spacing
            # spacing ~ 2*r ~ 2/sqrt(N)
            r_ring = k * (1.0 / np.sqrt(n)) * 0.95
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + rng.uniform(0, 2 * np.pi / count)
            for angle in angles:
                points.append([r_ring * np.cos(angle), r_ring * np.sin(angle)])
    
    points = np.array(points)
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 1e-3, points.shape)
    
    # Ensure all points are inside the unit circle
    norms = np.linalg.norm(points, axis=1)
    outside = norms > 0.999
    if np.any(outside):
        points[outside] *= 0.999 / norms[outside, np.newaxis]
    
    return points

def _hex_lattice_init(n, seed=0):
    """
    Initialize points in a hexagonal lattice clipped to the unit circle.
    """
    rng = np.random.RandomState(seed)
    points = []
    
    # Estimate spacing
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate a large grid of hex lattice points
    max_k = int(1.0 / (s * 0.5)) + 2
    
    candidates = []
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) >= n:
        # Sort by distance to origin to pick the most central ones
        radii = np.linalg.norm(candidates, axis=1)
        indices = np.argsort(radii)[:n]
        points = candidates[indices]
    else:
        # Fallback: use ring init
        points = _ring_init(n, seed)
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 1e-4, points.shape)
    
    return points

def _simulated_annealing(points, max_time=100.0):
    """
    Simulated annealing optimization for circle packing.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # Initial temperature
    T = 0.01
    cooling_rate = 0.9995
    min_T = 1e-6
    
    # Step size
    step_size = 0.01
    
    n_iter = 0
    while time.time() < deadline - 5.0 and T > min_T:
        # Generate a random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        
        # Try to move point idx
        new_point = current[idx].copy()
        new_point += delta
        
        # Keep inside container
        norm = np.linalg.norm(new_point)
        if norm > 0.999:
            new_point *= 0.999 / norm
        
        # Compute new score
        current[idx] = new_point
        new_score = _compute_score(current)
        
        # Metropolis criterion
        delta_score = new_score - current_score
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Revert
            current[idx] = current[idx] - delta
        
        # Cool down
        T *= cooling_rate
        step_size *= 0.9999
        
        n_iter += 1
        if n_iter % 10000 == 0:
            # Periodically try a larger perturbation
            if np.random.rand() < 0.1:
                large_idx = np.random.randint(N)
                large_delta = np.random.normal(0, step_size * 5, 2)
                new_point = current[large_idx].copy() + large_delta
                norm = np.linalg.norm(new_point)
                if norm <= 0.999:
                    current[large_idx] = new_point
                    large_score = _compute_score(current)
                    if large_score > best_score:
                        best_score = large_score
                        best = current.copy()
                    else:
                        current[large_idx] -= large_delta
    
    return best, best_score

def _gradient_polish(points, max_time=30.0):
    """
    Gradient-based polishing using L-BFGS with soft-min objective.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    def objective_and_grad(x, beta):
        p = x.reshape(-1, 2)
        
        # Calculate distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Pair distances (upper triangle)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu] / 2.0
        
        # Boundary distances
        norms = np.linalg.norm(p, axis=1)
        bound_dists = 1.0 - norms
        
        # Soft-min computation
        z_pairs = -beta * pair_dists
        z_bounds = -beta * bound_dists
        
        max_z_pair = np.max(z_pairs)
        max_z_bound = np.max(z_bounds)
        max_z = max(max_z_pair, max_z_bound)
        
        exp_pairs = np.exp(z_pairs - max_z)
        exp_bounds = np.exp(z_bounds - max_z)
        
        sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
        log_sum_exp = max_z + np.log(sum_exp)
        
        val = log_sum_exp / beta
        
        # Gradient
        w_pairs = exp_pairs / sum_exp
        w_bounds = exp_bounds / sum_exp
        
        grad = np.zeros_like(p)
        
        i_idx, j_idx = iu
        safe_pair_dists = np.where(pair_dists > 1e-12, pair_dists, 1.0)
        coeff = w_pairs * 0.25 / safe_pair_dists
        vecs = p[i_idx] - p[j_idx]
        
        np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
        np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
        
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        bound_grad_vecs = -p / safe_norms[:, np.newaxis]
        grad += w_bounds[:, np.newaxis] * bound_grad_vecs
        
        return val, -grad
    
    current = points.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 200.0, 1000.0, 5000.0]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                lambda x: objective_and_grad(x, beta)[0],
                current.flatten(),
                jac=lambda x: objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 50,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                    'maxls': 10,
                    'disp': False
                }
            )
            
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            
            if candidate_score > best_score:
                best_score = candidate_score
                best = candidate
                current = candidate
        except Exception:
            pass
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_score = -np.inf
    
    # Try multiple initialization strategies
    init_strategies = [
        ('ring', _ring_init),
        ('hex', _hex_lattice_init),
    ]
    
    # Phase 1: Simulated annealing from multiple starts
    for init_name, init_func in init_strategies:
        if time.time() > deadline - 60:
            break
        
        for seed in range(3):
            if time.time() > deadline - 50:
                break
            
            try:
                points = init_func(N, seed=seed)
                sa_points, sa_score = _simulated_annealing(points, max_time=25.0)
                
                if sa_score > best_score:
                    best_score = sa_score
                    best_points = sa_points
            except Exception:
                pass
    
    # Phase 2: Gradient polish on best result
    if best_points is not None and time.time() < deadline - 10:
        try:
            polished, polished_score = _gradient_polish(best_points, max_time=30.0)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
    
    # Phase 3: Additional SA if time permits
    if best_points is not None and time.time() < deadline - 15:
        try:
            sa_points, sa_score = _simulated_annealing(best_points, max_time=20.0)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_points
        except Exception:
            pass
    
    # Final gradient polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            polished, polished_score = _gradient_polish(best_points, max_time=10.0)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished
        except Exception:
            pass
    
    # Ensure points are inside the unit circle
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1)
        outside = norms > 1.0
        if np.any(outside):
            best_points[outside] /= norms[outside, np.newaxis]
            best_score = _compute_score(best_points)
    
    # Fallback if something went wrong
    if best_points is None or best_score <= 0:
        best_points = _ring_init(N, seed=42)
        best_score = _compute_score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
