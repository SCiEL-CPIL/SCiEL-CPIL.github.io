"""Combined program: force-directed layout with graph constraints + gradient refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 138.0  # Safety margin from 150s limit

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    n = len(centers)
    if n != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    norms = np.linalg.norm(centers, axis=1)
    min_boundary = 1.0 - np.max(norms)
    if min_boundary < 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    pairwise = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(pairwise, np.inf)
    min_pair = np.min(pairwise)
    score = min(min_boundary, min_pair / 2.0)
    return score if score > 0 else 0.0

def _soft_min(values, beta=50.0):
    """Compute the soft minimum of an array of values."""
    v = np.asarray(values, dtype=float)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))

def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    Returns (objective, gradient) for L-BFGS-B.
    """
    centers = params.reshape(n, 2)
    if not np.all(np.isfinite(centers)):
        return 1e6, np.zeros_like(params)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    norms = np.linalg.norm(centers, axis=1)
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    obj = -_soft_min(vals, beta)
    
    # Compute gradient
    v_min = np.min(vals)
    exp_vals = np.exp(-beta * (vals - v_min))
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(centers)
    
    num_pairs = n * (n - 1) // 2
    pair_weights = weights[:num_pairs]
    bound_weights = weights[num_pairs:]
    
    # Pair gradients (vectorized)
    for k in range(num_pairs):
        w = pair_weights[k]
        if w < 1e-12:
            continue
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
    
    # Boundary gradients
    for i in range(n):
        w = bound_weights[i]
        if w < 1e-12:
            continue
        if norms[i] > 1e-10:
            grad[i] -= w * centers[i] / norms[i]
    
    return obj, -grad

def _force_directed_layout(centers, iterations=500, gamma=0.01, repulsion_power=4):
    """
    Force-directed layout with repulsive potentials.
    Nodes represent circles, edges represent contact constraints.
    Uses inverse-distance^4 repulsion + attraction toward contact distance.
    """
    n = len(centers)
    centers = centers.copy().astype(float)
    current_r = _compute_score(centers)
    
    # Build initial contact graph: pairs closer than 2.5 * r
    contact_dist = 2.5 * max(current_r, 0.1)
    
    for it in range(iterations):
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Repulsive forces: F_rep = k * r^(-power) * direction
        # Attractive forces for contact pairs: F_attr = k * (d - d_target) * direction
        
        forces = np.zeros_like(centers)
        
        # Repulsion from all pairs (strong when close)
        min_d = np.min(dists)
        if min_d < 1e-8:
            # Handle coincident points
            for i in range(n):
                for j in range(i+1, n):
                    if dists[i,j] < 1e-8:
                        forces[i] += 1.0 * np.array([1.0, 0.0])
                        forces[j] -= 1.0 * np.array([1.0, 0.0])
            continue
        
        # Vectorized repulsion
        # F_rep_ij = gamma * (2r/d)^power * (c_i - c_j)/d
        # We want to push apart when d < 2r, so use:
        # F = gamma * (2*current_r / d)^repulsion_power * direction
        
        r_est = max(current_r, 0.05)
        
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                if d < 1e-8:
                    continue
                # Repulsive force (always active, stronger when close)
                ratio = (2 * r_est) / d
                f_rep = gamma * (ratio ** repulsion_power)
                
                # Attractive force for near-contact pairs
                if d < 2.5 * r_est:
                    f_attr = 0.5 * gamma * (d - 2 * r_est)
                else:
                    f_attr = 0.0
                
                f_total = f_rep - f_attr  # positive = repel, negative = attract
                vec = (centers[i] - centers[j]) / d
                forces[i] += f_total * vec
                forces[j] -= f_total * vec
        
        # Boundary repulsion: push inward if too close to edge
        norms = np.linalg.norm(centers, axis=1)
        for i in range(n):
            if norms[i] > 0.9:
                push = 0.5 * gamma * ((norms[i] - 0.9) / 0.1) ** 2
                forces[i] -= push * centers[i] / norms[i]
        
        # Update positions with damping
        centers += forces
        gamma *= 0.995  # Annealing
        
        # Project back into feasible region
        norms = np.linalg.norm(centers, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0.95:
            centers *= 0.95 / max_norm
    
    return centers

def _project_feasible(centers, r_est=0.125):
    """Project centers to be inside unit disk and non-overlapping."""
    centers = centers.copy()
    
    # Keep inside disk
    norms = np.linalg.norm(centers, axis=1)
    max_allowed = 1.0 - r_est
    mask = norms > max_allowed
    if np.any(mask):
        centers[mask] *= max_allowed / norms[mask]
    
    # Resolve overlaps iteratively
    for _ in range(50):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        min_d = np.min(dists)
        if min_d >= 2 * r_est:
            break
        
        # Find closest pair and separate
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        if i == j:
            continue
        d = dists[i, j]
        if d < 1e-10:
            centers[i] += np.array([0.01, 0.01])
            centers[j] -= np.array([0.01, 0.01])
            continue
        vec = (centers[i] - centers[j]) / d
        overlap = 2 * r_est - d
        if overlap > 0:
            centers[i] += vec * overlap / 2
            centers[j] -= vec * overlap / 2
    
    # Final boundary check
    norms = np.linalg.norm(centers, axis=1)
    max_allowed = 1.0 - r_est
    mask = norms > max_allowed
    if np.any(mask):
        centers[mask] *= max_allowed / norms[mask]
    
    return centers

def _optimize_gradient(centers, beta, maxiter=100, gtol=1e-7):
    """Optimize using L-BFGS-B with gradient."""
    try:
        res = minimize(
            _objective_and_gradient,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        result = res.x.reshape(N, 2)
        if np.all(np.isfinite(result)):
            return result
        return centers
    except Exception:
        return centers

def _polish_sequence(centers, start_time):
    """Progressive polishing with increasing beta."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        maxiter = min(200, int(30 * (DEADLINE - 15 - (time.time() - start_time))))
        if maxiter < 20:
            break
        polished = _optimize_gradient(current, beta, maxiter=maxiter, gtol=1e-7)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _final_high_precision_polish(centers, start_time):
    """Final high-precision polish."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    for beta in [5000.0, 10000.0, 20000.0, 50000.0]:
        if time.time() - start_time > DEADLINE - 5:
            break
        polished = _optimize_gradient(current, beta, maxiter=100, gtol=1e-9)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    
    return current, best_score

def _hex_lattice_init(n, radius=0.125, rotation=0.0):
    """Generate hexagonal lattice initialization."""
    spacing = 2 * radius
    sqrt3 = np.sqrt(3)
    points = []
    max_k = int(np.ceil(1.0 / spacing)) + 2
    for j in range(-max_k, max_k + 1):
        for i in range(-max_k, max_k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * sqrt3 / 2.0
            if np.hypot(x, y) <= 1.0 - radius:
                points.append((x, y))
    
    points = np.array(points) if points else np.zeros((n, 2))
    
    if len(points) < n:
        # Fill with random points
        rng = np.random.RandomState(42)
        while len(points) < n:
            r = np.sqrt(rng.random()) * (1.0 - radius)
            theta = rng.random() * 2 * np.pi
            points = np.vstack([points, [[r * np.cos(theta), r * np.sin(theta)]]])
    
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    points = points[indices[:n]]
    
    # Apply rotation
    if rotation != 0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        points = points @ rot_mat.T
    
    return points

def _ring_init(n, radius=0.125):
    """Initialize with concentric rings."""
    centers = [(0.0, 0.0)]
    remaining = n - 1
    ring_radius = radius * 2.0
    ring_index = 1
    
    while remaining > 0:
        circumference = 2 * np.pi * ring_radius
        max_on_ring = max(1, int(circumference / (2 * radius)))
        num_on_ring = min(remaining, max_on_ring)
        
        for k in range(num_on_ring):
            angle = 2 * np.pi * k / num_on_ring + (ring_index % 2) * np.pi / num_on_ring
            x = ring_radius * np.cos(angle)
            y = ring_radius * np.sin(angle)
            centers.append((x, y))
            remaining -= 1
        
        ring_radius += radius * 2.0 * 0.866
        ring_index += 1
    
    return np.array(centers[:n])

def _random_init(n, radius=0.12, seed=42):
    """Generate random initialization."""
    rng = np.random.RandomState(seed)
    centers = []
    for _ in range(n):
        r = np.sqrt(rng.random()) * (1.0 - radius)
        theta = rng.random() * 2 * np.pi
        centers.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(centers)

def _golden_spiral_init(n, radius=0.12):
    """Generate initialization using golden angle spiral."""
    golden_angle = np.pi * (3 - np.sqrt(5))
    centers = []
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * (1.0 - radius)
        theta = i * golden_angle
        centers.append((r * np.cos(theta), r * np.sin(theta)))
    return np.array(centers)

def _local_search(centers, start_time, max_iters=20, step_size=0.015):
    """Local search with random perturbations and gradient refinement."""
    current = centers.copy()
    current_score = _compute_score(current)
    rng = np.random.RandomState(12345)
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Random perturbation
        noisy = current + rng.normal(0, step_size, current.shape)
        
        # Project to feasible region
        noisy = _project_feasible(noisy, r_est=max(current_score, 0.1))
        
        # Optimize
        optimized = _optimize_gradient(noisy, beta=100.0, maxiter=40, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > current_score:
                current_score = r
                current = optimized.copy()
    
    return current, current_score

def construct():
    """Construct the best packing found within the time budget."""
    start_time = time.time()
    
    best_centers = None
    best_score = 0.0
    
    # Generate diverse initial configurations
    initializations = []
    
    # Hexagonal with various rotations
    for angle in [0.0, 0.3, 0.7, 1.2, 1.8, 2.5]:
        centers = _hex_lattice_init(N, radius=0.125, rotation=angle)
        initializations.append(centers)
    
    # Ring-based initialization
    initializations.append(_ring_init(N, radius=0.125))
    
    # Golden spiral
    initializations.append(_golden_spiral_init(N, radius=0.12))
    
    # Random initializations
    for seed in [42, 123, 456]:
        initializations.append(_random_init(N, radius=0.12, seed=seed))
    
    # Process each initialization
    for i, init_centers in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        remaining = DEADLINE - 15 - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Step 1: Force-directed layout to get a good starting point
        fd_centers = _force_directed_layout(init_centers, iterations=300, gamma=0.02, repulsion_power=4)
        
        # Project to feasible region
        fd_centers = _project_feasible(fd_centers, r_est=0.12)
        
        # Step 2: Gradient-based optimization with progressive beta
        polished, score = _polish_sequence(fd_centers, start_time)
        
        if score > best_score:
            best_score = score
            best_centers = polished
    
    # If we have a good solution, do local search to escape local minima
    if best_centers is not None and time.time() - start_time < DEADLINE - 20:
        ls_centers, ls_score = _local_search(best_centers, start_time, max_iters=15, step_size=0.012)
        if ls_score > best_score:
            best_score = ls_score
            best_centers = ls_centers
    
    # Final high-precision polish
    if best_centers is not None and time.time() - start_time < DEADLINE - 5:
        hp_centers, hp_score = _final_high_precision_polish(best_centers, start_time)
        if hp_score > best_score:
            best_score = hp_score
            best_centers = hp_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _hex_lattice_init(N, radius=0.12)
    
    return best_centers
