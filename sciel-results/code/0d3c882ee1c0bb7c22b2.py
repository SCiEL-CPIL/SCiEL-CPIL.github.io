"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_radius(pts):
    """Compute the minimum radius for a given set of centers."""
    n = len(pts)
    # Distance to boundary
    boundary_dist = 1.0 - np.linalg.norm(pts, axis=1)
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances (vectorized)
    min_pair = pdist(pts) / 2.0
    min_pair = np.min(min_pair)
    
    return min(min_boundary, min_pair)


def _soft_min_radius(pts, beta=100.0):
    """
    Compute a soft-min approximation of the radius using log-sum-exp.
    This provides a smooth objective for optimization.
    Vectorized for speed.
    """
    n = len(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances (half)
    pair_dists = pdist(pts) / 2.0
    
    # Combine all constraints into one array
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    # Soft min via log-sum-exp
    # min(x) ≈ -1/beta * log(sum(exp(-beta * x)))
    # For numerical stability, use the standard log-sum-exp trick
    min_val = np.min(all_dists)
    if min_val < -100:  # Prevent overflow
        return min_val
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return min_val
    
    soft_min = min_val - np.log(sum_exp) / beta
    
    return soft_min


def _soft_min_radius_grad(pts, beta=100.0):
    """
    Compute the gradient of the soft-min radius.
    Vectorized for speed.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # Boundary distances
    norms = np.linalg.norm(pts, axis=1)
    boundary_dist = 1.0 - norms
    
    # Pairwise distances (half)
    pair_dists = pdist(pts) / 2.0
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dist, pair_dists])
    
    min_val = np.min(all_dists)
    if min_val < -100:
        return grad
    
    exp_vals = np.exp(-beta * (all_dists - min_val))
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return grad
    
    # Weights for soft min
    weights = exp_vals / sum_exp
    
    # Boundary terms
    # d(1 - ||p_i||)/dp_i = -p_i / ||p_i||
    # We want to maximize soft_min, so we return gradient of soft_min.
    # The optimizer minimizes -soft_min, so we return -grad(soft_min) in the caller?
    # Wait, the caller defines objective as -_soft_min_radius and gradient as -_soft_min_radius_grad.
    # So _soft_min_radius_grad should return d(soft_min)/d(pts).
    
    # For boundary:
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    boundary_grad = -pts / safe_norms[:, np.newaxis]
    # Weighted sum
    grad += weights[:n, np.newaxis] * boundary_grad
    
    # Pair terms
    # d(||p_i - p_j||/2)/dp_i = (p_i - p_j) / ||p_i - p_j||
    # d(||p_i - p_j||/2)/dp_j = (p_j - p_i) / ||p_i - p_j|| = -(p_i - p_j) / ||p_i - p_j||
    
    # We need to map the weights back to pairs.
    # pdist returns condensed matrix.
    # We can use a loop or vectorized accumulation.
    # For N=50, N(N-1)/2 = 1225 pairs. A loop is acceptable if the rest is fast, 
    # but let's try to be efficient.
    
    # Get pairwise differences
    # diff[i, j] = pts[i] - pts[j]
    # We only need i < j
    
    # Construct indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    diffs = pts[i_idx] - pts[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    
    # Gradients for each pair
    # grad_i += w * (diff / dist)
    # grad_j += w * (-diff / dist)
    
    pair_weights = weights[n:]
    pair_grads = pair_weights[:, np.newaxis] * (diffs / safe_dists[:, np.newaxis])
    
    # Accumulate into grad
    # For each pair k, add pair_grads[k] to grad[i_idx[k]] and subtract from grad[j_idx[k]]
    np.add.at(grad, i_idx, pair_grads)
    np.add.at(grad, j_idx, -pair_grads)
    
    return grad


def _hex_init(spacing):
    """Generate hexagonal lattice points."""
    k = int(1.0 / spacing) + 2
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append([x, y])
    points = np.array(rows)
    radius = np.linalg.norm(points, axis=1)
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]
    return points


def _ring_init():
    """Generate ring-based initialization."""
    points = [[0.0, 0.0]]
    
    # Inner ring
    r1 = 0.3
    n1 = 6
    for i in range(n1):
        angle = 2 * np.pi * i / n1
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Outer ring
    r2 = 0.65
    n2 = 12
    for i in range(n2):
        angle = 2 * np.pi * i / n2 + np.pi / 12
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Second outer ring
    r3 = 0.85
    n3 = 18
    for i in range(n3):
        angle = 2 * np.pi * i / n3
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    return np.array(points)


def _random_init(seed):
    """Generate random initialization."""
    np.random.seed(seed)
    points = []
    for _ in range(N):
        r = np.sqrt(np.random.rand())
        theta = 2 * np.pi * np.random.rand()
        points.append([r * np.cos(theta), r * np.sin(theta)])
    return np.array(points)


def _optimize_from_init(init_pts, beta_start=50.0, beta_end=500.0, n_anneal=10, deadline=None):
    """
    Optimize from an initial configuration using annealing.
    """
    if deadline is not None and time.time() > deadline:
        return init_pts, _compute_radius(init_pts)
    
    best_pts = init_pts.copy()
    best_r = _compute_radius(init_pts)
    
    n_anneal = max(1, min(n_anneal, 20))
    
    for step in range(n_anneal):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        # Annealing schedule: exponential increase in beta
        t = step / max(1, n_anneal - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        # Use L-BFGS-B to maximize soft_min_radius (i.e., minimize negative soft_min)
        def objective(x, b=beta):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius(pts, b)
        
        def gradient(x, b=beta):
            pts = x.reshape(-1, 2)
            return -_soft_min_radius_grad(pts, b)
        
        x0 = best_pts.flatten()
        
        try:
            res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < objective(x0):
                new_pts = res.x.reshape(-1, 2)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    return best_pts, best_r


def _nelder_mead_polish(pts, deadline=None):
    """Polish with a local search that pushes overlapping circles apart."""
    if deadline is not None and time.time() > deadline - 5:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Simple local search: move points to resolve conflicts
    # This is a heuristic "push" method
    for iteration in range(100):
        if deadline is not None and time.time() > deadline - 2:
            break
            
        moved = False
        n = len(best_pts)
        
        # Find the most violating pair or boundary
        # We can just iterate through all pairs and push them apart slightly
        # Or use the gradient of the hard min (subgradient)
        
        # Calculate current radius
        r = _compute_radius(best_pts)
        
        # If we have time, do a small step in the direction of the subgradient of -radius
        # Subgradient of min(dists) is the average of gradients of active constraints
        # Active constraints are those close to the min
        
        # Let's just use a very high beta soft-min gradient as an approximation
        beta = 1000.0
        grad = _soft_min_radius_grad(best_pts, beta)
        
        # Normalize gradient
        norm_grad = np.linalg.norm(grad)
        if norm_grad > 1e-10:
            # Take a small step
            step_size = 0.01 * r # Scale step with radius?
            # Or fixed small step
            step_size = 0.005
            new_pts = best_pts + step_size * grad
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
                moved = True
        
        if not moved:
            # Try smaller steps or break
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_pts


def _final_polish(pts, deadline=None):
    """
    Final high-precision polish using a very high beta and tight tolerances.
    """
    if deadline is not None and time.time() > deadline - 3:
        return pts
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Use very high beta for final optimization
    beta = 5000.0
    
    def objective(x, b=beta):
        pts = x.reshape(-1, 2)
        return -_soft_min_radius(pts, b)
    
    def gradient(x, b=beta):
        pts = x.reshape(-1, 2)
        return -_soft_min_radius_grad(pts, b)
    
    x0 = best_pts.flatten()
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50})
        
        if res.success or res.fun < objective(x0):
            new_pts = res.x.reshape(-1, 2)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_pts = new_pts
                best_r = new_r
    except Exception:
        pass
    
    # Additional local search with very small steps
    if deadline is not None and time.time() < deadline - 1:
        for _ in range(50):
            if deadline is not None and time.time() > deadline - 1:
                break
            
            beta = 10000.0
            grad = _soft_min_radius_grad(best_pts, beta)
            norm_grad = np.linalg.norm(grad)
            
            if norm_grad > 1e-10:
                step_size = 0.001
                new_pts = best_pts + step_size * grad
                new_r = _compute_radius(new_pts)
                
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
                else:
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal lattices with different spacings
    for spacing in [0.2, 0.22, 0.24, 0.26]:
        pts = _hex_init(spacing)
        if len(pts) >= N:
            # Take the N closest to center
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
        elif len(pts) > 0:
            inits.append(pts)
    
    # Ring initialization
    ring_pts = _ring_init()
    if len(ring_pts) >= N:
        inits.append(ring_pts[:N])
    elif len(ring_pts) > 0:
        inits.append(ring_pts)
    
    # Random initializations
    for seed in [42, 123, 456]:
        if time.time() > deadline - 20:
            break
        inits.append(_random_init(seed))
    
    # Filter valid inits
    valid_inits = []
    for pts in inits:
        if len(pts) == N:
            r = _compute_radius(pts)
            if r > 0:
                valid_inits.append((pts, r))
    
    if not valid_inits:
        # Fallback: simple hex grid
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    # Sort by initial radius
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    best_pts = valid_inits[0][0]
    best_r = valid_inits[0][1]
    
    # Optimize from each initialization
    for i, (init_pts, init_r) in enumerate(valid_inits):
        if time.time() > deadline - 10:
            break
        
        # Skip if already very good
        if init_r > 0.125:
            # Polish directly
            polished = _nelder_mead_polish(init_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
            continue
        
        # Annealing optimization
        # Use a higher beta range for better convergence
        optimized_pts, optimized_r = _optimize_from_init(
            init_pts, 
            beta_start=50.0, 
            beta_end=500.0, 
            n_anneal=15,
            deadline=deadline
        )
        
        if optimized_r > best_r:
            best_pts = optimized_pts
            best_r = optimized_r
        
        # Final polish
        if time.time() < deadline - 5:
            polished = _nelder_mead_polish(best_pts, deadline)
            r = _compute_radius(polished)
            if r > best_r:
                best_pts = polished
                best_r = r
    
    # Final high-precision polish
    if time.time() < deadline - 2:
        polished = _final_polish(best_pts, deadline)
        r = _compute_radius(polished)
        if r > best_r:
            best_pts = polished
            best_r = r
    
    return best_pts
# EVOLVE-BLOCK-END
