"""Direct construction: hexagonal lattice + soft-min smoothed optimization via L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(positions):
    """Compute the radius of the best non-overlapping equal circles for given centers."""
    positions = np.asarray(positions)
    if len(positions) != N:
        return 0.0
    if not np.all(np.isfinite(positions)):
        return 0.0
    
    # Boundary constraint: min(1 - |c_i|)
    dists_from_origin = np.linalg.norm(positions, axis=1)
    boundary_radius = np.min(1.0 - dists_from_origin)
    
    if boundary_radius <= 0:
        return 0.0
    
    # Pairwise distances: min_{i<j} |c_i - c_j| / 2
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(boundary_radius, min_pairwise)


def _soft_min(values, beta=50.0):
    """Soft minimum using log-sum-exp trick for numerical stability."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    # Soft min = -soft_max(-values)
    # soft_max(x) = (1/beta) * log(sum(exp(beta*x)))
    # For numerical stability: soft_max(x) = (1/beta) * log(sum(exp(beta*(x - x_max)))) + x_max
    neg_vals = -values
    neg_max = np.max(neg_vals)
    exp_vals = np.exp(beta * (neg_vals - neg_max))
    soft_max_neg = neg_max + np.log(np.sum(exp_vals)) / beta
    return -soft_max_neg


def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate for the negative score.
    Uses soft-min to create a differentiable objective.
    We want to maximize min(boundary_radius, min_pairwise/2)
    So we minimize -soft_min([boundary_radius, min_pairwise/2])
    But soft_min of a single value is just that value.
    Instead, we can think of it as: the score is the minimum over all constraints.
    Let's define: for each pair (i,j), constraint value = dist_ij / 2
    And for each point i, boundary constraint = 1 - |c_i|
    The score = min over all these constraints.
    We want to maximize this, so minimize -score.
    Using soft-min: soft_min(all_constraints) approximates min(all_constraints)
    """
    positions = x.reshape(-1, 2)
    
    # Boundary constraints: 1 - |c_i| for each i
    dists_from_origin = np.linalg.norm(positions, axis=1)
    boundary_vals = 1.0 - dists_from_origin
    
    # Pairwise constraints: dist_ij / 2 for each pair i<j
    # For efficiency, compute all pairwise distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    pairwise_vals = dists / 2.0
    
    # Flatten to get all constraint values
    # Boundary: N values
    # Pairwise: N*(N-1)/2 values, but we have upper triangle from the matrix
    # Actually dists is N x N with inf on diagonal, so upper triangle has N*(N-1)/2 values
    upper_idx = np.triu_indices(N, k=1)
    pairwise_flat = pairwise_vals[upper_idx]
    
    # Combine all constraints
    all_constraints = np.concatenate([boundary_vals, pairwise_flat])
    
    # Soft min of all constraints
    score_approx = _soft_min(all_constraints, beta=beta)
    
    return -score_approx


def _smooth_gradient(x, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    This is complex, so we'll use numerical gradient via finite differences
    or let scipy handle it with a larger beta.
    For now, return None to let scipy use numerical gradients.
    """
    return None


def _generate_hexagonal_lattice(n_target, spacing):
    """Generate hexagonal lattice points within unit circle, select n_target closest to origin."""
    k = int(1.0 / spacing) + 3
    points_list = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                points_list.append([x, y])
    
    if len(points_list) < n_target:
        return None
    
    points = np.array(points_list)
    radii = np.linalg.norm(points, axis=1)
    indices = np.argsort(radii, kind='stable')[:n_target]
    return points[indices]


def _generate_ring_configuration(n_target):
    """Generate concentric ring configuration as alternative initialization."""
    points = [np.array([0.0, 0.0])]
    remaining = n_target - 1
    
    # Optimal ring radii and counts for ~300 circles
    # Based on known packings: center + rings with increasing count
    ring_configs = [
        (6, 0.18),    # 6 circles at r=0.18
        (12, 0.34),   # 12 circles at r=0.34
        (18, 0.49),   # 18 circles at r=0.49
        (24, 0.63),   # 24 circles at r=0.63
        (30, 0.76),   # 30 circles at r=0.76
        (36, 0.87),   # 36 circles at r=0.87
        (42, 0.96),   # 42 circles at r=0.96
    ]
    
    for count, radius in ring_configs:
        if remaining <= 0:
            break
        n_on_ring = min(count, remaining)
        angles = 2 * np.pi * np.arange(n_on_ring) / n_on_ring
        for angle in angles:
            points.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
        remaining -= n_on_ring
    
    # If still need more, add in a final ring
    if remaining > 0:
        radius = 0.97
        angles = 2 * np.pi * np.arange(remaining) / remaining
        for angle in angles:
            points.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    
    return np.array(points[:n_target])


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin: 150s limit, leave 15s buffer
    
    np.random.seed(42)
    
    # Generate initial configurations
    best_points = None
    best_score = -np.inf
    
    # Try multiple hexagonal lattice spacings
    base_spacing = np.sqrt(2.0 * np.sqrt(3.0) / N)
    spacings = np.linspace(base_spacing * 0.85, base_spacing * 1.15, 15)
    
    for spacing in spacings:
        if time.time() > deadline - 30:
            break
        points = _generate_hexagonal_lattice(N, spacing)
        if points is not None:
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    # Also try ring configuration
    ring_points = _generate_ring_configuration(N)
    ring_score = _compute_score(ring_points)
    if ring_score > best_score:
        best_score = ring_score
        best_points = ring_points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        xs = np.linspace(-0.9, 0.9, 20)
        ys = np.linspace(-0.9, 0.9, 20)
        grid = np.array([[x, y] for x in xs for y in ys if x*x + y*y <= 0.95])
        if len(grid) >= N:
            best_points = grid[:N]
            best_score = _compute_score(best_points)
        else:
            # Last resort: random
            best_points = np.random.uniform(-0.9, 0.9, size=(N, 2))
            best_score = _compute_score(best_points)
    
    # Optimization phase with smooth objective
    # Use L-BFGS-B which handles large-scale smooth optimization well
    # Start with moderate beta, then refine with higher beta
    
    beta_schedule = [30.0, 50.0, 80.0, 120.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 15:
            break
        
        # Try a few restarts for this beta
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            if restart == 0:
                x0 = best_points.flatten()
            else:
                # Perturb with decreasing noise
                noise_scale = 0.03 / (restart + 1)
                x0 = (best_points + np.random.normal(0, noise_scale, size=(N, 2))).flatten()
            
            def objective(x):
                if time.time() > deadline - 8:
                    return 0.0
                return _smooth_objective(x, beta=beta)
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8,
                                         'maxls': 20})
                
                new_points = result.x.reshape(-1, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    
            except Exception:
                pass
    
    # Final polish: use the actual (nonsmooth) score with Nelder-Mead on the best solution
    if time.time() < deadline - 5:
        try:
            def exact_objective(x):
                if time.time() > deadline - 3:
                    return -best_score
                positions = x.reshape(-1, 2)
                return -_compute_score(positions)
            
            result = minimize(exact_objective, best_points.flatten(), method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            
            new_points = result.x.reshape(-1, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
        except Exception:
            pass
    
    return best_points

# EVOLVE-BLOCK-END
