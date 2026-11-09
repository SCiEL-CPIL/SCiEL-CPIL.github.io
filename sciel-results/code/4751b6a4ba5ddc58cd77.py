"""Combined approach: hexagonal lattice init + soft-min smoothed optimization via L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 300


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
    upper_idx = np.triu_indices(N, k=1)
    pairwise_flat = pairwise_vals[upper_idx]
    
    # Combine all constraints
    all_constraints = np.concatenate([boundary_vals, pairwise_flat])
    
    # Soft min of all constraints
    score_approx = _soft_min(all_constraints, beta=beta)
    
    return -score_approx


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


def _local_search_polish(points, max_iter=50, step_size=0.01):
    """
    Local search polish: try small perturbations to improve the score.
    This is a simple hill-climbing approach.
    """
    current_score = _compute_score(points)
    best_points = points.copy()
    best_score = current_score
    
    for iteration in range(max_iter):
        improved = False
        
        # Try random perturbations
        for _ in range(20):
            perturbed = best_points.copy()
            # Perturb a few random points
            n_perturb = np.random.randint(1, 5)
            indices = np.random.choice(N, n_perturb, replace=False)
            perturbed[indices] += np.random.randn(n_perturb, 2) * step_size
            
            # Keep points inside the unit circle
            norms = np.linalg.norm(perturbed, axis=1)
            for i in range(len(perturbed)):
                if norms[i] > 1.0:
                    perturbed[i] = perturbed[i] / norms[i]
            
            new_score = _compute_score(perturbed)
            if new_score > best_score + 1e-12:
                best_points = perturbed.copy()
                best_score = new_score
                improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 150s limit, leave 10s buffer
    
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
        if time.time() > deadline - 20:
            break
        
        # Try a few restarts for this beta
        n_restarts = 2
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            
            # For restarts, add small random perturbation
            if restart == 0:
                x_init = best_points.flatten()
            else:
                perturbed = best_points + np.random.randn(N, 2) * 0.01
                # Keep inside unit circle
                norms = np.linalg.norm(perturbed, axis=1)
                for i in range(len(perturbed)):
                    if norms[i] > 1.0:
                        perturbed[i] = perturbed[i] / norms[i]
                x_init = perturbed.flatten()
            
            try:
                result = minimize(
                    _smooth_objective,
                    x_init,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < -best_score:
                    new_points = result.x.reshape(-1, 2)
                    new_score = _compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
            except Exception:
                pass
    
    # Final local search polish
    if time.time() < deadline - 5:
        best_points, best_score = _local_search_polish(best_points, max_iter=30, step_size=0.005)
    
    return best_points


# Test
if __name__ == "__main__":
    points = construct()
    print(f"Shape: {points.shape}")
    print(f"First few points: {points[:5]}")
    
    # Compute score
    score = _compute_score(points)
    print(f"Radius: {score}")
    print(f"Score: {score / 0.0531542991469}")
