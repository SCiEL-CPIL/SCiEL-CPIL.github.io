"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for the given centres."""
    n = len(centres)
    if n < 2:
        return 0.5
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(
        np.minimum(0.5 - centres[:, 0], 0.5 + centres[:, 0]),
        np.minimum(0.5 - centres[:, 1], 0.5 + centres[:, 1])
    )
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)


def _soft_min_penalty(centres, target_r, beta=50.0):
    """
    Smooth penalty using soft-min approximation.
    Instead of hard min, we use a soft sum to get a differentiable objective.
    """
    n = len(centres)
    
    # Pairwise distances
    diffs = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Only consider pairs where dist < 2*target_r (active constraints)
    # Use soft-min: -1/beta * log(sum(exp(-beta * d_ij / (2*r))))
    # But simpler: minimize sum of max(0, 2r - d)^2 with smoothing
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 2 * target_r:
                violation = 2 * target_r - d
                penalty += violation**2
    
    # Boundary constraints
    for i in range(n):
        x, y = centres[i]
        for coord in [x, y]:
            dist_to_edge = min(0.5 - coord, 0.5 + coord)
            if dist_to_edge < target_r:
                violation = target_r - dist_to_edge
                penalty += violation**2
    
    return penalty


def _soft_min_penalty_grad(centres, target_r):
    """
    Gradient of the soft-min penalty function.
    """
    n = len(centres)
    penalty = 0.0
    grad = np.zeros_like(centres)
    
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            diff = centres[i] - centres[j]
            d = np.sqrt(np.sum(diff**2))
            if d < 1e-10:
                d = 1e-10
            if d < 2 * target_r:
                violation = 2 * target_r - d
                penalty += violation**2
                grad[i] += 2 * violation * (-diff / d)
                grad[j] += 2 * violation * (diff / d)
    
    # Boundary
    for i in range(n):
        for axis in range(2):
            coord = centres[i, axis]
            left_dist = 0.5 - coord
            right_dist = 0.5 + coord
            if left_dist < right_dist:
                edge_dist = left_dist
                edge_grad = -1.0
            else:
                edge_dist = right_dist
                edge_grad = 1.0
            
            if edge_dist < target_r:
                violation = target_r - edge_dist
                penalty += violation**2
                grad[i, axis] += 2 * violation * (-edge_grad)
    
    return penalty, grad.flatten()


def _initial_grid():
    """Create an initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_random(seed=42):
    """Create a random initial configuration inside the square."""
    rng = np.random.RandomState(seed)
    margin = 0.05
    centres = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 2))
    return centres


def _repulsion_init(seed=42):
    """
    Initialize using repulsion-based placement.
    Start with a grid and then push points apart using repulsive forces.
    """
    rng = np.random.RandomState(seed)
    centres = _initial_grid()
    
    # Add small random perturbation to break symmetry
    centres += rng.normal(0, 0.001, size=centres.shape)
    
    # Iteratively push points apart
    for iteration in range(50):
        forces = np.zeros_like(centres)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = centres[i] - centres[j]
                d = np.sqrt(np.sum(diff**2))
                if d < 1e-10:
                    d = 1e-10
                # Repulsive force: stronger when closer
                force = diff / (d**3)
                forces[i] += force
        
        # Also push away from boundaries
        for i in range(N):
            x, y = centres[i]
            # Left boundary
            dist_left = x + 0.5
            if dist_left < 0.1:
                forces[i, 0] += 1.0 / max(dist_left, 0.01)
            # Right boundary
            dist_right = 0.5 - x
            if dist_right < 0.1:
                forces[i, 0] -= 1.0 / max(dist_right, 0.01)
            # Bottom boundary
            dist_bottom = y + 0.5
            if dist_bottom < 0.1:
                forces[i, 1] += 1.0 / max(dist_bottom, 0.01)
            # Top boundary
            dist_top = 0.5 - y
            if dist_top < 0.1:
                forces[i, 1] -= 1.0 / max(dist_top, 0.01)
        
        # Update positions with damping
        learning_rate = 0.01 / (1 + iteration * 0.1)
        centres += learning_rate * forces
        centres = np.clip(centres, -0.49, 0.49)
    
    return centres


def _polish(centres, r_target, max_iter=300, ftol=1e-12, gtol=1e-8):
    """
    Polish a configuration to maximize the minimum radius.
    Uses L-BFGS-B to minimize the penalty function.
    """
    def objective_and_grad(x):
        c = x.reshape(N, 2)
        return _soft_min_penalty_grad(c, r_target)
    
    result = minimize(
        objective_and_grad,
        centres.flatten(),
        method='L-BFGS-B',
        jac=True,
        bounds=[(-0.5, 0.5)] * (2 * N),
        options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
    )
    
    new_centres = result.x.reshape(N, 2)
    return new_centres


def _perturbation_search(centres, r_target, n_perturb=5, step_size=0.01, max_iter=200):
    """
    Perturbation search: try small random displacements of a few points
    and re-polish to escape local minima.
    """
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    rng = np.random.RandomState(12345)
    
    for _ in range(n_perturb):
        # Pick 2-5 random points to perturb
        n_points = rng.randint(2, 6)
        indices = rng.choice(N, size=n_points, replace=False)
        
        perturbed = best_centres.copy()
        for idx in indices:
            perturbed[idx] += rng.uniform(-step_size, step_size, size=2)
            perturbed[idx] = np.clip(perturbed[idx], -0.49, 0.49)
        
        # Re-polish
        polished = _polish(perturbed, r_target, max_iter=max_iter, ftol=1e-12, gtol=1e-8)
        r_new = _compute_radius(polished)
        
        if r_new > best_r:
            best_r = r_new
            best_centres = polished.copy()
    
    return best_centres, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s to leave safety margin
    
    best_centres = _initial_grid()
    best_radius = _compute_radius(best_centres)
    
    # Try multiple initial configurations
    r_lower = best_radius
    r_upper = 0.08
    
    # Try several random initializations and optimize
    n_trials = 6
    
    for trial in range(n_trials):
        if time.time() > deadline - 40:
            break
        
        # Choose initial configuration
        if trial == 0:
            centres = _initial_grid()
        elif trial == 1:
            centres = _repulsion_init(seed=42)
        elif trial == 2:
            centres = _repulsion_init(seed=87)
        else:
            centres = _initial_random(seed=trial * 100 + 7)
        
        # Binary search on radius
        lo = r_lower
        hi = r_upper
        
        for bs_iter in range(10):  # 10 iterations of binary search
            if time.time() > deadline - 30:
                break
            
            mid_r = (lo + hi) / 2.0
            
            # Polish with target radius mid_r
            polished = _polish(centres, mid_r, max_iter=150, ftol=1e-11, gtol=1e-7)
            actual_r = _compute_radius(polished)
            
            if actual_r >= mid_r - 1e-9:
                centres = polished
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_centres = polished.copy()
                lo = mid_r
            else:
                hi = mid_r
        
        # After binary search, update lower bound
        if lo > r_lower:
            r_lower = lo
    
    # Final polish: try to improve best_centres with more iterations
    if time.time() < deadline - 15:
        final_r = _compute_radius(best_centres)
        
        # First, do a standard polish
        polished = _polish(best_centres, final_r, max_iter=400, ftol=1e-13, gtol=1e-9)
        r_new = _compute_radius(polished)
        
        if r_new > best_radius:
            best_radius = r_new
            best_centres = polished.copy()
    
    # Perturbation search to escape local minima
    if time.time() < deadline - 10:
        current_r = _compute_radius(best_centres)
        best_centres, best_radius = _perturbation_search(
            best_centres, current_r, n_perturb=8, step_size=0.008, max_iter=150
        )
        
        # Final polish after perturbation
        if time.time() < deadline - 5:
            polished = _polish(best_centres, best_radius, max_iter=300, ftol=1e-13, gtol=1e-10)
            r_final = _compute_radius(polished)
            if r_final > best_radius:
                best_radius = r_final
                best_centres = polished.copy()
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
