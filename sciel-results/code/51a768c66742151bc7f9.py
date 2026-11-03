"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial.distance import cdist


def _score(centers):
    """Compute the radius for a given set of centers. Returns 0 if invalid."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # Min distance to boundary
    r_boundary = 0.5 - np.max(np.abs(centers), axis=1)
    r_min_boundary = np.min(r_boundary)
    if r_min_boundary < 0:
        return 0.0
    # Min pairwise distance / 2
    # For N=50, cdist is fine
    dists = cdist(centers, centers)
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(r_min_boundary, min_pair)


def _objective(x):
    """Negative radius to maximize."""
    centers = x.reshape(N, 2)
    return -_score(centers)


def _initial_grid():
    """Create an initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_jitter(base, sigma=0.01):
    """Add random jitter to base positions."""
    return base + np.random.normal(0, sigma, base.shape)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial grid
    base = _initial_grid()
    best_centers = base.copy()
    best_score = _score(best_centers)
    
    # Try a few random perturbations as starting points for SLSQP
    np.random.seed(42)
    
    # Constraints: centers must be in [-0.5, 0.5]^2
    # Box constraints are handled by bounds in scipy
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Try optimizing from the grid
    def _run_slsqp(x0, max_iter=1000):
        """Run SLSQP optimization from x0."""
        try:
            result = minimize(
                _objective,
                x0.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'disp': False}
            )
            return result.x.reshape(N, 2), result.fun
        except Exception:
            return x0, _score(x0)
    
    # First: optimize from the grid
    if time.time() < deadline - 5.0:
        opt_centers, opt_neg_score = _run_slsqp(best_centers, max_iter=500)
        opt_score = -opt_neg_score
        if opt_score > best_score:
            best_centers = opt_centers
            best_score = opt_score
    
    # Try a few different random starting points
    n_random_starts = 5
    for i in range(n_random_starts):
        if time.time() > deadline - 10.0:
            break
        
        # Different jitter amounts
        sigma = 0.005 * (i + 1)
        x0 = _random_jitter(base, sigma=sigma)
        # Clip to valid range
        x0 = np.clip(x0, -0.49, 0.49)
        
        opt_centers, opt_neg_score = _run_slsqp(x0, max_iter=300)
        opt_score = -opt_neg_score
        if opt_score > best_score:
            best_centers = opt_centers
            best_score = opt_score
    
    # Final polish: run SLSQP with tighter tolerance from the best found
    if time.time() < deadline - 5.0:
        opt_centers, opt_neg_score = _run_slsqp(best_centers, max_iter=1000)
        opt_score = -opt_neg_score
        if opt_score > best_score:
            best_centers = opt_centers
            best_score = opt_score
    
    # Ensure validity
    final_score = _score(best_centers)
    if final_score <= 0:
        # Fallback to grid
        best_centers = base
        final_score = _score(best_centers)
    
    return best_centers
# EVOLVE-BLOCK-END
