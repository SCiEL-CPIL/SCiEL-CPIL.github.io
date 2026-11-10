"""Improved packer: grid + random init, L-BFGS-B on soft-min surrogate, multi-restart."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# Deadline guard: return best found before 150s wall-clock (use 130s as safety margin)
DEADLINE = 130.0
T_START = time.time()


def _elapsed():
    return time.time() - T_START


def _remaining():
    return DEADLINE - _elapsed()


def _compute_radius(centers):
    """Compute the maximum common radius for a set of centers in [-0.5, 0.5]^2."""
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Distance to walls
    wall = 0.5 - np.abs(centers).max(axis=0)
    # Actually need per-point min distance to any wall
    wall_dist = np.minimum(0.5 - centers[:, 0], 0.5 + centers[:, 0])
    wall_dist = np.minimum(wall_dist, 0.5 - centers[:, 1])
    wall_dist = np.minimum(wall_dist, 0.5 + centers[:, 1])
    min_wall = wall_dist.min()
    return 0.5 * min(min_pair, min_wall)


def _soft_min(x, t=0.01):
    """Smooth approximation of min(x) = -logsumexp(-x)/t * t ... actually use log-sum-exp."""
    # soft min ≈ -log(sum(exp(-x/t)))/... no, use: soft_min = -t * log(sum(exp(-x/t)))
    # This gives a value slightly less than the true min.
    x = np.asarray(x)
    # For numerical stability
    m = x.min()
    return m - t * np.log(np.sum(np.exp(-(x - m) / t)))


def _objective(x, t=0.01):
    """Minimize negative soft-min radius. Returns a scalar to minimize."""
    n = N
    centers = x.reshape(n, 2)
    # Pairwise distances
    d = cdist(centers, centers)
    np.fill_diagonal(d, np.inf)
    # Distance to walls (per point, min over 4 walls)
    wall = np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1).min(axis=1)
    # Combine all distances
    all_d = np.concatenate([d[~np.eye(n, dtype=bool)], wall])
    # Soft min of all distances
    sm = _soft_min(all_d, t)
    # We want to maximize the min distance, so minimize -sm
    return -sm


def _grad_objective(x, t=0.01):
    """Gradient of the objective (negative soft-min radius)."""
    n = N
    centers = x.reshape(n, 2)
    d = cdist(centers, centers)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    # For soft min: sm = m - t*log(sum(exp(-(x-m)/t)))
    # where m = min(x)
    # Gradient of soft_min w.r.t. x_i = -exp(-(x_i - m)/t) / sum(exp(-(x_j - m)/t))
    # So gradient of -soft_min = +exp(-(x_i-m)/t)/sum(...)
    
    # Pairwise part
    d_masked = d.copy()
    d_masked[~mask] = np.inf
    m_pair = d_masked.min()
    # exp terms
    exp_terms = np.exp(-(d_masked - m_pair) / t)
    exp_terms[~mask] = 0
    s = exp_terms.sum()
    
    # Wall part
    wall = np.stack([
        0.5 - centers[:, 0],
        0.5 + centers[:, 0],
        0.5 - centers[:, 1],
        0.5 + centers[:, 1]
    ], axis=1)
    m_wall = wall.min()
    exp_wall = np.exp(-(wall - m_wall) / t)
    s_wall = exp_wall.sum()
    
    # Combine: total soft min over all distances
    # Actually let's just do numerical gradient for simplicity and robustness
    # Or compute analytically for the combined set
    
    # Let's use a simpler approach: compute gradient numerically with central differences
    # But that's expensive. Let's do analytic for the combined soft-min.
    
    # Combined set of all distances
    # This is getting complex; let's just use finite differences or let L-BFGS-B estimate
    return None  # Let L-BFGS-B use its own gradient estimation


def _make_grid_init():
    """Create a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _make_random_init(seed):
    """Create a random initialization."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(N, 2))
    return centers


def _optimize_from(centers, t=0.01, max_iter=500):
    """Run L-BFGS-B optimization from a starting configuration."""
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        return _objective(x, t)
    
    result = minimize(
        obj, x0, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
    )
    return result.x.reshape(N, 2), result.fun


def construct():
    """Construct centers of N equal circles in [-0.5, 0.5]^2."""
    best_centers = _make_grid_init()
    best_radius = _compute_radius(best_centers)
    
    # List of initial configurations to try
    inits = [
        ("grid", _make_grid_init()),
        ("random0", _make_random_init(42)),
        ("random1", _make_random_init(123)),
        ("random2", _make_random_init(999)),
        ("random3", _make_random_init(7777)),
    ]
    
    # Also try a slightly perturbed grid
    rng = np.random.default_rng(0)
    perturbed = _make_grid_init() + rng.normal(0, 0.01, size=(N, 2))
    inits.append(("perturbed", perturbed))
    
    for name, centers in inits:
        if _remaining() < 15:
            break
        
        # Try with different soft-min temperatures
        for t in [0.02, 0.01, 0.005]:
            if _remaining() < 10:
                break
            try:
                opt_centers, _ = _optimize_from(centers, t=t, max_iter=300)
                r = _compute_radius(opt_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = opt_centers.copy()
            except Exception:
                pass
    
    # Final polish: run L-BFGS-B with very small t on the best result
    if _remaining() > 15:
        try:
            opt_centers, _ = _optimize_from(best_centers, t=0.001, max_iter=500)
            r = _compute_radius(opt_centers)
            if r > best_radius:
                best_radius = r
                best_centers = opt_centers.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
