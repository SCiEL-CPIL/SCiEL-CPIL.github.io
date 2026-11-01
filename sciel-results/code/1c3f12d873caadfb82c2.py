"""Optimized point placement for n=14, d=3 minimizing (max/min pairwise distance)^2."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

# Deadline guard: evaluator kills at 150s, we use 140s as safety margin
DEADLINE = 140.0

def _pairwise_distances(points):
    """Compute all pairwise distances for n points in 3D."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dists, 0.0)
    return dists

def _score(points):
    """Compute (max/min pairwise distance)^2."""
    dists = _pairwise_distances(points)
    # Only consider upper triangle to avoid duplicates
    iu = np.triu_indices(N, 1)
    d = dists[iu]
    dmin = d.min()
    dmax = d.max()
    if dmin < 1e-15:
        return np.inf
    return (dmax / dmin) ** 2

def _objective(points):
    """Smooth surrogate objective for optimization.
    Uses log-sum-exp for max and log-sum-exp for min (inverted)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, 0.0)
    
    iu = np.triu_indices(N, 1)
    d_sq = dists_sq[iu]
    
    # Smooth max of distances (use squared distances for efficiency)
    # log-sum-exp for max
    m = d_sq.max()
    log_sum_exp_max = m + np.log(np.sum(np.exp(d_sq - m)))
    
    # Smooth min of distances: we want to maximize min, so minimize -min
    # Use -log-sum-exp of negative values
    m_neg = (-d_sq).max()  # = -d_sq.min()
    log_sum_exp_neg = m_neg + np.log(np.sum(np.exp(-d_sq - m_neg)))
    smooth_min = -log_sum_exp_neg  # This approximates -min(d_sq)
    
    # We want to minimize (max/min)^2 = max^2 / min^2
    # In log space: 2*log(max) - 2*log(min)
    # Using squared distances: log(d_sq_max) - log(d_sq_min)
    # smooth objective: log_sum_exp_max - (-smooth_min) = log_sum_exp_max + smooth_min
    # But we need to be careful with the relationship
    
    # Alternative: directly compute (smooth_max / smooth_min)^2
    # smooth_max_dist = sqrt(log_sum_exp_max) approximately
    # smooth_min_dist = sqrt(exp(-smooth_min)) approximately
    
    # Simpler: just use the ratio of squared distances
    # (max/min)^2 = d_sq_max / d_sq_min
    # Smooth: log(d_sq_max) - log(d_sq_min)
    # log(d_sq_max) ≈ log_sum_exp_max (in log domain of d_sq)
    # log(d_sq_min) ≈ -smooth_min (since smooth_min = -logsumexp(-d_sq))
    
    obj = log_sum_exp_max + smooth_min
    return obj

def _fibonacci_sphere(n, seed=0):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.RandomState(seed)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        r = np.sqrt(max(0.0, 1.0 - z ** 2))
        theta = golden_angle * i
        points[i] = np.array([r * np.cos(theta), r * np.sin(theta), z])
    return points

def _halton_sequence(n, d, base=None):
    """Generate Halton sequence points in [0,1]^d."""
    if base is None:
        base = [2, 3, 5][:d]
    points = np.zeros((n, d))
    for i in range(1, n + 1):
        for j, b in enumerate(base):
            f = 1.0
            result = 0.0
            x = i
            while x > 0:
                f /= b
                result += f * (x % b)
                x //= b
            points[i - 1, j] = result
    return points

def _random_init(n, d, seed=0):
    """Random initialization in unit cube."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, d)

def _optimize_from_init(points, time_limit):
    """Optimize points starting from given initialization."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'method': 'Nelder-Mead', 'options': {'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}}),
        ('BFGS', {'method': 'BFGS', 'options': {'maxiter': 2000, 'gtol': 1e-10}}),
    ]
    
    for method_name, options in strategies:
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = minimize(_objective, points, **options)
            if result.success or result.fun < best_score:
                new_score = _score(result.x)
                if new_score < best_score:
                    best_score = new_score
                    best_points = result.x.copy()
        except Exception:
            pass
    
    # Polish with tighter tolerance
    if time.time() - start_time < time_limit - 5:
        try:
            result = minimize(_objective, best_points, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-14, 'fatol': 1e-16})
            new_score = _score(result.x)
            if new_score < best_score:
                best_score = new_score
                best_points = result.x.copy()
        except Exception:
            pass
    
    return best_points, best_score

def _simulated_annealing(points, time_limit):
    """Simulated annealing to escape local minima."""
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.9995
    min_T = 1e-8
    rng = np.random.RandomState(42)
    
    while time.time() - start_time < time_limit and T > min_T:
        # Generate perturbation
        step_size = T * 0.1
        perturbation = rng.randn(N, D) * step_size
        new_points = current_points + perturbation
        
        new_score = _score(new_points)
        delta = new_score - current_score
        
        # Accept with probability
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-15)):
            current_points = new_points
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = current_points.copy()
        
        T *= cooling_rate
    
    return best_points, best_score

def construct():
    """Main entry point: construct 14 points in 3D minimizing (max/min pairwise dist)^2."""
    start_time = time.time()
    
    # Multiple initialization strategies
    inits = []
    
    # 1. Fibonacci sphere
    inits.append(_fibonacci_sphere(N, seed=0))
    
    # 2. Halton sequence scaled to unit cube
    halton = _halton_sequence(N, D)
    inits.append(halton)
    
    # 3. Random initializations
    for seed in range(5):
        inits.append(_random_init(N, D, seed=seed))
    
    # 4. Regular simplex-like (first 14 vertices of higher-dim simplex projected)
    # For 14 points in 3D, use a grid-based approach
    side = int(np.ceil(N ** (1.0 / D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    inits.append(np.array(grid, dtype=float))
    
    # Optimize each initialization
    best_points = None
    best_score = np.inf
    
    for i, init_points in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed
        if remaining < 10:
            break
        
        # Allocate time per initialization
        time_per_init = min(remaining / max(1, len(inits) - i), 30.0)
        
        try:
            optimized, score = _optimize_from_init(init_points, time_per_init)
            if score < best_score:
                best_score = score
                best_points = optimized.copy()
        except Exception:
            pass
    
    # If time remains, try simulated annealing from best point
    elapsed = time.time() - start_time
    if best_points is not None and DEADLINE - elapsed > 20:
        sa_time = min(DEADLINE - elapsed - 10, 60.0)
        try:
            sa_points, sa_score = _simulated_annealing(best_points, sa_time)
            if sa_score < best_score:
                best_score = sa_score
                best_points = sa_points.copy()
        except Exception:
            pass
    
    # Final polish
    elapsed = time.time() - start_time
    if best_points is not None and DEADLINE - elapsed > 5:
        try:
            result = minimize(_objective, best_points, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            new_score = _score(result.x)
            if new_score < best_score:
                best_points = result.x.copy()
                best_score = new_score
        except Exception:
            pass
    
    if best_points is None:
        # Fallback to grid
        side = int(np.ceil(N ** (1.0 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        best_points = np.array(grid, dtype=float)
    
    return best_points
# EVOLVE-BLOCK-END
