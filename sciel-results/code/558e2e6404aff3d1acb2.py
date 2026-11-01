"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(positions):
    """Compute the minimum radius for given positions."""
    n = len(positions)
    # Minimum distance to boundary
    r_boundary = np.min(0.5 - np.abs(positions))
    # Minimum pairwise distance / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_pair:
                min_pair = d
    r_pair = min_pair / 2.0
    return min(r_boundary, r_pair)


def _negative_score(positions):
    return -_score(positions)


def _soft_min(values, beta):
    """Soft minimum via log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    m = np.min(values)
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _soft_score(positions, beta=5000.0):
    """Smooth surrogate for the score."""
    n = len(positions)
    # Boundary terms: 0.5 - |x_i|, 0.5 - |y_i|
    boundary_terms = np.concatenate([
        0.5 - np.abs(positions[:, 0]),
        0.5 - np.abs(positions[:, 1])
    ])
    # Pairwise terms
    pair_terms = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_terms.append(np.linalg.norm(positions[i] - positions[j]) / 2.0)
    pair_terms = np.array(pair_terms)
    all_terms = np.concatenate([boundary_terms, pair_terms])
    return -_soft_min(all_terms, beta)


def _hexagonal_init(n, seed=0):
    """Generate a hexagonal packing initial guess."""
    rng = np.random.RandomState(seed)
    # Hexagonal lattice
    points = []
    # Generate hexagonal grid points
    spacing = 0.2
    for row in range(10):
        for col in range(10):
            x = -0.5 + spacing * (col + 0.5 * (row % 2)) + 0.1
            y = -0.5 + spacing * row * 0.866 + 0.1
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= n:
        # Randomly select n points
        idx = rng.choice(len(points), n, replace=False)
        return points[idx].copy()
    else:
        # Fall back to grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n].copy()


def _grid_init(n, seed=0):
    """Grid initialization with random jitter."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:n].copy()
    # Add small random jitter
    jitter = rng.randn(n, 2) * 0.01
    grid += jitter
    return grid


def _random_init(n, seed=0):
    """Random initialization inside the square."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.4, 0.4, size=(n, 2))


def _polish(positions, beta=8000.0, maxiter=2000, maxfun=10000):
    """High-precision polish using L-BFGS with tight tolerances."""
    result = minimize(
        _soft_score,
        positions,
        method='L-BFGS-B',
        bounds=[(-0.5, 0.5)] * 2 * len(positions),
        options={'maxiter': maxiter, 'maxfun': maxfun, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    return result.x


def _local_search(positions, n_restarts=5, seed=0):
    """Local search with multiple random restarts."""
    rng = np.random.RandomState(seed)
    best_pos = positions.copy()
    best_score = _score(positions)
    
    for i in range(n_restarts):
        # Perturb current best
        noise_scale = 0.02 * (1 - i / n_restarts)
        perturbed = best_pos + rng.randn(*best_pos.shape) * noise_scale
        # Clip to valid region
        perturbed = np.clip(perturbed, -0.45, 0.45)
        
        # Optimize
        result = minimize(
            _soft_score,
            perturbed,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * 2 * len(perturbed),
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        score = _score(result.x)
        if score > best_score:
            best_score = score
            best_pos = result.x
    
    return best_pos, best_score


def construct():
    """Construct the optimal packing of 50 circles in a unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guesses
    inits = []
    inits.append(_hexagonal_init(N, seed=42))
    inits.append(_grid_init(N, seed=123))
    inits.append(_random_init(N, seed=999))
    
    # Try each initial guess
    best_pos = None
    best_score = -np.inf
    
    for init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # First pass: moderate optimization
        result = minimize(
            _soft_score,
            init,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * 2 * N,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        pos = result.x
        score = _score(pos)
        
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
        
        # Local search from this result
        elapsed = time.time() - start_time
        if elapsed < deadline:
            pos_ls, score_ls = _local_search(pos, n_restarts=3, seed=int(elapsed * 1000) % 10000)
            if score_ls > best_score:
                best_score = score_ls
                best_pos = pos_ls.copy()
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline and best_pos is not None:
        polished = _polish(best_pos, beta=8000, maxiter=2000, maxfun=10000)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_pos = polished
    
    # Ensure valid output
    if best_pos is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pos = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Final validation and clipping
    best_pos = np.clip(best_pos, -0.5, 0.5)
    
    return best_pos
# EVOLVE-BLOCK-END
