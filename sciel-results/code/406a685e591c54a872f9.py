"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize


def _score(centers):
    """Compute the radius score for a set of centers."""
    # Boundary distance
    boundary = np.min(np.min(0.5 - np.abs(centers), axis=1))
    # Pairwise distances
    tree = cKDTree(centers)
    pairs, dists = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        pairwise_min = 1.0
    else:
        # Compute distances for all pairs
        diff = centers[pairs[:, 0]] - centers[pairs[:, 1]]
        dists = np.linalg.norm(diff, axis=1)
        pairwise_min = np.min(dists) / 2
    return min(boundary, pairwise_min)


def _hexagonal_grid(n):
    """Generate a hexagonal-like initial layout for n circles."""
    # For 50 circles, use a hexagonal packing pattern
    # Approximate: 7-7-7-7-7-6-6-3 or similar
    # Let's use a grid-based approach with hexagonal offset
    m = int(np.ceil(np.sqrt(n)))
    # Use a denser grid
    cols = 7
    rows = int(np.ceil(n / cols))
    centers = []
    dx = 1.0 / (cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    for r in range(rows):
        for c in range(cols):
            if len(centers) >= n:
                break
            x = -0.5 + (c + 0.5) * dx + (0.5 * dx if r % 2 else 0)
            y = -0.5 + (r + 0.5) * dy
            # Scale to fit in [-0.5, 0.5]
            x = x * (1.0 / (cols * dx)) * 1.0
            y = y * (1.0 / (rows * dy)) * 1.0
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def _random_init(n, seed=42):
    """Generate random initial layout."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-0.4, 0.4, size=(n, 2))
    return centers


def _simulated_annealing(centers, max_iter=5000, t0=0.01, t_min=1e-6, alpha=0.999):
    """Run simulated annealing to improve the packing."""
    current = centers.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    rng = np.random.RandomState(42)
    
    for i in range(max_iter):
        T = t0 * (alpha ** i)
        if T < t_min:
            break
        
        # Random perturbation
        step_size = 0.01 * (T / t0) ** 0.5
        delta = rng.normal(0, step_size, size=(N, 2))
        candidate = current + delta
        
        # Clip to square
        candidate = np.clip(candidate, -0.5, 0.5)
        
        candidate_score = _score(candidate)
        
        # Acceptance criterion
        if candidate_score > current_score or rng.random() < np.exp((candidate_score - current_score) / T):
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def _local_optimization(centers, max_iter=2000):
    """Local optimization using gradient-free methods."""
    current = centers.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    rng = np.random.RandomState(123)
    
    for i in range(max_iter):
        # Try moving each point slightly
        improved = False
        for j in range(N):
            for attempt in range(5):
                step = 0.005 * (1 - i / max_iter)
                delta = rng.normal(0, step, size=2)
                candidate = current.copy()
                candidate[j] = candidate[j] + delta
                candidate[j] = np.clip(candidate[j], -0.5, 0.5)
                
                candidate_score = _score(candidate)
                if candidate_score > current_score:
                    current = candidate
                    current_score = candidate_score
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score
                    improved = True
                    break
        
        if not improved and i > 100:
            # Stagnation: try larger perturbation
            j = rng.randint(N)
            step = 0.02
            delta = rng.normal(0, step, size=2)
            candidate = current.copy()
            candidate[j] = candidate[j] + delta
            candidate[j] = np.clip(candidate[j], -0.5, 0.5)
            candidate_score = _score(candidate)
            if candidate_score > current_score * 0.95:  # Accept slight degradation
                current = candidate
                current_score = candidate_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 second safety margin
    
    # Try multiple initializations and optimization strategies
    best_centers = None
    best_score = 0.0
    
    # Initialization 1: Hexagonal grid
    centers_hex = _hexagonal_grid(N)
    score_hex = _score(centers_hex)
    if score_hex > best_score:
        best_centers = centers_hex.copy()
        best_score = score_hex
    
    # Initialization 2: Random
    centers_rand = _random_init(N, seed=42)
    score_rand = _score(centers_rand)
    if score_rand > best_score:
        best_centers = centers_rand.copy()
        best_score = score_rand
    
    # Initialization 3: Another random seed
    centers_rand2 = _random_init(N, seed=123)
    score_rand2 = _score(centers_rand2)
    if score_rand2 > best_score:
        best_centers = centers_rand2.copy()
        best_score = score_rand2
    
    # Run simulated annealing on the best initial
    if time.time() < deadline - 60:
        sa_centers, sa_score = _simulated_annealing(best_centers, max_iter=3000)
        if sa_score > best_score:
            best_centers = sa_centers
            best_score = sa_score
    
    # Run local optimization
    if time.time() < deadline - 30:
        lo_centers, lo_score = _local_optimization(best_centers, max_iter=1500)
        if lo_score > best_score:
            best_centers = lo_centers
            best_score = lo_score
    
    # Try SA on other inits if time permits
    if time.time() < deadline - 30:
        for init_centers, init_score in [(centers_hex, score_hex), (centers_rand, score_rand), (centers_rand2, score_rand2)]:
            if time.time() < deadline - 20:
                sa_c, sa_s = _simulated_annealing(init_centers, max_iter=2000)
                if sa_s > best_score:
                    best_centers = sa_c
                    best_score = sa_s
    
    # Final local optimization pass
    if time.time() < deadline - 10:
        lo_centers, lo_score = _local_optimization(best_centers, max_iter=1000)
        if lo_score > best_score:
            best_centers = lo_centers
            best_score = lo_score
    
    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
