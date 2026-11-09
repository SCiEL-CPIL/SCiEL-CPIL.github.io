# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
import scipy.linalg

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _cma_es_minimize(points, time_budget, rng, pop_size=50):
    """
    Deterministic CMA-ES to maximize the score (by minimizing negative score).
    """
    n, d = points.shape
    x = points.copy()
    
    # Normalize to unit scale
    norm = np.max(np.linalg.norm(x, axis=1))
    if norm > 1e-10:
        x = x / norm

    # CMA-ES parameters
    sigma = 0.1  # Initial step size
    mu = pop_size // 2  # Number of parents
    weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = int(np.sum(weights[:1]) ** 2 / np.sum(weights ** 2))
    c_sigma = (mueff + 2) / (n + mueff / 2 + 5)
    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    c_1 = 2 / ((n + 1.3) ** 2 + mueff)
    c_mu = min(1 - c_1, 2 * ((mueff - 2 + 2 / mueff) / ((n + 2) ** 2 + mueff)))
    chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))
    
    C = np.eye(n * d)
    p_sigma = np.zeros(n * d)
    p_c = np.zeros(n * d)
    
    start_time = time.time()
    best_x = x.copy()
    best_f = _score(x)
    
    while time.time() - start_time < time_budget:
        # Generate population
        X = np.zeros((pop_size, n * d))
        for i in range(pop_size):
            z = rng.standard_normal(n * d)
            # Use scipy.linalg.cholesky for sqrt(C)
            try:
                L = scipy.linalg.cholesky(C)
            except:
                # If C is not positive definite, reset C
                C = np.eye(n * d)
                p_sigma = np.zeros(n * d)
                p_c = np.zeros(n * d)
                sigma = 0.1
                L = scipy.linalg.cholesky(C)
            X[i] = x + sigma * z @ L.T
        
        # Evaluate fitness
        F = np.zeros(pop_size)
        for i in range(pop_size):
            pts = X[i].reshape(n, d)
            F[i] = -_score(pts)  # Minimize negative score
            
            if -F[i] > best_f:
                best_f = -F[i]
                best_x = pts.copy()
        
        # Sort by fitness
        idx = np.argsort(F)
        X_best = X[idx[:mu]]
        F_best = F[idx[:mu]]
        
        # Update mean
        x_new = np.sum(weights[:, None] * X_best, axis=0)
        
        # Update sigma
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C @ (x_new - x))
        h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
        sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
        
        # Update covariance matrix
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x)
        
        C = (1 + c_1 + c_mu - c_c - c_c * c_c) * C + c_1 * (np.outer(x_new, x_new) - C) + c_mu * np.sum(weights[:, None, None] * np.outer(X_best - x, X_best - x), axis=0) - (c_c + c_c * c_c - c_1) * np.outer(p_c, p_c)
        
        # Symmetrize C
        C = (C + C.T) / 2
        
        x = x_new
        
        # Safety check
        if not np.all(np.isfinite(x)) or sigma < 1e-12:
            break
            
    return best_x, best_f

def _generate_hexagonal_16():
    """
    Generate a hexagonal packing configuration.
    """
    hex_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    dists = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists)
    pts = hex_pts[idx[:16]]
    return pts

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 630s wall-clock, leave 30s margin
    deadline = time.time() + 570

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Hexagonal Packing
    initial_configs.append(_generate_hexagonal_16())
    
    # Config 2: Circle
    angles_c = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts5 = np.column_stack([np.cos(angles_c), np.sin(angles_c)])
    initial_configs.append(pts5)
    
    # Config 3: Random
    initial_configs.append(rng.standard_normal((n, d)))

    # Evaluate and improve each initial config
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # Normalize to unit scale
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # CMA-ES
        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_budget = min(remaining - 10, 150)  # Cap per-config time
        improved_pts, improved_score = _cma_es_minimize(init, time_budget, rng)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # --- Phase 2: Refine the best solution found so far ---
    if best_points is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        if remaining > 5:
            time_budget = min(remaining - 5, 100)
            improved_pts, improved_score = _cma_es_minimize(best_points, time_budget, rng)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
