import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 14
D = 3
DEADLINE = 115.0  # safety margin before 120s kill

def _pairwise_sq(X):
    """Compute squared pairwise distances (upper triangle)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return sq[iu]

def _objective(params):
    """Objective: (max d / min d)^2 = max(d^2) / min(d^2)."""
    X = params.reshape(N, D)
    sq = _pairwise_sq(X)
    mn = np.min(sq)
    mx = np.max(sq)
    if mn < 1e-12:
        return 1e12
    return mx / mn

def _objective_smooth(params, alpha=50.0):
    """Smooth surrogate using log-sum-exp for max and log for min."""
    X = params.reshape(N, D)
    sq = _pairwise_sq(X)
    # soft-max: exp(alpha * max) approx via log-sum-exp
    a = alpha * sq
    a_max = np.max(a)
    soft_max = np.exp(a_max) + np.log(np.sum(np.exp(a - a_max)))
    # soft-min: -log-sum-exp(-alpha * sq)
    b = -alpha * sq
    b_max = np.max(b)
    soft_min = -(np.exp(b_max) + np.log(np.sum(np.exp(b - b_max))))
    # We want to minimize soft_max / soft_min, but this is tricky.
    # Instead, minimize soft_max - c * soft_min where c is chosen
    # Actually, let's just use the raw objective for reliability
    mn = np.min(sq)
    mx = np.max(sq)
    if mn < 1e-12:
        return 1e12
    return mx / mn

def _icosahedron():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    pts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _make_initialization_14():
    """Create a 14-point initialization: icosahedron + 2 points."""
    ico = _icosahedron()
    # Add 2 points: one at +z, one at -z (along axis through opposite vertices)
    # The icosahedron has vertices at (0, ±1, ±phi) normalized
    # Let's add points along the z-axis
    p13 = np.array([0, 0, 1.0])
    p14 = np.array([0, 0, -1.0])
    # Scale slightly to avoid coinciding with icosahedron vertices
    # Actually, let's place them at a different radius
    # Or better: use a different symmetric structure
    # Let's try: icosahedron vertices scaled by r1, plus 2 points at different radius
    # For now, just concatenate and let optimization fix it
    pts = np.vstack([ico, p13, p14])
    return pts

def _random_init(seed):
    """Random initialization on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(N, D)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    # Add some random radius variation
    radii = 0.8 + 0.4 * rng.rand(N)
    pts = pts * radii
    return pts

def _optimize_from(X0, deadline):
    """Optimize from initial point X0, return best X found."""
    params0 = X0.flatten()
    best_X = X0.copy()
    best_obj = _objective(params0)
    
    # Try multiple optimization methods
    for method in ['L-BFGS-B', 'SLSQP']:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                _objective,
                params0,
                method=method,
                options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_X = result.x.reshape(N, D)
        except Exception:
            pass
    
    return best_X, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_X = None
    best_obj = float('inf')
    
    # Initialization 1: Icosahedron-based
    X0_ico = _make_initialization_14()
    X_ico, obj_ico = _optimize_from(X0_ico, deadline)
    if obj_ico < best_obj:
        best_obj = obj_ico
        best_X = X_ico
    
    if time.time() > deadline:
        return best_X
    
    # Initialization 2: Random with multiple seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        X0_rand = _random_init(seed)
        X_rand, obj_rand = _optimize_from(X0_rand, deadline)
        if obj_rand < best_obj:
            best_obj = obj_rand
            best_X = X_rand
    
    if time.time() > deadline:
        return best_X
    
    # Final polish: run more iterations on the best solution
    if best_X is not None:
        params = best_X.flatten()
        try:
            result = minimize(
                _objective,
                params,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_X = result.x.reshape(N, D)
        except Exception:
            pass
    
    # Ensure distinctness: if any two points are too close, nudge them
    sq = _pairwise_sq(best_X)
    if np.min(sq) < 1e-10:
        # Nudge the closest pair apart
        iu = np.triu_indices(N, k=1)
        min_idx = np.argmin(sq)
        i, j = iu[0][min_idx], iu[1][min_idx]
        # Move point j slightly
        best_X[j] += 1e-5 * np.array([1, 0, 0])
    
    return best_X
