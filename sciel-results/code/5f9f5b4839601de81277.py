"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

def _compute_score(points):
    """Compute the maximum radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Min distance between any pair / 2
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_pair_r = min_pair / 2.0
    
    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    min_boundary = 1.0 - np.max(norms)
    
    return min(min_pair_r, min_boundary)

def _soft_min_surrogate(points, beta=100.0):
    """
    Compute a smooth surrogate for the min of:
    - pairwise distances / 2
    - boundary distances (1 - norm)
    using log-sum-exp style soft-min.
    """
    n = len(points)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min of pairwise distances / 2
    pair_vals = dists / 2.0
    # Exclude diagonal (inf)
    flat_vals = pair_vals.flatten()
    flat_vals = flat_vals[~np.isinf(flat_vals)]
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary_vals = 1.0 - norms
    
    # Combine all values
    all_vals = np.concatenate([flat_vals, boundary_vals])
    
    # Soft-min via log-sum-exp
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    soft_min = m - np.log(np.sum(exp_vals)) / beta
    
    return soft_min

def _objective(x, beta=100.0):
    """Objective function for optimization: negative soft-min radius + boundary penalty."""
    pts = x.reshape(N, 2)
    
    # Hard penalty for points outside unit circle
    norms = np.linalg.norm(pts, axis=1)
    boundary_violation = np.maximum(0, norms - 1.0)
    penalty = 100.0 * np.sum(boundary_violation)
    
    # Negative soft-min radius
    radius = _soft_min_surrogate(pts, beta)
    
    return -radius + penalty

def _objective_grad(x, beta=100.0):
    """Gradient of objective function (numerical approximation)."""
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _objective(x, beta)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        grad[i] = (_objective(xp, beta) - f0) / eps
    return grad

def _ring_initialization():
    """
    Generate initial configuration with concentric rings.
    Known optimal packing has: 1 center, then rings of 6, 12, 18, 13 = 50
    """
    points = []
    
    # Center
    points.append([0.0, 0.0])
    
    # Ring 1: 6 circles
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Ring 2: 12 circles
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12  # offset
        points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    
    # Ring 3: 18 circles
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        points.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    
    # Ring 4: 13 circles (partial outer ring)
    r4 = 0.90
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        points.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    return np.array(points)

def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _polish_smooth(points, start_time, deadline, max_beta=500.0):
    """
    Final high-precision polish using smooth surrogate with increasing beta.
    """
    current = points.copy()
    current_score = _compute_score(current)
    
    # Gradually increase beta to sharpen the approximation
    betas = [50, 100, 200, 400, 800]
    
    for beta in betas:
        if time.time() > deadline - 5:
            break
        
        def obj(x):
            return _objective(x, beta)
        
        # Use L-BFGS-B with the smooth objective
        try:
            result = minimize(
                obj,
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_points = result.x.reshape(N, 2)
            
            # Check validity
            if np.all(np.isfinite(new_points)):
                norms = np.linalg.norm(new_points, axis=1)
                if np.max(norms) <= 1.0:
                    new_score = _compute_score(new_points)
                    if new_score > current_score:
                        current = new_points
                        current_score = new_score
        except Exception:
            pass
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try multiple initializations
    initializations = [
        ("ring", _ring_initialization()),
        ("hex", _hexagonal_initial()),
    ]
    
    best_points = None
    best_score = -np.inf
    
    for name, init_points in initializations:
        if time.time() > deadline - 30:
            break
        
        score = _compute_score(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
    
    # Simulated annealing with adaptive step size
    rng = np.random.default_rng(42)
    current_points = best_points.copy()
    current_score = best_score
    
    T = 0.005  # Initial temperature
    T_min = 1e-9
    cooling = 0.9997
    max_iter = 200000
    
    for iteration in range(max_iter):
        if time.time() > deadline - 25:
            break
        
        # Random perturbation with adaptive scale
        idx = rng.integers(0, N)
        scale = max(T, 1e-6) * 0.15
        delta = rng.normal(0, scale, size=2)
        new_points = current_points.copy()
        new_points[idx] += delta
        
        # Ensure points stay in unit circle
        norm = np.linalg.norm(new_points[idx])
        if norm > 1.0:
            new_points[idx] = new_points[idx] / norm * 0.999
        
        new_score = _compute_score(new_points)
        
        # Accept or reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= cooling
        if T < T_min:
            break
    
    # Final high-precision polish with smooth surrogate
    if time.time() < deadline - 10:
        best_points, best_score = _polish_smooth(best_points, start_time, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
