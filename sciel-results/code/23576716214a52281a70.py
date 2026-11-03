"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(centers):
    """Compute the packing score (largest common radius)."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check all centers are inside the unit circle
    dists_to_origin = np.sqrt(np.sum(centers**2, axis=1))
    if np.any(dists_to_origin >= 1.0):
        return 0.0
    # Compute all pairwise distances
    min_pair_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_pair_dist:
                min_pair_dist = d
    min_boundary = 1.0 - np.max(dists_to_origin)
    r = min(min_pair_dist / 2.0, min_boundary)
    return r

def _objective(x, N):
    """Objective to minimize: negative of the packing score (we want to maximize).
    We'll use a smooth approximation based on violations."""
    centers = x.reshape(-1, 2)
    # Compute pairwise distances
    min_pair_dist = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_pair_dist:
                min_pair_dist = d
    dists_to_origin = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists_to_origin)
    r = min(min_pair_dist / 2.0, min_boundary)
    return -r

def _gradient(x, N):
    """Compute gradient of the objective (negative score)."""
    centers = x.reshape(-1, 2)
    grad = np.zeros_like(x)
    
    # Find the binding constraint
    min_pair_dist = float('inf')
    min_pair_i, min_pair_j = -1, -1
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_pair_dist:
                min_pair_dist = d
                min_pair_i, min_pair_j = i, j
    
    dists_to_origin = np.sqrt(np.sum(centers**2, axis=1))
    min_boundary = 1.0 - np.max(dists_to_origin)
    min_boundary_idx = np.argmax(dists_to_origin)
    
    r = min(min_pair_dist / 2.0, min_boundary)
    
    # Gradient of -r
    if min_pair_dist / 2.0 < min_boundary:
        # Pair distance is binding
        # d(-r)/d(centers[i]) = -0.5 * d(min_pair_dist)/d(centers[i])
        # d(min_pair_dist)/d(centers[i]) = (centers[i] - centers[j]) / min_pair_dist
        if min_pair_dist > 1e-10:
            diff = centers[min_pair_i] - centers[min_pair_j]
            grad[2*min_pair_i:2*min_pair_i+2] = -0.5 * diff / min_pair_dist
            grad[2*min_pair_j:2*min_pair_j+2] = 0.5 * diff / min_pair_dist
    else:
        # Boundary is binding
        # d(-r)/d(centers[min_boundary_idx]) = -d(min_boundary)/d(centers[min_boundary_idx])
        # d(min_boundary)/d(centers[i]) = -centers[i] / |centers[i]|
        c = centers[min_boundary_idx]
        norm = np.sqrt(np.sum(c**2))
        if norm > 1e-10:
            grad[2*min_boundary_idx:2*min_boundary_idx+2] = c / norm
    
    return grad

def _hex_lattice_init(spacing=None):
    """Generate hexagonal lattice points inside unit circle."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Select N points closest to center
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        spacing *= 0.98

def _random_init():
    """Generate random initial configuration."""
    centers = np.random.uniform(-0.8, 0.8, size=(N, 2))
    # Ensure all are inside unit circle
    dists = np.sqrt(np.sum(centers**2, axis=1))
    scale = 0.9 / np.maximum(dists, 1e-10)
    centers = centers * scale[:, np.newaxis]
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_centers = None
    best_score = 0.0
    
    # Try hex lattice initialization
    try:
        init_centers = _hex_lattice_init()
        x0 = init_centers.flatten()
        
        # Optimize using L-BFGS-B
        result = minimize(_objective, x0, args=(N,), method='L-BFGS-B', 
                         jac=_gradient, options={'maxiter': 1000, 'ftol': 1e-10})
        
        centers = result.x.reshape(-1, 2)
        score = _compute_score(centers)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    except Exception as e:
        pass
    
    if time.time() > deadline - 30:
        if best_centers is not None:
            return best_centers
        return _hex_lattice_init()
    
    # Try multiple random restarts
    for attempt in range(20):
        if time.time() > deadline - 20:
            break
        
        try:
            x0 = _random_init().flatten()
            result = minimize(_objective, x0, args=(N,), method='L-BFGS-B',
                             jac=_gradient, options={'maxiter': 500, 'ftol': 1e-8})
            
            centers = result.x.reshape(-1, 2)
            score = _compute_score(centers)
            
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
        except Exception as e:
            pass
    
    if best_centers is not None:
        return best_centers
    
    # Fallback to hex lattice
    return _hex_lattice_init()
# EVOLVE-BLOCK-END
