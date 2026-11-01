"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE_MARGIN = 10.0  # seconds before the 150s limit
START_TIME = None


def _compute_score(points):
    """Compute the score: min(min pairwise dist/2, min (1 - |c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance from origin
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    
    r = min(min_pair_dist / 2.0, boundary_margin)
    return r


def _hex_lattice_init(n=N):
    """Generate initial points using hexagonal lattice pattern."""
    # Try different spacings to get at least n points
    for spacing in np.linspace(0.15, 0.25, 20):
        # Generate hexagonal lattice points
        points = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n:
            # Sort by distance from center, take closest n
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists, kind='stable')[:n]
            return points[idx]
    
    # Fallback: random points
    return np.random.uniform(-0.8, 0.8, size=(n, 2))


def _force_optimize(points, max_iters=500, seed=None):
    """Simple force-directed optimization to maximize minimum distance."""
    if seed is not None:
        np.random.seed(seed)
    
    points = points.copy()
    n = len(points)
    
    # Compute pairwise distances
    def min_dist(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        return np.min(d)
    
    best_score = _compute_score(points)
    
    for it in range(max_iters):
        if it % 50 == 0 and time.time() - START_TIME > 140:
            break
        
        # Compute forces: repulsion between close pairs, attraction to maximize radius
        forces = np.zeros_like(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(d, 1e10)
        
        # Repulsive force (inverse square)
        repulsion = diff / (d[:, :, np.newaxis] ** 2 + 1e-6)
        forces += np.sum(repulsion, axis=1)
        
        # Attractive force toward boundary (encourage larger radius)
        dists = np.linalg.norm(points, axis=1)
        attraction = -points / (dists[:, np.newaxis] + 1e-6) * 0.1
        forces += attraction
        
        # Keep inside unit circle
        for i in range(n):
            if np.linalg.norm(points[i]) > 0.95:
                forces[i] -= 0.5 * points[i]
        
        # Update positions
        learning_rate = 0.01 * (1 - it / max_iters)
        points += learning_rate * forces
        
        # Project back inside unit circle
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.98:
                points[i] *= 0.98 / dists[i]
        
        score = _compute_score(points)
        if score > best_score:
            best_score = score
    
    return points, best_score


def _nelder_mead_optimize(points, max_iters=200, seed=None):
    """Nelder-Mead optimization on flattened points array."""
    if seed is not None:
        np.random.seed(seed)
    
    def objective(flat):
        pts = flat.reshape(N, 2)
        return -_compute_score(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-8})
    
    return result.x.reshape(N, 2), -result.fun


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Multiple random restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() - START_TIME > 140:
            break
        
        # Different initialization strategies
        if restart == 0:
            points = _hex_lattice_init(N)
        elif restart == 1:
            # Ring-based initialization
            points = np.zeros((N, 2))
            # Center point
            points[0] = [0, 0]
            # Two rings
            for i in range(1, 16):
                angle = 2 * np.pi * i / 15
                points[i] = [0.45 * np.cos(angle), 0.45 * np.sin(angle)]
            for i in range(16, N):
                angle = 2 * np.pi * (i - 15) / (N - 15)
                points[i] = [0.75 * np.cos(angle), 0.75 * np.sin(angle)]
        elif restart == 2:
            # Random with some structure
            points = np.random.uniform(-0.7, 0.7, size=(N, 2))
            # Add a few on rings
            for i in range(0, N, 10):
                angle = np.random.uniform(0, 2 * np.pi)
                points[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
        elif restart == 3:
            # Concentric rings
            points = np.zeros((N, 2))
            ring_sizes = [1, 6, 12, 18, 13]  # sums to 50
            idx = 0
            for ring_idx, size in enumerate(ring_sizes):
                r = 0.35 * (ring_idx + 1)
                for j in range(size):
                    angle = 2 * np.pi * j / size
                    points[idx] = [r * np.cos(angle), r * np.sin(angle)]
                    idx += 1
        else:
            # Random
            points = np.random.uniform(-0.8, 0.8, size=(N, 2))
        
        # Ensure valid initial points
        dists = np.linalg.norm(points, axis=1)
        for i in range(N):
            if dists[i] > 0.95:
                points[i] *= 0.95 / dists[i]
        
        # Force optimization
        points_f, score_f = _force_optimize(points, max_iters=300, seed=restart)
        if score_f > best_score:
            best_score = score_f
            best_points = points_f.copy()
        
        # Nelder-Mead refinement
        if time.time() - START_TIME > 135:
            break
        points_nm, score_nm = _nelder_mead_optimize(points_f, max_iters=100, seed=restart+100)
        if score_nm > best_score:
            best_score = score_nm
            best_points = points_nm.copy()
    
    if best_points is None:
        best_points = _hex_lattice_init(N)
    
    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
