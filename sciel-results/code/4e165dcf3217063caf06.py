"""Rewritten program: hexagonal lattice init + repulsive-force relaxation for 200 equal circles."""
import numpy as np
import time
from scipy.spatial.distance import cdist

N = 200
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _score(points):
    """Compute the radius score for a set of points."""
    # Boundary constraint: min distance from each point to the square boundary
    # Square is [-0.5, 0.5]^2
    boundary_dists = np.minimum(
        np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0]),
        np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
    )
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    return min(min_boundary, min_pair / 2.0)


def _hexagonal_init(n, seed=0):
    """Generate n points on a hexagonal lattice pattern, clipped to the square."""
    rng = np.random.RandomState(seed)
    
    # Hexagonal lattice spacing
    # For dense packing, hexagonal has spacing s with neighbor distance s
    # In a unit square, we want to fit as many as possible
    # Try various spacings and pick the densest valid arrangement
    
    best_points = None
    best_score = -1
    
    for trial in range(20):
        # Random hexagonal lattice with random rotation and offset
        s = 0.035 + rng.uniform(0, 0.02)  # lattice spacing
        angle = rng.uniform(0, 2 * np.pi)
        ox = rng.uniform(-0.5, 0.5)
        oy = rng.uniform(-0.5, 0.5)
        
        # Generate hexagonal lattice points
        # Hexagonal lattice: rows offset by half spacing
        # Basis vectors for hexagonal: (s, 0) and (s/2, s*sqrt(3)/2)
        a1 = np.array([s, 0])
        a2 = np.array([s * 0.5, s * np.sqrt(3) / 2])
        
        # Rotate
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        a1 = R @ a1
        a2 = R @ a2
        
        # Generate grid of lattice points
        # Estimate range needed
        max_range = int(2.0 / s) + 5
        points = []
        for i in range(-max_range, max_range + 1):
            for j in range(-max_range, max_range + 1):
                p = i * a1 + j * a2 + np.array([ox, oy])
                if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                    points.append(p)
        
        points = np.array(points)
        if len(points) < n:
            continue
        
        # Randomly select n points
        idx = rng.choice(len(points), n, replace=False)
        pts = points[idx].copy()
        
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: uniform random
        return rng.uniform(-0.45, 0.45, size=(n, 2))
    
    return best_points


def _relax(points, max_iters=5000, step_size=0.01, decay=0.999):
    """Repulsive force-based relaxation. Push apart overlapping circles and keep inside square."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    
    # Use a simple repulsion model: for each pair closer than 2r_target, push apart
    # But since r is what we're optimizing, use a fixed repulsion radius
    
    # Alternative: use a potential energy approach
    # Energy = sum over pairs of max(0, (d_min - |pi-pj|))^2 + boundary penalty
    # where d_min is the current minimum distance
    
    for it in range(max_iters):
        # Check deadline
        if time.time() > time.time() + 0:  # placeholder
            pass
        
        # Compute pairwise distances
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        # Find the current minimum pair distance
        min_d = np.min(d)
        
        # Also compute boundary distances
        bd = np.minimum(
            np.minimum(0.5 - points[:, 0], 0.5 + points[:, 0]),
            np.minimum(0.5 - points[:, 1], 0.5 + points[:, 1])
        )
        min_bd = np.min(bd)
        
        current_score = min(min_bd, min_d / 2.0)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Compute forces
        # For pairs closer than some threshold, repel
        threshold = min_d * 1.2 if min_d > 0 else 0.04
        forces = np.zeros_like(points)
        
        # Pairwise repulsion for close pairs
        # Use a vectorized approach: for each pair, if dist < threshold, add repulsive force
        # To be efficient, only process pairs within threshold
        # But for N=200, full cdist is fine
        
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < threshold and dist > 1e-12:
                    # Repulsive force: proportional to (threshold - dist)
                    f = (threshold - dist) * diff / dist
                    forces[i] += f
                    forces[j] -= f
        
        # Boundary forces: push back if too close to boundary
        margin = 0.005
        for i in range(N):
            for axis in range(2):
                if points[i, axis] > 0.5 - margin:
                    forces[i, axis] -= (points[i, axis] - (0.5 - margin)) * 0.5
                if points[i, axis] < -0.5 + margin:
                    forces[i, axis] += ((-0.5 + margin) - points[i, axis]) * 0.5
        
        # Update positions
        step = step_size * np.exp(-it * (1 - decay))
        points += step * forces
        
        # Clamp to square with small margin
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-6, 0.5 - 1e-6)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-6, 0.5 - 1e-6)
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    # Try multiple random initializations and relaxations
    best_points = None
    best_score = -1
    
    n_trials = 30  # Number of random restarts to try
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        # Generate hexagonal lattice initialization
        pts = _hexagonal_init(N, seed=trial * 7 + 1)
        
        # Relax
        relaxed = _relax(pts, max_iters=2000, step_size=0.008, decay=0.9995)
        
        sc = _score(relaxed)
        if sc > best_score:
            best_score = sc
            best_points = relaxed.copy()
        
        # Also try the un-relaxed if it's better
        sc_init = _score(pts)
        if sc_init > best_score:
            best_score = sc_init
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
