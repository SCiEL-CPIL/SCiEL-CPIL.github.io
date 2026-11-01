"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# We will use a combination of:
# 1. Multiple diverse initializations (hex lattice, ring structures, random with repulsion)
# 2. A robust optimizer that maximizes the minimum distance (radius)
# 3. Deadline guard to ensure we finish in time

def _score(points):
    """Compute the packing radius for given points."""
    # Points must be inside unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0):
        return 0.0
    
    # Compute pairwise distances
    n = len(points)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(points[i,0] - points[j,0], points[i,1] - points[j,1])
            dists[i,j] = d
            dists[j,i] = d
    
    min_pair_dist = np.min(dists[np.triu_indices(n, k=1)])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_lattice_init(n_points):
    """Generate points from hexagonal lattice."""
    points = []
    # Try different spacings
    for spacing in np.linspace(0.15, 0.25, 20):
        k = int(1.0 / spacing) + 2
        for j in range(-k, k+1):
            for i in range(-k, k+1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - spacing/2.0)**2:
                    points.append([x, y])
        if len(points) >= n_points:
            points = np.array(points)
            # Sort by distance from center
            dists = np.hypot(points[:,0], points[:,1])
            idx = np.argsort(dists)
            return points[idx[:n_points]]
    # Fallback: random
    return _random_init(n_points)


def _ring_init(n_points):
    """Generate points on concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    
    # Rings
    ring_radii = [0.3, 0.6, 0.85]
    ring_counts = [6, 12, 18]  # Total: 1 + 6 + 12 + 18 = 37, need 50
    
    # Adjust to get exactly n_points
    while len(points) < n_points:
        r = min(0.95, len(points) * 0.05 + 0.2)
        count = min(n_points - len(points), int(2 * np.pi * r / 0.15))
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:n_points])


def _random_init(n_points, seed=42):
    """Random initialization with rejection sampling to avoid overlaps."""
    rng = np.random.RandomState(seed)
    points = []
    min_dist = 0.15  # Initial minimum distance
    
    # Try to place points
    max_attempts = 5000
    for _ in range(max_attempts):
        if len(points) >= n_points:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x*x + y*y > (1.0 - min_dist/2)**2:
            continue
        
        # Check distance to existing points
        if points:
            pts = np.array(points)
            dists = np.hypot(pts[:,0] - x, pts[:,1] - y)
            if np.min(dists) < min_dist:
                continue
        
        points.append([x, y])
    
    # If we didn't get enough, fill with random
    while len(points) < n_points:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        if x*x + y*y <= 0.9**2:
            points.append([x, y])
    
    return np.array(points[:n_points])


def _optimize(points, time_budget=10.0):
    """Optimize points to maximize packing radius using iterative repulsion."""
    points = points.copy()
    start_time = time.time()
    
    # Adaptive step size
    step = 0.01
    best_points = points.copy()
    best_score = _score(points)
    
    while time.time() - start_time < time_budget:
        n = len(points)
        # Compute forces: repulsion from nearby points, attraction to stay inside
        forces = np.zeros_like(points)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i+1, n):
                d = points[i] - points[j]
                dist = np.hypot(d[0], d[1])
                if dist < 1e-10:
                    # Random nudge
                    d = np.array([rng.uniform(-1,1), rng.uniform(-1,1)])
                    dist = np.hypot(d[0], d[1])
                # Strong repulsion when close
                force_mag = 1.0 / (dist * dist + 1e-6)
                force = force_mag * d / dist
                forces[i] += force
                forces[j] -= force
        
        # Boundary constraint: push inward if too close to edge
        for i in range(n):
            r = np.hypot(points[i,0], points[i,1])
            if r > 0.95:
                # Push toward center
                forces[i] -= 0.5 * points[i] / (r + 1e-10)
        
        # Update positions
        new_points = points + step * forces
        
        # Project back inside unit circle
        new_radii = np.hypot(new_points[:,0], new_points[:,1])
        mask = new_radii > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / new_radii[mask][:, None] * 0.999
        
        # Check if improvement
        new_score = _score(new_points)
        if new_score > best_score:
            best_score = new_score
            best_points = new_points.copy()
            points = new_points
            step *= 1.05  # Increase step slightly
        else:
            step *= 0.9  # Decrease step
            points = best_points.copy()
        
        # Early termination if step is too small
        if step < 1e-8:
            break
    
    return best_points, best_score


def construct():
    """Main function to construct the packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Generate multiple initializations
    initializations = []
    
    # Hex lattice
    init1 = _hex_lattice_init(N)
    initializations.append(init1)
    
    # Ring structure
    init2 = _ring_init(N)
    initializations.append(init2)
    
    # Random with different seeds
    for seed in [1, 2, 3, 4, 5]:
        init = _random_init(N, seed=seed)
        initializations.append(init)
    
    # Optimize each initialization
    best_points = None
    best_score = 0.0
    
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        # Allocate remaining time to this optimization
        remaining = deadline - elapsed - 5.0
        time_per_opt = max(1.0, min(remaining / (len(initializations) - i), 15.0))
        
        optimized, score = _optimize(init, time_budget=time_per_opt)
        
        if score > best_score:
            best_score = score
            best_points = optimized.copy()
    
    # Final check
    if best_points is None:
        best_points = initializations[0]
    
    # Ensure validity
    radii = np.hypot(best_points[:,0], best_points[:,1])
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask][:, None] * 0.999
    
    return best_points
# EVOLVE-BLOCK-END
