"""Improved packing for 50 equal circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _make_hex_lattice(n_target, max_radius=1.0, spacing=None):
    """Generate a hexagonal lattice of points, return up to n_target points inside unit disk."""
    if spacing is None:
        spacing = 2.0 / (np.sqrt(n_target) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n_target:
            # Sort by distance from origin, take closest n_target
            r = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(r, kind='stable')[:n_target]
            return pts[idx]
        spacing *= 0.97
        if spacing < 0.001:
            break
    return np.zeros((n_target, 2))


def _compute_score(centers):
    """Compute the radius r for a given set of centers."""
    # Check all points are finite
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check inside unit circle
    dists = np.hypot(centers[:, 0], centers[:, 1])
    if np.any(dists >= 1.0):
        return 0.0
    
    # Min distance to boundary
    r_boundary = np.min(1.0 - dists)
    
    # Min pairwise distance / 2
    n = len(centers)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    
    r_pair = min_pair / 2.0
    
    return min(r_boundary, r_pair)


def _random_init(n, center=(0, 0), radius=0.95):
    """Generate n random points inside a circle of given radius."""
    angles = np.random.uniform(0, 2 * np.pi, n)
    radii = np.random.uniform(0, radius, n)
    xs = center[0] + radii * np.cos(angles)
    ys = center[1] + radii * np.sin(angles)
    return np.column_stack([xs, ys])


def _simulated_annealing(centers, n_iter=20000, t_start=0.05, t_end=0.001, step_size=0.05):
    """Run simulated annealing to improve the packing."""
    centers = centers.copy()
    current_score = _compute_score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    for it in range(n_iter):
        t = t_start * (t_end / t_start) ** (it / n_iter)
        
        # Pick a random point to move
        idx = np.random.randint(n_iter and len(centers) or len(centers))
        idx = np.random.randint(len(centers))
        
        # Random perturbation
        delta_angle = np.random.uniform(0, 2 * np.pi)
        delta_r = np.random.uniform(-step_size, step_size)
        
        new_center = centers[idx].copy()
        new_center[0] += delta_r * np.cos(delta_angle)
        new_center[1] += delta_r * np.sin(delta_angle)
        
        # Keep inside unit circle
        dist = np.hypot(new_center[0], new_center[1])
        if dist > 0.999:
            new_center *= 0.999 / dist
        
        old_center = centers[idx].copy()
        centers[idx] = new_center
        
        new_score = _compute_score(centers)
        
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / t):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            centers[idx] = old_center
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin
    
    # Start with hexagonal lattice
    centers = _make_hex_lattice(N)
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    print(f"Initial hex lattice score: {best_score:.10f}")
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        
        if restart == 0:
            init_centers = centers.copy()
        else:
            init_centers = _random_init(N, radius=0.9)
        
        # Run simulated annealing
        sa_centers, sa_score = _simulated_annealing(
            init_centers,
            n_iter=15000,
            t_start=0.03,
            t_end=0.0005,
            step_size=0.04
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            print(f"Restart {restart}: improved to {best_score:.10f}")
        
        # Check time
        if time.time() - start_time > deadline:
            break
    
    # Try to further improve with more SA iterations if time allows
    if time.time() - start_time < deadline - 5:
        sa_centers, sa_score = _simulated_annealing(
            best_centers,
            n_iter=20000,
            t_start=0.01,
            t_end=0.0001,
            step_size=0.02
        )
        if sa_score > best_score:
            best_score = sa_score
            best_centers = sa_centers.copy()
            print(f"Final refinement: {best_score:.10f}")
    
    # Final check
    final_score = _compute_score(best_centers)
    if final_score <= 0:
        # Fallback to hex lattice
        best_centers = _make_hex_lattice(N)
    
    return best_centers

# EVOLVE-BLOCK-END
