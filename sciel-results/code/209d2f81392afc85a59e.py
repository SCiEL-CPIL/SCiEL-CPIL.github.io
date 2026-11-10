"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _compute_radius(centers):
    """Compute the valid radius for a set of centers inside the unit circle."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
    r_boundary = np.min(dists_boundary)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    dx = centers[:, 0:1] - centers[0:1, 0].T
    dy = centers[:, 1:2] - centers[0:1, 1].T
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    r = min(r_boundary, r_pair)
    if r <= 0:
        return 0.0
    return r


def _hexagonal_packing():
    """Generate a hexagonal lattice packing as initial guess."""
    # Start with a reasonable spacing
    spacing = 0.25
    while True:
        k = int(1.0 / spacing) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * math.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            # Select N points closest to center for better packing
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists, kind='stable')[:N]
            return points[idx]
        spacing *= 0.95


def _anneal(centers, max_iter=20000, start_temp=0.01, end_temp=0.0001,
             time_limit=100.0):
    """Simulated annealing to improve the packing."""
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (it / max_iter)
        
        # Pick a random point to move
        idx = np.random.randint(N)
        orig = current_centers[idx].copy()
        
        # Random perturbation
        current_centers[idx] = orig + np.random.randn(2) * temp
        
        # Keep inside unit circle
        norm = np.linalg.norm(current_centers[idx])
        if norm > 0.999:
            current_centers[idx] = current_centers[idx] / norm * 0.999
        
        new_radius = _compute_radius(current_centers)
        
        # Accept or reject
        if new_radius >= current_radius or np.random.rand() < math.exp(
                (new_radius - current_radius) / (temp * 0.01 + 1e-10)):
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
        else:
            current_centers[idx] = orig
    
    return best_centers, best_radius


def construct():
    """Construct 50 equal circles inside the unit circle."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # Initial hexagonal packing
    centers = _hexagonal_packing()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Multiple annealing runs with different seeds
    np.random.seed(42)
    
    for run in range(5):
        if time.time() > deadline:
            break
        
        # Perturb initial solution
        perturbed = best_centers + np.random.randn(N, 2) * 0.05
        # Keep inside unit circle
        norms = np.linalg.norm(perturbed, axis=1)
        for i in range(N):
            if norms[i] > 0.999:
                perturbed[i] = perturbed[i] / norms[i] * 0.999
        
        # Anneal
        annealed, radius = _anneal(perturbed, max_iter=15000,
                                   start_temp=0.008, end_temp=0.00005,
                                   time_limit=25.0)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = annealed.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
