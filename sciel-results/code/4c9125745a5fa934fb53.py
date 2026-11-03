"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n):
    """Generate initial hexagonal lattice positions inside unit disk."""
    # Start with a reasonable spacing
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Keep points that fit with some margin
        margin = spacing * 0.1
        mask = radius <= 1.0 - margin
        points, radius = points[mask], radius[mask]
        if len(points) >= n:
            # Sort by radius to prefer center points
            idx = np.argsort(radius, kind="stable")[:n]
            return points[idx].copy()
        spacing *= 0.95


def _score(points):
    """Compute the packing radius for given points."""
    n = len(points)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _optimize(points, time_limit=115.0):
    """Force-directed optimization with soft-min surrogate."""
    start_time = time.time()
    n = len(points)
    points = points.astype(np.float64).copy()
    
    # Soft-min parameter for smooth surrogate
    beta = 50.0  # Higher = sharper minimum
    
    def soft_min_dist(points, beta):
        """Compute soft-min of pairwise distances using log-sum-exp."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Soft-min: -log(sum(exp(-beta*d))) / beta
        # Use log-sum-exp trick for numerical stability
        min_d = np.min(dists)
        log_sum = np.log(np.sum(np.exp(-beta * (dists - min_d)))) + (-beta * min_d)
        return -log_sum / beta
    
    def force(points, beta):
        """Compute forces for optimization."""
        n = len(points)
        forces = np.zeros_like(points)
        
        # Pairwise repulsion forces
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Use soft repulsion: force ~ 1/dist^2 but capped
        # For efficiency, only consider nearby pairs
        cutoff = 0.15
        mask = (dists < cutoff) & (dists > 1e-8)
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    d = dists[i, j]
                    f = 1.0 / (d * d + 1e-8)
                    f = min(f, 100.0)  # Cap force
                    dir_ij = (points[j] - points[i]) / d
                    forces[i] += f * dir_ij
                    forces[j] -= f * dir_ij
        
        # Boundary force: push points inward if too close to edge
        radii = np.hypot(points[:, 0], points[:, 1])
        for i in range(n):
            if radii[i] > 0.85:
                f = (radii[i] - 0.85) * 2.0
                if radii[i] > 0:
                    forces[i] -= f * points[i] / radii[i]
        
        return forces
    
    # Optimization loop
    lr = 0.01
    best_score = _score(points)
    best_points = points.copy()
    patience = 0
    max_patience = 500
    
    for iteration in range(20000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute forces
        f = force(points, beta)
        
        # Update positions
        points = points + lr * f
        
        # Clip to valid region (slightly inside boundary)
        radii = np.hypot(points[:, 0], points[:, 1])
        for i in range(n):
            if radii[i] > 0.99:
                points[i] *= 0.99 / radii[i]
        
        # Compute score
        score = _score(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            patience = 0
        else:
            patience += 1
        
        # Adaptive learning rate
        if iteration % 100 == 0:
            lr *= 0.95
        
        if patience > max_patience:
            break
    
    return best_points


def construct():
    """Main entry point: generate optimal packing."""
    start_time = time.time()
    
    # Generate initial positions
    points = _hex_init(N)
    
    # Optimize
    points = _optimize(points, time_limit=115.0)
    
    # Final validation and cleanup
    points = np.asarray(points, dtype=np.float64)
    
    # Ensure all points are strictly inside the unit circle
    radii = np.hypot(points[:, 0], points[:, 1])
    for i in range(len(points)):
        if radii[i] >= 1.0:
            points[i] *= 0.999 / radii[i]
    
    # Ensure no coincident points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                points[j] += 1e-8
    
    return points
# EVOLVE-BLOCK-END
