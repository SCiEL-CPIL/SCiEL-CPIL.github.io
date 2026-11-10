"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def construct():
    """
    Use a multi-start local optimization approach:
    1. Initialize points using a ring-based heuristic (concentric rings with angular offsets).
    2. Run gradient-free local optimization (Nelder-Mead) to maximize the minimum distance.
    3. Return the best configuration found.
    """
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    def score(x):
        """
        Compute the score for a given configuration.
        x is a flat array of shape (2*N,) representing N points in 2D.
        Returns the minimum of:
          - half the minimum pairwise distance
          - the minimum distance from any point to the boundary (1 - |p_i|)
        """
        pts = x.reshape(N, 2)
        # Check if any point is outside the unit circle
        radii = np.sqrt(np.sum(pts**2, axis=1))
        if np.any(radii > 1.0):
            return -1.0  # Invalid
        
        # Compute pairwise distances
        # Use a more efficient method: compute all pairwise distances
        # For N=300, this is 300*299/2 ~ 45k pairs, which is manageable
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        min_boundary_dist = 1.0 - np.max(radii)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    def negative_score(x):
        return -score(x)
    
    # Generate initial configurations using different strategies
    best_x = None
    best_score = -1.0
    
    # Strategy 1: Ring-based initialization
    def ring_init(num_rings):
        """Initialize points in concentric rings."""
        pts = []
        for ring in range(num_rings):
            if ring == 0:
                pts.append([0.0, 0.0])
            else:
                # Distribute points evenly on the ring
                # The radius of each ring is chosen to maximize spacing
                r = ring / num_rings * 0.95
                # Number of points on this ring proportional to circumference
                n_pts = max(1, int(2 * np.pi * r * num_rings / 1.5))
                # Cap the number of points to avoid too many on outer rings
                n_pts = min(n_pts, 20)
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                # Add some randomness to angles
                angles += np.random.uniform(-0.1, 0.1, size=n_pts)
                for a in angles:
                    pts.append([r * np.cos(a), r * np.sin(a)])
        
        # Pad or trim to N points
        while len(pts) < N:
            r = np.random.uniform(0, 0.95)
            a = np.random.uniform(0, 2 * np.pi)
            pts.append([r * np.cos(a), r * np.sin(a)])
        
        pts = pts[:N]
        return np.array(pts).flatten()
    
    # Strategy 2: Random initialization with some structure
    def random_init():
        pts = []
        # Place a center point
        pts.append([0.0, 0.0])
        # Place remaining points randomly
        for _ in range(N - 1):
            r = np.sqrt(np.random.uniform(0, 0.95**2))
            a = np.random.uniform(0, 2 * np.pi)
            pts.append([r * np.cos(a), r * np.sin(a)])
        return np.array(pts).flatten()
    
    # Strategy 3: Hexagonal lattice initialization
    def hex_init():
        pts = []
        # Use a hexagonal lattice
        spacing = 0.1
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x**2 + y**2) <= 0.95:
                    pts.append([x, y])
        
        # Pad or trim to N points
        while len(pts) < N:
            r = np.random.uniform(0, 0.95)
            a = np.random.uniform(0, 2 * np.pi)
            pts.append([r * np.cos(a), r * np.sin(a)])
        
        pts = pts[:N]
        return np.array(pts).flatten()
    
    # Try multiple initializations
    initializations = [
        ring_init(8),
        ring_init(10),
        ring_init(12),
        random_init(),
        random_init(),
        hex_init(),
    ]
    
    for x0 in initializations:
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                negative_score,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            s = -result.fun
            if s > best_score:
                best_score = s
                best_x = result.x
        except Exception:
            continue
    
    # Convert best_x back to points
    pts = best_x.reshape(N, 2)
    
    # Final check and cleanup
    radii = np.sqrt(np.sum(pts**2, axis=1))
    # Ensure all points are inside the unit circle
    pts[radii > 1.0] = pts[radii > 1.0] * (1.0 / radii[radii > 1.0][:, np.newaxis])
    
    return pts
# EVOLVE-BLOCK-END
