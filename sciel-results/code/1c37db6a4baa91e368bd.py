"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import KDTree

def construct():
    """
    Place 130 points on the unit sphere to minimize the covering radius.
    Uses Simulated Annealing to minimize a proxy for the covering radius:
    the sum of exponentials of the nearest-neighbor distances.
    This approach directly targets the reduction of large empty regions (holes)
    rather than just maximizing the minimum pairwise distance.
    """
    
    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return pts / norms

    def covering_radius_proxy(pts):
        """
        Computes a smooth proxy for the covering radius.
        We use the sum of exp(k * d_nearest) where d_nearest is the distance
        to the nearest other point.
        A large d_nearest implies a large empty space around that point.
        Minimizing this sum helps equalize the distribution and reduce the max gap.
        """
        tree = KDTree(pts)
        # Find nearest neighbor (excluding self)
        # query returns distances and indices. The first column is distance to self (0).
        # We need the second nearest (which is the nearest other).
        dists, idxs = tree.query(pts, k=2)
        nearest_dists = dists[:, 1]
        
        # Scale the distances to make the exponential sensitive to large gaps
        # Typical nearest neighbor distance for 130 points on sphere is approx sqrt(4*pi/130) ~ 0.31 rad
        # We want to penalize distances significantly larger than average.
        scale = 10.0 
        # Sum of exp(scale * dist)
        # To avoid overflow, we can shift or use log-sum-exp, but for optimization 
        # the raw sum is a valid scalar objective if we are careful with scales.
        # Actually, minimizing the MAX is better, but it's non-differentiable.
        # Sum of exp is a soft-max approximation.
        return np.sum(np.exp(scale * nearest_dists))

    def rotate_point(point, angle, axis):
        """Rotate a point around an axis by an angle."""
        axis = axis / np.linalg.norm(axis)
        # Rodrigues' rotation formula
        c = np.cos(angle)
        s = np.sin(angle)
        k = axis
        v = point
        v_rot = v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)
        return v_rot

    # Initial configuration
    pts = fibonacci_sphere(N)
    pts = normalize(pts)
    
    best_pts = pts.copy()
    best_score = covering_radius_proxy(pts)
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    alpha = 0.999  # Cooling rate
    max_iters = 5000
    
    rng = np.random.RandomState(42)
    
    for i in range(max_iters):
        if T < T_min:
            break
            
        # Perturb a random subset of points
        n_perturb = rng.randint(1, 5)
        indices = rng.choice(N, size=n_perturb, replace=False)
        
        new_pts = pts.copy()
        for idx in indices:
            # Random rotation
            axis = rng.randn(3)
            angle = rng.uniform(-T, T)
            new_pts[idx] = rotate_point(pts[idx], angle, axis)
            
        new_pts = normalize(new_pts)
        
        # Calculate new score
        new_score = covering_radius_proxy(new_pts)
        
        # Accept or reject
        delta = new_score - best_score
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            pts = new_pts
            if new_score < best_score:
                best_score = new_score
                best_pts = pts.copy()
        
        T *= alpha

    return best_pts
# EVOLVE-BLOCK-END
