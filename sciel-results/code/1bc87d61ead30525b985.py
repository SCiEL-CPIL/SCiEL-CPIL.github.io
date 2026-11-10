"""Ring-based packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 300

def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    Optimizes ring radii and angular offsets to maximize the minimum distance.
    """
    
    def build_points(params, counts):
        """
        params: [r_1, phi_1, r_2, phi_2, ...]
        counts: [n_1, n_2, ...] (excluding center)
        Returns: (N, 2) array of points.
        """
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]  # Center
        idx = 1
        for i in range(len(counts)):
            n_i = counts[i]
            r_i = params[2*i]
            phi_i = params[2*i + 1]
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        return points

    def objective(params, counts):
        points = build_points(params, counts)
        # Check boundary
        dists = np.linalg.norm(points, axis=1)
        if np.any(dists >= 1.0):
            return 1.0 # Penalty
        
        # Vectorized pairwise distance
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_pair = np.sqrt(np.min(dist_sq))
        
        min_dist_boundary = 1.0 - np.max(dists)
        
        r_val = min(min_dist_pair / 2.0, min_dist_boundary)
        
        return -r_val

    best_r = 0.0
    best_points = None

    # Define several ring count distributions
    distributions = []
    
    # 1. Linear distribution
    for n_rings in range(6, 12):
        c = 2.0 * (N - 1) / (n_rings * (n_rings + 1))
        counts = []
        for i in range(1, n_rings + 1):
            n_i = int(round(c * i))
            counts.append(n_i)
        diff = (N - 1) - sum(counts)
        counts[-1] += diff
        if counts[-1] < 1:
            continue
        distributions.append(counts)

    # 2. Hexagonal-like distribution
    for k in range(8, 13):
        counts = []
        for i in range(1, k + 1):
            counts.append(6 * i)
        diff = (N - 1) - sum(counts)
        if diff != 0:
            counts[-1] += diff
            if counts[-1] < 1:
                continue
        distributions.append(counts)

    # 3. Dense Core distribution (n_i proportional to i^2)
    for n_rings in range(8, 12):
        denom = sum([i**2 for i in range(1, n_rings+1)])
        c = (N - 1) / denom
        counts = []
        for i in range(1, n_rings + 1):
            n_i = int(round(c * i**2))
            counts.append(n_i)
        diff = (N - 1) - sum(counts)
        counts[-1] += diff
        if counts[-1] < 1:
            continue
        distributions.append(counts)

    # Optimize each distribution
    for counts in distributions:
        n_rings = len(counts)
        init_params = np.zeros(2 * n_rings)
        for i in range(n_rings):
            init_params[2*i] = (i + 1) / (n_rings + 0.5) * 0.9
            init_params[2*i + 1] = 0.0
            
        try:
            result = minimize(objective, init_params, args=(counts,),
                             method='Powell', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            r_val = -result.fun
            if r_val > best_r:
                best_r = r_val
                best_points = build_points(result.x, counts)
        except Exception:
            pass

    # Fallback
    if best_points is None or best_r < 0.04:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_points
# EVOLVE-BLOCK-END
