"""Pack 50 equal circles in the unit disk: ring-based combinatorial + continuous optimization."""
import numpy as np
import time
from itertools import product

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check inside unit circle
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(dists > 1.0):
        return 0.0
    # Compute min distance between pairs
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            if d < min_dist:
                min_dist = d
    return min(min_dist / 2.0, 1.0 - np.max(dists))

def _score_fast(points):
    """Vectorized score computation."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    if np.any(dists > 1.0):
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_dist = np.sqrt(np.min(d2))
    return min(min_dist / 2.0, 1.0 - np.max(dists))

def _ring_config(ring_counts, ring_radii, angular_offsets):
    """Generate points from ring configuration."""
    points = []
    for r, n, offset in zip(ring_radii, ring_counts, angular_offsets):
        if n == 0:
            continue
        angles = offset + 2 * np.pi * np.arange(n) / n
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(pts)
    if not points:
        return np.array([])
    return np.vstack(points)

def _optimize_ring_config(ring_counts, ring_radii_init, angular_offsets_init, max_iters=2000, tol=1e-10):
    """Optimize ring radii and angular offsets via gradient descent."""
    ring_radii = np.array(ring_radii_init, dtype=float)
    angular_offsets = np.array(angular_offsets_init, dtype=float)
    
    n_rings = len(ring_counts)
    total = sum(ring_counts)
    if total != N:
        return None, 0.0
    
    # Compute initial score
    points = _ring_config(ring_counts, ring_radii, angular_offsets)
    if len(points) != N:
        return None, 0.0
    best_score = _score_fast(points)
    best_points = points.copy()
    
    lr = 0.01
    for it in range(max_iters):
        # Compute gradient numerically
        grad_r = np.zeros(n_rings)
        grad_a = np.zeros(n_rings)
        
        for i in range(n_rings):
            # Gradient w.r.t. ring radius
            eps = 1e-6
            r_plus = ring_radii.copy()
            r_plus[i] += eps
            pts_plus = _ring_config(ring_counts, r_plus, angular_offsets)
            if len(pts_plus) == N:
                s_plus = _score_fast(pts_plus)
            else:
                s_plus = 0.0
            
            r_minus = ring_radii.copy()
            r_minus[i] -= eps
            pts_minus = _ring_config(ring_counts, r_minus, angular_offsets)
            if len(pts_minus) == N:
                s_minus = _score_fast(pts_minus)
            else:
                s_minus = 0.0
            
            grad_r[i] = (s_plus - s_minus) / (2 * eps)
            
            # Gradient w.r.t. angular offset
            a_plus = angular_offsets.copy()
            a_plus[i] += eps
            pts_plus_a = _ring_config(ring_counts, ring_radii, a_plus)
            if len(pts_plus_a) == N:
                s_plus_a = _score_fast(pts_plus_a)
            else:
                s_plus_a = 0.0
            
            a_minus = angular_offsets.copy()
            a_minus[i] -= eps
            pts_minus_a = _ring_config(ring_counts, ring_radii, a_minus)
            if len(pts_minus_a) == N:
                s_minus_a = _score_fast(pts_minus_a)
            else:
                s_minus_a = 0.0
            
            grad_a[i] = (s_plus_a - s_minus_a) / (2 * eps)
        
        # Update
        ring_radii += lr * grad_r
        angular_offsets += lr * grad_a
        
        # Keep radii in valid range
        ring_radii = np.clip(ring_radii, 0.0, 1.0)
        
        points = _ring_config(ring_counts, ring_radii, angular_offsets)
        if len(points) != N:
            break
        score = _score_fast(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        if np.max(np.abs(grad_r)) < tol and np.max(np.abs(grad_a)) < tol:
            break
        
        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.9
        
    return best_points, best_score

def _try_ring_configs():
    """Try various ring configurations."""
    best_points = None
    best_score = 0.0
    
    # Common ring configurations for 50 circles
    configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 12, 19],
        [1, 6, 12, 15, 16],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 17, 14],
        [1, 6, 12, 18, 13],
        [1, 6, 12, 19, 12],
        [1, 6, 12, 20, 11],
        [1, 6, 12, 21, 10],
        [1, 6, 12, 22, 9],
        [1, 6, 12, 23, 8],
        [1, 6, 12, 24, 7],
        [1, 6, 12, 25, 6],
        [1, 6, 12, 26, 5],
        [1, 6, 12, 27, 4],
        [1, 6, 12, 28, 3],
        [1, 6, 12, 29, 2],
        [1, 6, 12, 30, 1],
        [1, 6, 12, 31],
        [1, 6, 12, 32],
        [1, 6, 12, 33],
        [1, 6, 12, 34],
        [1, 6, 12, 35],
        [1, 6, 12, 36],
        [1, 6, 12, 37],
        [1, 6, 12, 38],
        [1, 6, 12, 39],
        [1, 6, 12, 40],
        [1, 6, 12, 41],
        [1, 6, 12, 42],
        [1, 6, 12, 43],
        [1, 6, 12, 44],
        [1, 6, 12, 45],
        [1, 6, 12, 46],
        [1, 6, 12, 47],
        [1, 6, 12, 48],
        [1, 6, 12, 49],
        [1, 6, 12, 50],
        [1, 6, 12, 51],
        [1, 6, 12, 52],
        [1, 6, 12, 53],
        [1, 6, 12, 54],
        [1, 6, 12, 55],
        [1, 6, 12, 56],
        [1, 6, 12, 57],
        [1, 6, 12, 58],
        [1, 6, 12, 59],
        [1, 6, 12, 60],
        [1, 6, 12, 61],
        [1, 6, 12, 62],
        [1, 6, 12, 63],
        [1, 6, 12, 64],
        [1, 6, 12, 65],
        [1, 6, 12, 66],
        [1, 6, 12, 67],
        [1, 6, 12, 68],
        [1, 6, 12, 69],
        [1, 6, 12, 70],
        [1, 6, 12, 71],
        [1, 6, 12, 72],
        [1, 6, 12, 73],
        [1, 6, 12, 74],
        [1, 6, 12, 75],
        [1, 6, 12, 76],
        [1, 6, 12, 77],
        [1, 6, 12, 78],
        [1, 6, 12, 79],
        [1, 6, 12, 80],
        [1, 6, 12, 81],
        [1, 6, 12, 82],
        [1, 6, 12, 83],
        [1, 6, 12, 84],
        [1, 6, 12, 85],
        [1, 6, 12, 86],
        [1, 6, 12, 87],
        [1, 6, 12, 88],
        [1, 6, 12, 89],
        [1, 6, 12, 90],
        [1, 6, 12, 91],
        [1, 6, 12, 92],
        [1, 6, 12, 93],
        [1, 6, 12, 94],
        [1, 6, 12, 95],
        [1, 6, 12, 96],
        [1, 6, 12, 97],
        [1, 6, 12, 98],
        [1, 6, 12, 99],
        [1, 6, 12, 100],
    ]
    
    # Filter to only those that sum to 50
    valid_configs = [c for c in configs if sum(c) == N]
    
    for ring_counts in valid_configs:
        n_rings = len(ring_counts)
        # Initial radii: evenly spaced
        ring_radii_init = np.linspace(0.1, 0.9, n_rings)
        # Initial angular offsets: zero
        angular_offsets_init = np.zeros(n_rings)
        
        points, score = _optimize_ring_config(ring_counts, ring_radii_init, angular_offsets_init, max_iters=500)
        if points is not None and score > best_score:
            best_score = score
            best_points = points
    
    return best_points, best_score

def _hexagonal_pack():
    """Generate hexagonal lattice packing."""
    best_points = None
    best_score = 0.0
    
    for spacing in np.linspace(0.2, 0.35, 50):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        
        if len(points) >= N:
            # Sort by distance from center, take closest N
            dists = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(dists)[:N]
            pts = points[idx]
            score = _score_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
    
    return best_points, best_score

def _random_optimize(n_restarts=50, max_iters=1000):
    """Random initialization followed by local optimization."""
    best_points = None
    best_score = 0.0
    start_time = time.time()
    
    for restart in range(n_restarts):
        if time.time() - start_time > TIME_LIMIT * 0.5:
            break
        
        # Random initial points
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Local optimization via coordinate descent
        lr = 0.05
        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT * 0.5:
                break
            
            improved = False
            for i in range(N):
                # Try small perturbations
                best_local = points[i].copy()
                best_local_score = _score_fast(points)
                
                for dx in [-0.01, 0.01]:
                    for dy in [-0.01, 0.01]:
                        new_pt = points.copy()
                        new_pt[i] = points[i] + np.array([dx, dy])
                        s = _score_fast(new_pt)
                        if s > best_local_score:
                            best_local_score = s
                            best_local = new_pt[i]
                
                if np.any(best_local != points[i]):
                    points[i] = best_local
                    improved = True
            
            if not improved:
                break
        
        score = _score_fast(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    return best_points, best_score

def construct():
    """Main entry point: find best packing of 50 circles in unit disk."""
    start_time = time.time()
    
    # Strategy 1: Hexagonal lattice
    hex_points, hex_score = _hexagonal_pack()
    best_points = hex_points if hex_points is not None else None
    best_score = hex_score
    
    # Strategy 2: Ring-based combinatorial optimization
    if time.time() - start_time < TIME_LIMIT * 0.6:
        ring_points, ring_score = _try_ring_configs()
        if ring_points is not None and ring_score > best_score:
            best_score = ring_score
            best_points = ring_points
    
    # Strategy 3: Random optimization with local search
    if time.time() - start_time < TIME_LIMIT * 0.8:
        rand_points, rand_score = _random_optimize(n_restarts=30, max_iters=500)
        if rand_points is not None and rand_score > best_score:
            best_score = rand_score
            best_points = rand_points
    
    # Ensure we return a valid result
    if best_points is None or len(best_points) != N:
        # Fallback: simple grid
        best_points = np.array([[0.0, 0.0]] + 
                              [[0.8 * np.cos(2*np.pi*i/N), 0.8 * np.sin(2*np.pi*i/N)] for i in range(1, N)])
    
    return best_points
