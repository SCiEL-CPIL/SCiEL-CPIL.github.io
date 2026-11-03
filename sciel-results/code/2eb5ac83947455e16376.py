"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _hexagonal_initial(n=50):
    """Create initial hexagonal lattice points inside unit disk as fallback."""
    best_points = None
    best_score = -1
    
    for base_spacing in [0.22, 0.23, 0.24, 0.25, 0.26]:
        spacing = base_spacing
        for _ in range(10):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n:
                # Select n points with best packing (innermost)
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                pts = points[idx]
                
                # Compute score
                min_dist = 1e10
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(pts[i] - pts[j])
                        if d < min_dist:
                            min_dist = d
                    if np.linalg.norm(pts[i]) > 1.0:
                        min_dist = 0
                score = min(min_dist / 2.0, 1.0 - np.max(np.linalg.norm(pts, axis=1)))
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
            spacing *= 0.97
    
    if best_points is None:
        # Fallback: uniform random
        rng = np.random.default_rng(42)
        best_points = np.zeros((n, 2))
        for i in range(n):
            while True:
                angle = rng.uniform(0, 2*np.pi)
                r = np.sqrt(rng.uniform(0, 1))
                pt = np.array([r*np.cos(angle), r*np.sin(angle)])
                if i == 0 or np.all(np.linalg.norm(pt - best_points[:i], axis=1) > 0.15):
                    best_points[i] = pt
                    break
    return best_points

def _compute_score(points):
    """Compute the packing score: min(min_pair_dist/2, min(1-|c_i|))."""
    n = len(points)
    if n == 0:
        return 0.0
    
    # Min distance from any center to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Min pairwise distance
    min_pair = 1e10
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
    
    return min(min_pair / 2.0, min_boundary)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Ring structure: 1 center + rings of 6, 12, 12, 19
    ring_sizes = [1, 6, 12, 12, 19]
    
    def get_centers(params):
        """Generate centers from parameters: [r1, r2, r3, r4, theta1, theta2, theta3, theta4]"""
        centers = [[0.0, 0.0]]
        for i in range(1, len(ring_sizes)):
            r = params[i-1]
            theta_offset = params[4 + i - 1]
            n = ring_sizes[i]
            angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta_offset
            for a in angles:
                centers.append([r * np.cos(a), r * np.sin(a)])
        return np.array(centers)
    
    def objective(params):
        """Negative of the minimum radius (to be minimized)"""
        centers = get_centers(params)
        # Check if all centers are inside unit circle
        dists = np.hypot(centers[:, 0], centers[:, 1])
        if np.any(dists > 1.0):
            return 1e6
        # Compute minimum distance between pairs
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < min_dist:
                    min_dist = d
        # The radius is min(min_pair_dist/2, min(1 - dist_to_origin))
        r_pair = min_dist / 2
        r_boundary = 1.0 - np.max(dists)
        return -min(r_pair, r_boundary)
    
    # Initial guess based on hexagonal packing estimates
    # Rough estimate: r ≈ 0.125
    # Ring radii approximately: 2r, 4r, 6r, 8r but need to fit in unit circle
    initial_params = np.array([0.25, 0.5, 0.75, 0.95, 0, np.pi/6, 0, np.pi/12])
    
    best_result = None
    best_score = -np.inf
    
    # Try multiple initializations to escape local minima
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > deadline - 10:
            break
            
        if trial == 0:
            params0 = initial_params.copy()
        else:
            # Perturb the initial parameters
            params0 = initial_params + rng.normal(0, 0.05, 8)
        
        try:
            res = minimize(objective, params0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x.copy()
        except Exception:
            continue
    
    # Compare with hexagonal lattice
    hex_points = _hexagonal_initial(N)
    hex_score = _compute_score(hex_points)
    
    if best_result is not None:
        ring_centers = get_centers(best_result)
        ring_score = _compute_score(ring_centers)
        
        # Choose the better one
        if ring_score > hex_score:
            return ring_centers
        else:
            return hex_points
    else:
        return hex_points

# EVOLVE-BLOCK-END
