"""Circle packing in a unit square: 200 equal circles, maximize common radius."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200
HALF = 0.5

def _score(pts):
    """Compute the radius for a set of points in [-0.5, 0.5]^2."""
    # Boundary constraint
    min_boundary = np.min(np.minimum(np.minimum(0.5 - pts[:, 0], 0.5 + pts[:, 0]),
                                      np.minimum(0.5 - pts[:, 1], 0.5 + pts[:, 1])))
    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(0.1)  # rough upper bound; will refine
    if len(pairs) == 0:
        min_dist = 1.0
    else:
        min_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
        # Also check all pairs to be safe (N=200 is small enough)
        if min_dist > 0.05:  # if the tree-based search might have missed some
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < min_dist:
                        min_dist = d
    return min(min_boundary, min_dist / 2.0)

def _hexagonal_seed(n):
    """Generate a hexagonal packing seed for n circles in the unit square."""
    # Hexagonal packing: rows offset by half spacing
    # Estimate spacing
    # Area per circle in hex packing: sqrt(3)/2 * s^2 where s is spacing
    # For n circles in area 1: s ≈ sqrt(2/(sqrt(3)*n))
    s = np.sqrt(2.0 / (np.sqrt(3.0) * n))
    
    pts = []
    y = 0.0
    row = 0
    while len(pts) < n:
        x_offset = (row % 2) * s * 0.5
        x = x_offset
        while x <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += s
        y += s * np.sqrt(3.0) / 2.0
        row += 1
    
    pts = np.array(pts[:n])
    # Center and scale to fit in [-0.5, 0.5]^2
    pts -= pts.mean(axis=0)
    # Scale to fit
    max_extent = np.max(np.abs(pts))
    if max_extent > 0.5:
        pts *= 0.5 / max_extent
    return pts

def _grid_seed(n):
    """Square grid seed."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _local_optimize(pts, max_iter=5000, tol=1e-10):
    """Local optimization using gradient-free approach with repulsive forces."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Use a simple iterative repulsion method
    # Each point is pushed away from neighbors and pulled toward center
    dt = 0.01
    for it in range(max_iter):
        if it % 100 == 0 and time.time() > _deadline - 5:
            break
            
        forces = np.zeros_like(pts)
        
        # Repulsion from other circles
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random nudge
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                # Force proportional to 1/dist^2 for strong repulsion when close
                force_mag = 1.0 / (dist ** 2)
                forces[i] += force_mag * diff / dist
        
        # Attraction to center (weak)
        for i in range(N):
            forces[i] -= 0.01 * pts[i]
        
        # Boundary repulsion
        for i in range(N):
            for k in range(2):
                if pts[i, k] > HALF - 0.05:
                    forces[i, k] -= (pts[i, k] - (HALF - 0.05)) * 2.0
                if pts[i, k] < -HALF + 0.05:
                    forces[i, k] += ((-HALF + 0.05) - pts[i, k]) * 2.0
        
        # Update positions
        pts += dt * forces
        
        # Clamp to boundaries
        pts = np.clip(pts, -HALF, HALF)
        
        # Compute score
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Check convergence
        if it > 100 and abs(r - best_r) < tol:
            # Decay step size
            dt *= 0.95
            if dt < 1e-8:
                break
    
    return best_pts, best_r

def _simulated_annealing(pts, max_iter=20000, T0=0.01, T_end=1e-6, cooling=0.9995):
    """Simulated annealing for circle packing."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_r = _score(pts)
    T = T0
    
    for it in range(max_iter):
        if time.time() > _deadline - 5:
            break
        
        # Propose a move: pick random point and perturb
        i = np.random.randint(N)
        delta = np.random.randn(2) * T
        new_pts = pts.copy()
        new_pts[i] += delta
        new_pts[i] = np.clip(new_pts[i], -HALF, HALF)
        
        new_r = _score(new_pts)
        
        # Accept if better or with probability exp((new_r - best_r)/T)
        if new_r >= best_r or np.random.rand() < np.exp((new_r - best_r) / T):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        
        T *= cooling
        if T < T_end:
            T = T_end
    
    return best_pts, best_r

_deadline = None

def construct():
    global _deadline
    start_time = time.time()
    _deadline = start_time + 130  # 130s out of 150s budget
    
    # Generate multiple seeds
    seeds = []
    seeds.append(_grid_seed(N))
    seeds.append(_hexagonal_seed(N))
    
    # Add some random perturbations
    for _ in range(5):
        seed = _grid_seed(N) + np.random.randn(N, 2) * 0.01
        seed = np.clip(seed, -HALF, HALF)
        seeds.append(seed)
    
    best_pts = seeds[0]
    best_r = _score(best_pts)
    
    # Local optimization on each seed
    for seed in seeds:
        if time.time() > _deadline - 30:
            break
        pts, r = _local_optimize(seed, max_iter=2000)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Simulated annealing on best result
    if time.time() < _deadline - 20:
        pts, r = _simulated_annealing(best_pts, max_iter=10000)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Final local polish
    if time.time() < _deadline - 5:
        pts, r = _local_optimize(best_pts, max_iter=3000)
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Ensure validity: clamp to boundaries
    best_pts = np.clip(best_pts, -HALF, HALF)
    
    return best_pts
