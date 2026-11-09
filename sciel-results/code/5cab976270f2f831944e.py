"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

DEADLINE = 115.0  # safety margin before 120s limit

def compute_radius(points):
    """Compute the maximum common radius for a set of points."""
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    wall_dists = np.min(np.stack([
        points[:, 0] + 0.5,
        0.5 - points[:, 0],
        points[:, 1] + 0.5,
        0.5 - points[:, 1],
        points[:, 2] + 0.5,
        0.5 - points[:, 2],
    ], axis=1), axis=1)
    min_wall = np.min(wall_dists)
    dists = cdist(points, points)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_wall, min_pair)

def init_grid(n):
    """Create a near-uniform grid initialization inside the cube."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    # Use a body-centered-like offset for better initial spacing
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= n:
        return grid[:n].copy()
    else:
        # Pad with random if needed
        extra = np.random.uniform(-0.45, 0.45, size=(n - len(grid), 3))
        return np.vstack([grid, extra])

def push_apart(points, tree, target_r, max_iters=50):
    """
    Targeted repair: find the tightest constraints and push them apart.
    Returns modified points.
    """
    pts = points.copy()
    for _ in range(max_iters):
        # Build tree
        tree = cKDTree(pts)
        # Find all pairs closer than 2*target_r + small margin
        margin = 0.001
        pairs = tree.query_pairs(2 * target_r + margin, output_type='ndarray')
        if len(pairs) == 0:
            break
        
        # Also check wall violations
        wall_violations = []
        for i in range(len(pts)):
            for k in range(3):
                if pts[i, k] < -0.5 + target_r:
                    wall_violations.append((i, k, -0.5 + target_r - pts[i, k]))
                if pts[i, k] > 0.5 - target_r:
                    wall_violations.append((i, k, pts[i, k] - (0.5 - target_r)))
        
        # Pick the worst constraint (smallest distance or largest wall violation)
        worst_dist = np.inf
        worst_pair = None
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < worst_dist:
                worst_dist = d
                worst_pair = (i, j)
        
        worst_wall_viol = 0.0
        worst_wall = None
        for i, k, viol in wall_violations:
            if viol > worst_wall_viol:
                worst_wall_viol = viol
                worst_wall = (i, k)
        
        # Decide which to fix: pair or wall
        if worst_pair is not None and (worst_wall is None or worst_dist < 2 * target_r + worst_wall_viol):
            i, j = worst_pair
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-10:
                # Random nudge
                pts[i] += np.random.randn(3) * 0.01
                pts[j] -= np.random.randn(3) * 0.01
            else:
                direction = (pts[i] - pts[j]) / d
                push = (2 * target_r - d) / 2.0 + 0.001
                # Don't push too hard; dampen
                push = min(push, 0.05)
                pts[i] += direction * push
                pts[j] -= direction * push
        elif worst_wall is not None:
            i, k = worst_wall
            if pts[i, k] < 0:
                pts[i, k] += worst_wall_viol + 0.001
            else:
                pts[i, k] -= worst_wall_viol + 0.001
        else:
            break
        
        # Clip to cube
        pts = np.clip(pts, -0.5, 0.5)
    
    return pts

def kmc_search(points, target_r, time_budget, seed=42):
    """
    Kinetic Monte Carlo: random displacements with acceptance based on 
    minimum radius not decreasing (with simulated annealing temperature).
    """
    np.random.seed(seed)
    pts = points.copy()
    current_r = compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    start = time.time()
    T = 0.005  # initial temperature for simulated annealing
    T_min = 1e-6
    cooling = 0.9995
    
    step_size = 0.02
    
    while time.time() - start < time_budget:
        # Pick random sphere
        i = np.random.randint(len(pts))
        
        # Random displacement
        delta = np.random.randn(3) * step_size
        new_pt = pts[i] + delta
        
        # Keep inside cube
        new_pt = np.clip(new_pt, -0.5, 0.5)
        
        # Tentatively move
        old_pt = pts[i].copy()
        pts[i] = new_pt
        
        # Compute new minimum radius (fast approximation using tree)
        tree = cKDTree(pts)
        # Check nearest neighbors
        dists, _ = tree.query(pts, k=3)  # k=3 includes self, so use 2nd and 3rd
        min_pair = np.min(dists[:, 1:]) / 2.0
        wall_dists = np.min(np.stack([
            pts[:, 0] + 0.5, 0.5 - pts[:, 0],
            pts[:, 1] + 0.5, 0.5 - pts[:, 1],
            pts[:, 2] + 0.5, 0.5 - pts[:, 2],
        ], axis=1), axis=1)
        min_wall = np.min(wall_dists)
        new_r = min(min_pair, min_wall)
        
        # Acceptance criterion
        delta_r = new_r - current_r
        if delta_r >= 0 or np.random.rand() < np.exp(delta_r / max(T, T_min)):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            pts[i] = old_pt
        
        # Cool
        T = max(T * cooling, T_min)
        # Occasionally increase step size if stuck
        if np.random.rand() < 0.01:
            step_size = np.random.uniform(0.005, 0.05)
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    
    # Grid initialization
    points = init_grid(N)
    best_pts = points.copy()
    best_r = compute_radius(points)
    
    # Phase 1: KMC search with moderate time budget
    kmc_budget = min(40.0, DEADLINE - (time.time() - start_time) - 30)
    if kmc_budget > 5:
        pts1, r1 = kmc_search(points, best_r, kmc_budget, seed=42)
        if r1 > best_r:
            best_r = r1
            best_pts = pts1
    
    # Phase 2: Push-apart repair on best
    target_r = best_r
    pts2 = push_apart(best_pts, None, target_r, max_iters=100)
    r2 = compute_radius(pts2)
    if r2 > best_r:
        best_r = r2
        best_pts = pts2
    
    # Phase 3: Another KMC round with refined target
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 10:
        pts3, r3 = kmc_search(best_pts, best_r, min(remaining, 50), seed=123)
        if r3 > best_r:
            best_r = r3
            best_pts = pts3
    
    # Phase 4: Final push-apart
    target_r = best_r
    pts4 = push_apart(best_pts, None, target_r, max_iters=200)
    r4 = compute_radius(pts4)
    if r4 > best_r:
        best_r = r4
        best_pts = pts4
    
    # Ensure validity
    result = np.clip(best_pts, -0.5, 0.5)
    
    # Final check
    final_r = compute_radius(result)
    if final_r >= best_r * 0.999:  # allow tiny numerical tolerance
        return result
    else:
        # Fallback to previous best
        return np.clip(best_pts, -0.5, 0.5)
# EVOLVE-BLOCK-END
