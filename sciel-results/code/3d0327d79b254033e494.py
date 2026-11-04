"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-1/2, 1/2]^3."""
    n = len(centers)
    if n == 0:
        return 0.0
    # Wall constraint: min over i,k of (1/2 - |c_ik|)
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall)
    # Pairwise constraint
    if n > 1:
        d = pdist(centers)
        r_pair = np.min(d) / 2.0
    else:
        r_pair = r_wall
    return min(r_wall, r_pair)


def _fcc_lattice_init(n, scale=1.0):
    """Generate FCC lattice points centered at origin, scaled to fit in unit cube."""
    # FCC lattice vectors
    a = 1.0  # lattice parameter
    # Generate FCC points in a box
    points = []
    # Conventional cell has points at corners and face centers
    # For simplicity, generate a grid of FCC unit cells
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
    
    # Number of cells per dimension
    m = int(np.ceil((n / 4) ** (1/3)))
    
    for i in range(m):
        for j in range(m):
            for k in range(m):
                origin = np.array([i, j, k]) * a
                for b in basis:
                    pts = origin + b * a
                    points.append(pts)
    
    points = np.array(points[:n])
    
    # Center and scale to fit in [-1/2, 1/2]^3
    # First center
    points = points - points.mean(axis=0)
    # Scale so max |coord| <= 0.45 (leave margin)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points = points * (0.45 / max_coord)
    
    # If we don't have enough points, add more
    if len(points) < n:
        # Add random points near existing ones
        extra_needed = n - len(points)
        extra = points[:extra_needed] + np.random.normal(0, 0.05, (extra_needed, 3))
        points = np.vstack([points, extra])
        points = points[:n]
    
    return points[:n]


def _bcc_lattice_init(n):
    """Generate BCC lattice points centered at origin."""
    a = 1.0
    basis = np.array([[0,0,0], [0.5,0.5,0.5]])
    
    points = []
    m = int(np.ceil((n / 2) ** (1/3)))
    
    for i in range(m):
        for j in range(m):
            for k in range(m):
                origin = np.array([i, j, k]) * a
                for b in basis:
                    pts = origin + b * a
                    points.append(pts)
    
    points = np.array(points[:n])
    
    if len(points) == 0:
        return np.random.uniform(-0.4, 0.4, (n, 3))
    
    points = points - points.mean(axis=0)
    max_coord = np.max(np.abs(points))
    if max_coord > 0:
        points = points * (0.45 / max_coord)
    
    if len(points) < n:
        extra_needed = n - len(points)
        extra = points[:extra_needed] + np.random.normal(0, 0.05, (extra_needed, 3))
        points = np.vstack([points, extra])
        points = points[:n]
    
    return points[:n]


def _random_init(n):
    """Random initialization within the cube."""
    return np.random.uniform(-0.45, 0.45, (n, 3))


def _project_to_box(centers, margin=0.0):
    """Project centers to stay within [-1/2+margin, 1/2-margin]^3."""
    return np.clip(centers, -0.5 + margin, 0.5 - margin)


def _alternating_projection(centers, max_iter=2000, tol=1e-10):
    """
    Alternating projection to maximize minimum distance.
    We iteratively:
    1. Project onto box constraints
    2. Move points apart to increase pairwise distances
    """
    n = len(centers)
    centers = centers.copy()
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iter):
        # Compute current pairwise distances
        if n > 1:
            d = pdist(centers)
        else:
            d = np.array([1.0])
        
        min_d = np.min(d)
        
        # Identify closest pairs
        # Use a simple approach: for each point, find its nearest neighbor
        # and push them apart
        
        improved = False
        for i in range(n):
            if n == 1:
                break
            # Find nearest neighbor of point i
            dists = np.array([np.linalg.norm(centers[i] - centers[j]) for j in range(n) if j != i])
            idxs = [j for j in range(n) if j != i]
            min_idx = idxs[np.argmin(dists)]
            min_dist = dists[np.argmin(dists)]
            
            if min_dist < 0.01:
                # Push apart
                direction = centers[i] - centers[min_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-12:
                    direction = direction / norm
                    step = 0.001
                    centers[i] += direction * step
                    centers[min_idx] -= direction * step
                    improved = True
        
        # Project to box
        centers = _project_to_box(centers)
        
        # Compute radius
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
        
        if not improved and it > 100:
            break
    
    return best_centers, best_radius


def _local_search(centers, max_iter=5000, step_init=0.01, step_min=1e-6, time_limit=100.0):
    """
    Local search using gradient-like updates to maximize the minimum radius.
    Uses a smooth surrogate: soft-min of distances.
    """
    n = len(centers)
    centers = centers.copy()
    start_time = time.time()
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = step_init
    patience = 0
    best_patience_radius = best_radius
    
    # Precompute pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute all pairwise distances
        if n > 1:
            d = pdist(centers)
        else:
            d = np.array([1.0])
        
        min_d = np.min(d)
        r_pair = min_d / 2.0
        r_wall = np.min(0.5 - np.max(np.abs(centers), axis=1))
        r = min(r_pair, r_wall)
        
        if r > best_radius + 1e-12:
            best_radius = r
            best_centers = centers.copy()
            patience = 0
            best_patience_radius = r
        else:
            patience += 1
        
        if patience > 500:
            step *= 0.95
            if step < step_min:
                break
            patience = 0
        
        # Compute gradient of soft-min distance
        # Use log-sum-exp for soft-min
        beta = 50.0  # temperature parameter
        if len(d) > 0:
            log_sum = np.log(np.sum(np.exp(-beta * d)))
            weights = np.exp(-beta * d) / np.sum(np.exp(-beta * d))
        else:
            weights = np.array([1.0])
        
        # Gradient of soft-min w.r.t. positions
        grad = np.zeros((n, 3))
        
        # Map pairwise distances to pairs
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                w = weights[pair_idx]
                direction = centers[i] - centers[j]
                dist = np.linalg.norm(direction)
                if dist > 1e-12:
                    unit_dir = direction / dist
                    # Gradient of distance w.r.t. centers[i] is unit_dir
                    # Gradient of soft-min is weighted sum of gradients
                    grad[i] += w * unit_dir
                    grad[j] -= w * unit_dir
                pair_idx += 1
        
        # Wall constraint gradient
        for i in range(n):
            for k in range(3):
                if centers[i, k] > 0:
                    grad[i, k] -= 0.1  # push away from +wall
                elif centers[i, k] < 0:
                    grad[i, k] += 0.1  # push away from -wall
        
        # Update
        centers += step * grad
        centers = _project_to_box(centers)
        
        # Anneal step size
        step *= 0.999
    
    return best_centers, best_radius


def _high_precision_polish(centers, max_iter=10000, time_limit=10.0):
    """
    High-precision polish: use exact local optimization with tight tolerances.
    """
    n = len(centers)
    centers = centers.copy()
    start_time = time.time()
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 0.001
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Find the critical constraint (min distance pair or wall)
        d = pdist(centers)
        min_d = np.min(d)
        r_pair = min_d / 2.0
        r_wall = np.min(0.5 - np.max(np.abs(centers), axis=1))
        r = min(r_pair, r_wall)
        
        if r > best_radius + 1e-12:
            best_radius = r
            best_centers = centers.copy()
        
        # Find critical pair
        min_idx = np.argmin(d)
        # Convert linear index to pair
        pair_list = []
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                pair_list.append((i, j))
                idx += 1
        ci, cj = pair_list[min_idx]
        
        # Push this pair apart
        direction = centers[ci] - centers[cj]
        dist = np.linalg.norm(direction)
        if dist > 1e-12:
            unit_dir = direction / dist
            step_eff = min(step, (min_d * 0.1))
            centers[ci] += unit_dir * step_eff
            centers[cj] -= unit_dir * step_eff
        
        # Also handle wall constraints
        for i in range(n):
            for k in range(3):
                if centers[i, k] > 0.45:
                    centers[i, k] -= 0.001
                elif centers[i, k] < -0.45:
                    centers[i, k] += 0.001
        
        centers = _project_to_box(centers)
        step *= 0.9995
    
    return best_centers, best_radius


def construct():
    """Construct centers of 50 spheres in unit cube to maximize radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(42)
    
    best_centers = None
    best_radius = -1.0
    
    # Try multiple initializations
    inits = []
    inits.append(('fcc', _fcc_lattice_init(N)))
    inits.append(('bcc', _bcc_lattice_init(N)))
    inits.append(('random', _random_init(N)))
    
    # Add perturbed versions
    for name, init in list(inits):
        perturbed = init + np.random.normal(0, 0.05, (N, 3))
        perturbed = _project_to_box(perturbed)
        inits.append((name + '_pert', perturbed))
    
    for init_name, centers in inits:
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Alternating projection
        c1, r1 = _alternating_projection(centers, max_iter=1000, tol=1e-10)
        
        # Phase 2: Local search
        remaining = deadline - time.time() - 10
        if remaining < 5:
            c2, r2 = c1, r1
        else:
            c2, r2 = _local_search(c1, max_iter=3000, time_limit=min(remaining, 60.0))
        
        # Phase 3: High-precision polish
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c3, r3 = _high_precision_polish(c2, max_iter=5000, time_limit=min(remaining, 15.0))
        else:
            c3, r3 = c2, r2
        
        if r3 > best_radius:
            best_radius = r3
            best_centers = c3.copy()
    
    # Final validation
    if best_centers is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        best_radius = _compute_radius(best_centers)
    
    # Ensure valid output
    best_centers = _project_to_box(best_centers)
    
    return best_centers

# EVOLVE-BLOCK-END
