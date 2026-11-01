"""Circle packing in unit square n=32 - Voronoi-based optimization with force-directed relaxation."""
import time
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import minimize

N = 32
DEADLINE = 145.0  # safety margin before 150s kill


def _voronoi_radii(positions):
    """Compute maximum radii for circles at given positions using Voronoi cells.
    
    The radius of each circle is limited by:
    1. Distance to the boundary of [0,1]^2
    2. Half the distance to the nearest other circle (via Voronoi)
    
    We use Voronoi to find the nearest neighbor distances efficiently.
    """
    n = len(positions)
    if n < 3:
        # Fallback for small n
        radii = np.zeros(n)
        for i in range(n):
            min_dist = min(positions[i, 0], 1 - positions[i, 0],
                          positions[i, 1], 1 - positions[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(0, min_dist - 1e-9)
        return radii
    
    # Build Voronoi diagram
    try:
        vor = Voronoi(positions)
    except Exception:
        # Fallback if Voronoi fails
        radii = np.zeros(n)
        for i in range(n):
            min_dist = min(positions[i, 0], 1 - positions[i, 0],
                          positions[i, 1], 1 - positions[i, 1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(0, min_dist - 1e-9)
        return radii
    
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundary
        min_dist = min(positions[i, 0], 1 - positions[i, 0],
                      positions[i, 1], 1 - positions[i, 1])
        
        # Find nearest neighbor via Voronoi
        # The region for vertex i is vor.regions[vor.point_region[i]]
        region = vor.regions[vor.point_region[i]]
        if len(region) > 0:
            # Get the vertices of this Voronoi cell
            for v in region:
                if v < 0:
                    continue  # unbounded edge
                # The vertex v is equidistant from i and some other point(s)
                # The distance from position i to this vertex is a bound
                d_to_vertex = np.linalg.norm(positions[i] - vor.vertices[v])
                # But we need the distance to the nearest OTHER circle
                # Actually, the Voronoi vertex is where the bisector meets
                # The distance to nearest neighbor = min over edges of the cell
                # of the distance to the edge midpoint... 
                # Simpler: just use all pairwise for accuracy since n=32
                pass
        
        # For n=32, O(n^2) is fine and more reliable
        for j in range(n):
            if i != j:
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = min(min_dist, d / 2.0)
        
        radii[i] = max(0, min_dist - 1e-9)
    
    return radii


def _sum_radii(positions):
    """Compute sum of maximum feasible radii for given positions."""
    radii = _voronoi_radii(positions)
    return np.sum(radii)


def _objective_and_grad(x, n):
    """Objective: minimize negative sum of radii.
    
    We use a smooth approximation: for each pair (i,j), the constraint is
    r_i + r_j <= ||p_i - p_j||. The maximum r_i given positions is:
    r_i = min(boundary_dist_i, min_{j!=i} ||p_i - p_j||/2)
    
    The sum is non-smooth at points where the min switches. We use a
    smooth min via softmin.
    """
    positions = x.reshape(n, 2)
    
    # Compute all pairwise distances
    # dist[i,j] = ||p_i - p_j||
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            dist[i, j] = d
            dist[j, i] = d
    
    # Boundary distances
    bdist = np.minimum(
        np.minimum(positions[:, 0], 1.0 - positions[:, 0]),
        np.minimum(positions[:, 1], 1.0 - positions[:, 1])
    )
    
    # For each circle i, r_i = min(bdist_i, min_{j!=i} dist[i,j]/2)
    # Use softmin for smoothness
    alpha = 50.0  # sharpness of softmin
    
    radii = np.zeros(n)
    for i in range(n):
        # Candidate radii: bdist_i and dist[i,j]/2 for all j != i
        candidates = np.array([bdist[i]])
        for j in range(n):
            if i != j:
                candidates = np.append(candidates, dist[i, j] / 2.0)
        
        # Softmin: -logsumexp(-alpha * x) / alpha
        neg_alpha_x = -alpha * candidates
        max_val = np.max(neg_alpha_x)
        logsumexp = max_val + np.log(np.sum(np.exp(neg_alpha_x - max_val)))
        radii[i] = -logsumexp / alpha
    
    return -np.sum(radii)


def _force_directed(positions, max_iter=2000, time_budget=5.0):
    """Force-directed relaxation: circles repel each other and are attracted to center."""
    n = len(positions)
    positions = positions.copy().astype(float)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        radii = _voronoi_radii(positions)
        current_sum = np.sum(radii)
        
        # Compute forces
        forces = np.zeros_like(positions)
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = positions[i] - positions[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-10:
                    d_vec = np.random.randn(2) * 1e-5
                    d = np.linalg.norm(d_vec)
                
                # Desired separation
                desired = radii[i] + radii[j] + 0.02
                if d < desired:
                    # Repulsive force
                    strength = (desired - d) / max(d, 1e-6)
                    forces[i] += strength * d_vec
                    forces[j] -= strength * d_vec
            
            # Mild attraction to center to prevent drift
            to_center = np.array([0.5, 0.5]) - positions[i]
            forces[i] += 0.005 * to_center
        
        # Adaptive step
        step = 0.01
        for _ in range(5):
            new_pos = positions + step * forces
            new_pos = np.clip(new_pos, 0.0, 1.0)
            new_sum = _sum_radii(new_pos)
            if new_sum > current_sum:
                positions = new_pos
                break
            step *= 0.5
        else:
            # No improvement, break
            if step < 1e-6:
                break
    
    return positions


def _optimize_slsqp(positions, time_budget):
    """Use SLSQP to optimize positions."""
    n = N
    x0 = positions.flatten().copy()
    
    bounds = [(0.0, 1.0)] * (2 * n)
    
    start_time = time.time()
    
    def obj(x):
        if time.time() - start_time > time_budget:
            return 1e6  # timeout penalty
        return _objective_and_grad(x, n)
    
    result = minimize(
        obj, x0, method='SLSQP',
        bounds=bounds,
        options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.fun < _objective_and_grad(x0, n):
        return result.x.reshape(n, 2)
    return positions


def _grid_init(n, seed=0):
    """Grid initialization with jitter."""
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side + rng.randn() * 0.01
                y = (j + 0.5) / side + rng.randn() * 0.01
                positions.append([x, y])
    return np.array(positions[:n])


def _hex_init(n, seed=0):
    """Hexagonal initialization."""
    rng = np.random.RandomState(seed)
    r = 1.0 / 6.5
    dx = 2 * r
    dy = np.sqrt(3) * r
    positions = []
    for row in range(8):
        for col in range(8):
            if len(positions) < n:
                x = dx * (col + 0.5 * (row % 2)) + 0.08 + rng.randn() * 0.005
                y = dy * row + 0.08 + rng.randn() * 0.005
                positions.append([x, y])
    return np.array(positions[:n])


def _random_init(n, seed=0):
    """Random initialization in [0.1, 0.9]^2."""
    rng = np.random.RandomState(seed)
    return rng.rand(n, 2) * 0.8 + 0.1


def _is_valid(positions, radii, tol=1e-7):
    """Check if circles are valid (inside [0,1]^2 and non-overlapping)."""
    n = len(positions)
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _fix_validity(positions, radii):
    """Fix any small violations by shrinking radii."""
    n = len(positions)
    for i in range(n):
        # Ensure inside [0,1]^2
        radii[i] = min(radii[i], positions[i, 0], 1 - positions[i, 0],
                      positions[i, 1], 1 - positions[i, 1])
        radii[i] = max(0, radii[i])
    
    # Fix overlaps
    for _ in range(10):
        fixed = True
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                min_dist = radii[i] + radii[j]
                if d < min_dist - 1e-9:
                    fixed = False
                    scale = max(0, (d - 1e-8)) / max(min_dist, 1e-12)
                    radii[i] *= scale
                    radii[j] *= scale
        if fixed:
            break
    
    return radii


def construct():
    """Construct 32 disjoint circles in [0,1]^2 maximizing sum of radii."""
    start_time = time.time()
    
    best_positions = None
    best_radii = None
    best_sum = -np.inf
    
    def update_best(pos, rad):
        nonlocal best_positions, best_radii, best_sum
        s = np.sum(rad)
        if s > best_sum:
            best_sum = s
            best_positions = pos.copy()
            best_radii = rad.copy()
    
    # Try multiple initializations
    inits = [
        lambda: _grid_init(N, seed=42),
        lambda: _hex_init(N, seed=42),
        lambda: _random_init(N, seed=1),
        lambda: _random_init(N, seed=2),
        lambda: _grid_init(N, seed=100),
        lambda: _hex_init(N, seed=100),
    ]
    
    for idx, init_fn in enumerate(inits):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        positions = init_fn()
        radii = _voronoi_radii(positions)
        update_best(positions, radii)
        
        # Force-directed relaxation
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 3:
            positions = _force_directed(positions, max_iter=1000, time_budget=min(4, remaining - 2))
            radii = _voronoi_radii(positions)
            update_best(positions, radii)
        
        # SLSQP optimization
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 5:
            positions = _optimize_slsqp(positions, time_budget=min(8, remaining - 3))
            radii = _voronoi_radii(positions)
            update_best(positions, radii)
    
    # If we have a good result, try to refine with more iterations
    if best_positions is not None:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 5:
            # Additional force-directed pass
            pos = _force_directed(best_positions, max_iter=3000, time_budget=min(10, remaining - 3))
            rad = _voronoi_radii(pos)
            update_best(pos, rad)
        
        # Final SLSQP polish
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 3 and best_positions is not None:
            pos = _optimize_slsqp(best_positions, time_budget=min(5, remaining - 2))
            rad = _voronoi_radii(pos)
            update_best(pos, rad)
    
    # Fallback if nothing worked
    if best_positions is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
        best_positions = np.array([c[:2] for c in cells])
        best_radii = np.array([c[2] for c in cells])
    
    # Final validity fix
    best_radii = _fix_validity(best_positions, best_radii)
    
    result = np.zeros((N, 3))
    result[:, 0] = best_positions[:, 0]
    result[:, 1] = best_positions[:, 1]
    result[:, 2] = best_radii
    
    return result
