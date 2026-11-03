"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize


def _compute_radii(centers):
    """
    Given centers (n, 2), compute radii such that each circle is tangent to
    its nearest neighbor and boundary, using Delaunay triangulation.
    
    For each circle i, the radius is limited by:
    - Distance to nearest neighbor j: r_i <= dist(i,j) / 2 (if we want them tangent)
    - But actually, in a tight packing, r_i + r_j = dist(i,j) for tangent pairs.
    
    A simpler approach: for each circle, the max radius is half the distance to 
    the nearest other center, AND also bounded by distance to boundary.
    
    However, this doesn't guarantee disjointness if we just use nearest neighbor.
    
    Better approach: Use the fact that in optimal packings, circles are often 
    mutually tangent. We'll compute radii as follows:
    - For each pair of circles that are neighbors in the Delaunay triangulation,
      they should be tangent: r_i + r_j = d_ij
    - This is a system of linear equations, but it's over-determined.
    
    Simpler practical approach: 
    For each circle i, r_i = min over all j != i of (d_ij - r_j) ... but that's circular.
    
    Let's use an iterative approach or a simpler heuristic:
    r_i = 0.5 * min(
        min_{j != i} d_ij,  # half distance to nearest neighbor
        x_i, 1 - x_i, y_i, 1 - y_i  # distance to boundary
    )
    
    This guarantees disjointness (since d_ij >= r_i + r_j if both use this formula? No, not necessarily)
    
    Actually, if r_i = 0.5 * min_j d_ij and r_j = 0.5 * min_k d_jk, then for the pair (i,j) 
    that achieves the minimum for i, we have r_i = d_ij/2. But r_j could be smaller, so r_i + r_j <= d_ij. 
    Wait, that's not right either. If r_i = d_ij/2 and r_j = d_jk/2 where k != i, then r_i + r_j 
    could be > d_ij if d_jk is large.
    
    Let me think differently. The constraint is: for all i != j, r_i + r_j <= d_ij, and 
    r_i <= min(x_i, 1-x_i, y_i, 1-y_i).
    
    To maximize sum of radii, we want r_i as large as possible. This is a linear program:
    maximize sum(r_i)
    subject to: r_i + r_j <= d_ij for all i < j
                 r_i <= b_i for all i (where b_i is distance to boundary)
                 r_i >= 0
    
    But solving an LP for every evaluation during optimization is expensive.
    
    Alternative: Use the Delaunay neighbors only. In optimal packings, non-neighbor circles 
    are not tangent, so we can restrict to Delaunay edges. This gives a smaller LP.
    
    Even simpler: Use a heuristic where r_i = min(b_i, 0.5 * min_{j in neighbors(i)} d_ij)
    This might not be optimal but is fast and ensures disjointness for neighbor pairs.
    For non-neighbor pairs, we need to check and potentially reduce radii.
    
    Let me implement a proper approach:
    1. Compute Delaunay triangulation
    2. For each circle, set initial radius as min(boundary_dist, 0.5 * min_neighbor_dist)
    3. Check all pairs for violations, and if violated, scale down radii
    4. Repeat until no violations
    
    Actually, let's just use the LP approach with Delaunay edges only, solved via scipy.
    """
    n = len(centers)
    if n < 2:
        return np.array([0.0] * n)
    
    # Compute distances to boundary
    boundary_dist = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                               np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Get Delaunay neighbors
    try:
        tri = Delaunay(centers)
        # Build neighbor list
        neighbors = [set() for _ in range(n)]
        for simplex in tri.simplices:
            for i in simplex:
                for j in simplex:
                    if i != j:
                        neighbors[i].add(j)
    except:
        # Fallback: all pairs are neighbors
        neighbors = [set(range(n)) - {i} for i in range(n)]
    
    # Build constraint matrix for LP: r_i + r_j <= d_ij for Delaunay neighbor pairs
    # Also: r_i <= boundary_dist[i], r_i >= 0
    constraints = []
    for i in range(n):
        for j in neighbors[i]:
            if i < j:
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                constraints.append({'type': 'ineq', 'fun': lambda row=row, d=dists[i, j]: d - row @ np.ones(n) * 0})  # placeholder
    
    # This is getting complicated. Let me use a simpler iterative approach.
    
    # Initialize radii
    radii = np.zeros(n)
    for i in range(n):
        min_neighbor_dist = np.inf
        for j in neighbors[i]:
            min_neighbor_dist = min(min_neighbor_dist, dists[i, j])
        radii[i] = min(boundary_dist[i], 0.5 * min_neighbor_dist)
    
    # Iteratively fix violations
    for _ in range(50):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                if radii[i] + radii[j] > dists[i, j] + 1e-10:
                    violated = True
                    # Scale down both radii proportionally
                    excess = radii[i] + radii[j] - dists[i, j]
                    total = radii[i] + radii[j]
                    if total > 0:
                        radii[i] -= excess * (radii[i] / total)
                        radii[j] -= excess * (radii[j] / total)
                    else:
                        radii[i] = dists[i, j] / 2
                        radii[j] = dists[i, j] / 2
        if not violated:
            break
    
    # Ensure non-negative and within boundary
    radii = np.maximum(radii, 0)
    radii = np.minimum(radii, boundary_dist)
    
    return radii


def _objective_and_grad(centers_flat):
    """
    Objective: negative sum of radii (we minimize negative sum to maximize sum)
    Returns (value, gradient)
    """
    centers = centers_flat.reshape(-1, 2)
    radii = _compute_radii(centers)
    value = -np.sum(radii)
    
    # Compute gradient via finite differences
    grad = np.zeros_like(centers_flat)
    eps = 1e-5
    for i in range(len(centers_flat)):
        centers_plus = centers_flat.copy()
        centers_plus[i] += eps
        centers_plus_arr = centers_plus.reshape(-1, 2)
        radii_plus = _compute_radii(centers_plus_arr)
        grad[i] = (-np.sum(radii_plus) - value) / eps
    
    return value, grad


def construct():
    """
    Construct 26 circles with disjoint interiors in the unit square,
    maximizing the sum of radii.
    """
    np.random.seed(42)
    
    # Initialize centers in a grid-like pattern with some randomness
    side = int(np.ceil(np.sqrt(N)))
    centers = np.zeros((N, 2))
    for i in range(N):
        row = i // side
        col = i % side
        centers[i, 0] = (col + 0.5) / side + np.random.randn() * 0.05
        centers[i, 1] = (row + 0.5) / side + np.random.randn() * 0.05
        # Clamp to [0.01, 0.99]
        centers[i] = np.clip(centers[i], 0.01, 0.99)
    
    # Optimize center positions
    # Use L-BFGS-B with bounds
    bounds = [(0.01, 0.99)] * (2 * N)
    
    result = minimize(
        _objective_and_grad,
        centers.flatten(),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    # Also try a few random restarts
    best_centers = result.x.reshape(-1, 2)
    best_value = -result.fun
    
    for restart in range(3):
        np.random.seed(42 + restart * 1000)
        centers_init = np.zeros((N, 2))
        for i in range(N):
            centers_init[i, 0] = np.random.uniform(0.05, 0.95)
            centers_init[i, 1] = np.random.uniform(0.05, 0.95)
        
        result_r = minimize(
            _objective_and_grad,
            centers_init.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if -result_r.fun > best_value:
            best_value = -result_r.fun
            best_centers = result_r.x.reshape(-1, 2)
    
    # Compute final radii
    radii = _compute_radii(best_centers)
    
    # Final check: ensure disjointness and boundary constraints
    for i in range(N):
        # Clamp center to ensure radius fits
        best_centers[i, 0] = np.clip(best_centers[i, 0], radii[i], 1 - radii[i])
        best_centers[i, 1] = np.clip(best_centers[i, 1], radii[i], 1 - radii[i])
    
    # Recompute radii after clamping
    radii = _compute_radii(best_centers)
    
    # Build output
    result_array = np.column_stack([best_centers, radii])
    
    return result_array
# EVOLVE-BLOCK-END
