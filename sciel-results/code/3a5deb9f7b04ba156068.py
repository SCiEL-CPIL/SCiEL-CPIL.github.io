"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh
import time


def _compute_radius(pts):
    """Compute the maximum common radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    # Boundary constraint
    dist_to_origin = np.linalg.norm(pts, axis=1)
    r_boundary = 1.0 - dist_to_origin
    if np.any(r_boundary <= 0):
        return 0.0
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(pts, metric='euclidean')
    if len(d) == 0:
        return r_boundary.min()
    r_pair = d.min() / 2.0
    return min(r_boundary.min(), r_pair)


def _soft_min(values, temperature=0.01):
    """Soft minimum using log-sum-exp trick for stability."""
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return values[0]
    # For soft min, we use -logsumexp(-values/temperature) * temperature
    # But this can be numerically unstable for large arrays.
    # Let's use a simpler approach: weighted average that emphasizes minimum
    values = np.asarray(values)
    if np.any(~np.isfinite(values)):
        return 0.0
    min_val = values.min()
    # Shift to avoid overflow
    shifted = values - min_val
    weights = np.exp(-shifted / temperature)
    weights /= weights.sum()
    return min_val + (shifted * weights).sum()


def _spectral_objective(pts, temperature=0.01):
    """
    Compute a spectral objective based on the Delaunay triangulation.
    We maximize algebraic connectivity (Fiedler value) which promotes
    uniformity and structural rigidity.
    """
    if len(pts) < 3:
        return 0.0
    
    try:
        tri = Delaunay(pts)
    except Exception:
        return 0.0
    
    # Get edges from Delaunay
    edges = np.array(tri.simplices)
    # Extract unique edges
    edge_list = set()
    for simplex in edges:
        for i in range(3):
            j = (i + 1) % 3
            a, b = int(simplex[i]), int(simplex[j])
            if a > b:
                a, b = b, a
            edge_list.add((a, b))
    
    if len(edge_list) == 0:
        return 0.0
    
    edges = np.array(list(edge_list))
    n = len(pts)
    
    # Build adjacency with weights inversely proportional to distance
    row, col = edges[:, 0], edges[:, 1]
    # Compute edge lengths
    d = np.linalg.norm(pts[row] - pts[col], axis=1)
    d = np.maximum(d, 1e-10)
    weights = 1.0 / d
    
    # Build sparse Laplacian
    from scipy.sparse import csr_matrix
    data = np.concatenate([weights, weights])
    row_idx = np.concatenate([row, col])
    col_idx = np.concatenate([col, row])
    A = csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    
    # Laplacian L = D - A
    deg = np.asarray(A.sum(axis=1)).flatten()
    from scipy.sparse import diags
    D = diags(deg)
    L = D - A
    
    # Compute Fiedler value (second smallest eigenvalue)
    try:
        if n > 10:
            eigvals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
            fiedler = sorted(eigvals)[1]
        else:
            eigvals = np.linalg.eigvalsh(L.toarray())
            fiedler = eigvals[1]
    except Exception:
        return 0.0
    
    if not np.isfinite(fiedler) or fiedler < 0:
        return 0.0
    
    return fiedler


def _force_directed_step(pts, alpha=0.01, beta=0.001, gamma=0.005):
    """
    One step of force-directed optimization.
    Forces:
    - Repulsion between all pairs (soft-min based)
    - Attraction to maintain spectral connectivity
    - Boundary repulsion
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Random small perturbation
                forces[i] += np.random.randn(2) * 0.001
                forces[j] -= np.random.randn(2) * 0.001
                continue
            # Repulsive force, stronger when closer
            # Use a soft potential that peaks at small distances
            force_mag = alpha / (dist ** 2 + 1e-6)
            forces[i] += force_mag * d / dist
            forces[j] -= force_mag * d / dist
    
    # Boundary attraction (keep points inside)
    for i in range(n):
        r = np.linalg.norm(pts[i])
        if r > 0.5:
            # Gently pull back toward center
            forces[i] -= gamma * (r - 0.5) * pts[i] / max(r, 1e-10)
    
    # Update positions with damping
    pts_new = pts + 0.01 * forces
    pts_new = np.clip(pts_new, -0.99, 0.99)
    
    return pts_new


def _local_polish(pts, max_iters=200, tol=1e-10):
    """
    Local polish using gradient-based optimization on the smooth surrogate.
    """
    n = len(pts)
    current_r = _compute_radius(pts)
    
    for it in range(max_iters):
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        d = pdist(pts, metric='euclidean')
        if len(d) == 0:
            break
        
        # Find closest pair
        min_d = d.min()
        min_idx = np.argmin(d)
        
        # Convert flat index to pair
        n_pairs = n * (n - 1) // 2
        i = int((np.sqrt(1 + 8 * min_idx) - 1) / 2)
        j = int(min_idx - i * (n - 1 - i) - i)
        
        if min_d < 1e-15:
            break
        
        # Gradient of r with respect to points i and j
        # r = min(d_ij/2, 1-|c_i|, 1-|c_j|, ...)
        # Focus on the closest pair for now
        diff = pts[i] - pts[j]
        dist = np.linalg.norm(diff)
        
        # Move i and j apart slightly
        if dist > 1e-15:
            step = 0.001 * (min_d - current_r * 2) / dist
            if step > 0:
                pts[i] += step * diff / dist
                pts[j] -= step * diff / dist
        
        # Check boundary
        for k in range(n):
            r_k = np.linalg.norm(pts[k])
            if r_k > 1 - current_r:
                # Pull back
                pts[k] *= (1 - current_r - 0.001) / r_k
        
        new_r = _compute_radius(pts)
        if new_r <= current_r:
            # Revert
            break
        current_r = new_r
        
        if current_r > 0.053:
            break
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate hexagonal lattice initial points
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            break
        spacing *= 0.98
    
    # Sort by distance from origin and take N closest
    points = points[np.argsort(radius, kind="stable")[:N]]
    
    # Ensure valid
    current_r = _compute_radius(points)
    best_pts = points.copy()
    best_r = current_r
    
    # Force-directed optimization with spectral guidance
    max_force_iters = 500
    for it in range(max_force_iters):
        if time.time() > deadline - 10.0:
            break
        
        # Compute spectral objective for guidance
        if it % 10 == 0:
            spectral_val = _spectral_objective(points, temperature=0.01)
        
        # Force-directed step
        points = _force_directed_step(points, alpha=0.01, beta=0.001, gamma=0.005)
        
        # Check improvement
        new_r = _compute_radius(points)
        if new_r > best_r:
            best_r = new_r
            best_pts = points.copy()
        
        # Early termination if no improvement for many iterations
        if it > 100 and new_r <= best_r * (1 - 1e-6):
            # Try a different alpha
            alpha = 0.005
            points = _force_directed_step(points, alpha=alpha, beta=0.001, gamma=0.005)
            new_r = _compute_radius(points)
            if new_r > best_r:
                best_r = new_r
                best_pts = points.copy()
    
    # Local polish
    if time.time() < deadline - 5.0:
        polished = _local_polish(best_pts, max_iters=300, tol=1e-10)
        polished_r = _compute_radius(polished)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished
    
    # Final validation
    final_r = _compute_radius(best_pts)
    if final_r <= 0 or not np.isfinite(final_r):
        # Fallback to initial
        best_pts = points
        final_r = _compute_radius(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
