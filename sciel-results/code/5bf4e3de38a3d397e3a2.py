"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150 s)


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _covering_radius_hull(pts):
    """Exact covering radius (radians) via convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    best = 0.0
    for f in hull.simplices:
        tri = pts[f]
        a, b, c = tri[0], tri[1], tri[2]
        # Compute circumcenter of triangle on unit sphere
        # For points on unit sphere, circumcenter direction is normal to plane
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
        # Check if circumcenter is on unit sphere (it always is by construction)
        # The cap is "empty" if cc is on the same side as the facet normal
        # For a convex hull, the circumcenter should be outside or on the sphere
        cosv = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(cosv)
        if r > best:
            best = r
    if best == 0.0:
        G = np.clip(pts @ pts.T, -1.0, 1.0)
        np.fill_diagonal(G, -1.0)
        best = np.arccos(G.min())
    return best


def _covering_radius_fast(pts, n_grid=200):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    # Generate grid points on sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    # Compute angular distances from each grid point to nearest config point
    # cos(angle) = dot product
    G = grid @ pts.T  # shape (n_grid_points, N)
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)  # closest point
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build the Hardin-Sloane-Smith 50-point configuration.
    This is based on a truncated icosahedron with specific vertices removed.
    """
    t = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    
    # Icosahedron vertices
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    # Find edges of icosahedron
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    # Truncated icosahedron vertices: 2 per edge
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    # We have 60 vertices. Need to remove 10 to get 50.
    # Map each truncated vertex to its parent icosahedron vertices
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    # Find antipodal pair in icosahedron
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
    # Remove vertices associated with these two icosahedron vertices
    remove = set()
    for idx, (i, j) in enumerate(vmap):
        if i == pair_i or j == pair_i or i == pair_j or j == pair_j:
            remove.add(idx)
    
    keep = [k for k in range(len(verts)) if k not in remove]
    out = verts[keep]
    if len(out) != 50:
        out = out[:50]
    return _normalize(out)


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_optimize_fast(pts, time_budget):
    """Local optimization using fast covering radius evaluation."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    start = time.time()
    
    # Use coordinate-wise descent with adaptive step size
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step = 0
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    for ss in step_sizes[current_step:current_step+2]:
                        cand = pts.copy()
                        cand[i, axis] += sgn * ss
                        norm = np.linalg.norm(cand[i])
                        if norm < 1e-10:
                            continue
                        cand[i] /= norm
                        r = _covering_radius_fast(cand)
                        if r < best_r - 1e-12:
                            pts = cand
                            best_r = r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            current_step += 1
            if current_step >= len(step_sizes):
                break
    
    return pts, best_r


def _repulsive_force_optimize(pts, time_budget):
    """Repulsive force-based dynamics: points flow away from crowded regions.
    
    Each point experiences repulsion from neighbors, scaled inversely by local
    covering density. Points in sparse regions feel less repulsion and can move
    into empty areas, while clustered points push apart.
    """
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    start = time.time()
    
    # Parameters
    lr = 0.01  # learning rate
    decay = 0.999  # learning rate decay
    min_lr = 1e-5
    
    while time.time() - start < time_budget and lr > min_lr:
        # Compute pairwise dot products
        G = pts @ pts.T  # shape (N, N)
        G = np.clip(G, -1.0, 1.0)
        np.fill_diagonal(G, 1.0)
        
        # For each point, find its k nearest neighbors and compute local density
        # Use a fixed k for efficiency
        k = 8
        # Sort by distance (1 - cos = 1 - G)
        dists = 1.0 - G  # angular distance proxy
        np.fill_diagonal(dists, np.inf)
        
        # Get k nearest neighbors for each point
        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]  # (N, k)
        
        # Compute local covering density: average angular distance to k-NN
        # Smaller distance = more crowded = higher density
        local_dists = np.take_along_axis(dists, nn_idx, axis=1)  # (N, k)
        local_density = local_dists.mean(axis=1)  # (N,) - smaller = more crowded
        
        # Normalize density to [0, 1] for scaling
        d_min, d_max = local_density.min(), local_density.max()
        if d_max - d_min > 1e-10:
            local_density_norm = (local_density - d_min) / (d_max - d_min)
        else:
            local_density_norm = np.ones(N) * 0.5
        
        # Compute repulsive forces
        # Force on point i from point j: direction away from j, magnitude ~ 1/(1+dist^2) * density_scale
        forces = np.zeros_like(pts)
        for i in range(N):
            # Scale repulsion by local density: crowded points push harder
            density_scale = 1.0 + 2.0 * (1.0 - local_density_norm[i])
            
            for j in range(N):
                if i == j:
                    continue
                # Direction from j to i
                d = pts[i] - pts[j]
                d_norm = np.linalg.norm(d)
                if d_norm < 1e-10:
                    continue
                d_hat = d / d_norm
                
                # Angular distance between i and j
                cos_ij = np.clip(np.dot(pts[i], pts[j]), -1.0, 1.0)
                ang_dist = np.arccos(cos_ij)
                
                # Repulsion magnitude: stronger for close points, scaled by density
                # Use inverse square-like falloff
                rep_mag = density_scale * (0.01 / (1.0 + ang_dist * ang_dist * 50.0))
                
                forces[i] += rep_mag * d_hat
        
        # Update points: move along force direction, project back to sphere
        new_pts = pts + lr * forces
        new_pts = _normalize(new_pts)
        
        # Evaluate
        r = _covering_radius_fast(new_pts)
        if r < best_r:
            best_r = r
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Revert if worse
            pass
        
        # Decay learning rate
        lr *= decay
    
    return best_pts, best_r


def _nelder_mead_optimize(pts, time_budget):
    """Nelder-Mead style optimization on the sphere."""
    from scipy.optimize import minimize
    
    pts = _normalize(pts).copy()
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_budget:
            return np.inf
        p = x.reshape(N, 3)
        p = _normalize(p)
        return _covering_radius_fast(p)
    
    # Flatten the points
    x0 = pts.ravel()
    
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    
    if result.success or result.fun < _covering_radius_fast(pts):
        best_pts = _normalize(result.x.reshape(N, 3))
    else:
        best_pts = pts
    
    return best_pts


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Random configurations with symmetry
    rng = np.random.RandomState(42)
    for _ in range(3):
        # Start with icosahedral symmetry
        ico = np.array([
            [0, 1, (1+np.sqrt(5))/2], [0, -1, (1+np.sqrt(5))/2],
            [0, 1, -(1+np.sqrt(5))/2], [0, -1, -(1+np.sqrt(5))/2],
            [1, (1+np.sqrt(5))/2, 0], [-1, (1+np.sqrt(5))/2, 0],
            [1, -(1+np.sqrt(5))/2, 0], [-1, -(1+np.sqrt(5))/2, 0],
            [(1+np.sqrt(5))/2, 0, 1], [-(1+np.sqrt(5))/2, 0, 1],
            [(1+np.sqrt(5))/2, 0, -1], [-(1+np.sqrt(5))/2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        # Add 38 random points
        rand_pts = rng.randn(38, 3)
        rand_pts = rand_pts / np.linalg.norm(rand_pts, axis=1, keepdims=True)
        
        all_pts = np.vstack([ico, rand_pts])
        configs.append(all_pts)
    
    # Evaluate all configs and pick the best
    best_pts = None
    best_r = np.inf
    
    for cfg in configs:
        cfg = _normalize(cfg)
        r = _covering_radius_fast(cfg)
        if r < best_r:
            best_r = r
            best_pts = cfg
    
    if best_pts is None:
        best_pts = fib
        best_r = _covering_radius_fast(best_pts)
    
    # Repulsive force-based optimization (new strategy)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 20.0:
        try:
            rep_pts, rep_r = _repulsive_force_optimize(best_pts, min(remaining - 20.0, 50.0))
            if rep_r < best_r:
                best_pts, best_r = rep_pts, rep_r
        except Exception:
            pass
    
    # Local optimization
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0:
        opt_pts, opt_r = _local_optimize_fast(best_pts, min(remaining - 10.0, 40.0))
        if opt_r < best_r:
            best_pts, best_r = opt_pts, opt_r
    
    # Try Nelder-Mead if time permits
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15.0:
        try:
            nm_pts = _nelder_mead_optimize(best_pts, min(remaining - 15.0, 30.0))
            nm_r = _covering_radius_fast(nm_pts)
            if nm_r < best_r:
                best_pts, best_r = nm_pts, nm_r
        except Exception:
            pass
    
    # Final polish with exact hull radius
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 5.0:
        try:
            exact_r = _covering_radius_hull(best_pts)
            # If exact is much worse than fast estimate, do a final local polish
            if exact_r > best_r + 0.005:
                # Use exact radius for final polish
                polish_pts = best_pts.copy()
                polish_start = time.time()
                polish_budget = min(remaining - 5.0, 10.0)
                
                # Simple hill-climbing with exact radius
                step = 0.01
                for _ in range(1000):
                    if time.time() - polish_start > polish_budget:
                        break
                    improved = False
                    for i in range(N):
                        for axis in range(3):
                            for sgn in (1.0, -1.0):
                                cand = polish_pts.copy()
                                cand[i, axis] += sgn * step
                                norm = np.linalg.norm(cand[i])
                                if norm < 1e-10:
                                    continue
                                cand[i] /= norm
                                r = _covering_radius_hull(cand)
                                if r < exact_r - 1e-10:
                                    polish_pts = cand
                                    exact_r = r
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if not improved:
                        step *= 0.7
                        if step < 1e-4:
                            break
                
                if exact_r < best_r:
                    best_pts = polish_pts
                    best_r = exact_r
        except Exception:
            pass
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
