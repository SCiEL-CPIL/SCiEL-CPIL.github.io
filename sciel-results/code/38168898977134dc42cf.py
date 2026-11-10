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
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        cc = n / nn
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
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    return angles.max()


def _build_hss50():
    """Build the Hardin-Sloane-Smith 50-point configuration."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico = _normalize(ico)
    
    D = 1.0 - ico @ ico.T
    min_d = D.min()
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_d) < 1e-8:
                edges.append((i, j))
    
    verts = []
    for (i, j) in edges:
        verts.append(ico[i] * (2.0 / 3.0) + ico[j] * (1.0 / 3.0))
        verts.append(ico[i] * (1.0 / 3.0) + ico[j] * (2.0 / 3.0))
    verts = np.array(verts)
    verts = _normalize(verts)
    
    vmap = []
    for (i, j) in edges:
        vmap.append((i, j))
        vmap.append((i, j))
    
    G = ico @ ico.T
    np.fill_diagonal(G, -2.0)
    pair_i, pair_j = np.unravel_index(np.argmin(G), G.shape)
    
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


def _soft_max_covering(pts, grid, temperature=0.01):
    """
    Smooth surrogate: soft-max of angular distances via log-sum-exp.
    This gives a differentiable upper bound on the max covering radius.
    """
    G = grid @ pts.T  # (n_grid, N)
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    
    # Soft-max: T * log(sum(exp(angles/T)))
    a_max = angles.max()
    if temperature < 1e-10:
        return a_max
    shifted = (angles - a_max) / temperature
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = a_max + temperature * log_sum
    return soft_max


def _soft_max_gradient(pts, grid, temperature=0.01):
    """
    Compute gradient of soft-max covering radius w.r.t. point positions.
    Uses chain rule through the soft-max.
    """
    G = grid @ pts.T  # (n_grid, N)
    G_clipped = np.clip(G, -1.0, 1.0)
    max_cos = G_clipped.max(axis=1)
    angles = np.arccos(max_cos)
    
    # Weights for soft-max
    a_max = angles.max()
    if temperature < 1e-10:
        # Hard max: gradient is subgradient
        worst_idx = np.argmax(angles)
        grad = np.zeros_like(pts)
        nearest = G_clipped[worst_idx].argmax()
        worst_pt = grid[worst_idx]
        grad[nearest] = worst_pt
        return grad
    
    shifted = (angles - a_max) / temperature
    shifted = np.clip(shifted, -100, 100)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient: sum over grid points of weight_i * d(angle_i)/d(pts)
    # angle_i = arccos(max_j cos(theta_ij))
    # d(angle_i)/d(pts_j) = -1/sin(angle_i) * d(max_j cos)/d(pts_j)
    # For the argmax j, d(cos(theta_ij))/d(pts_j) = -grid_i (since cos = grid_i . pts_j)
    
    grad = np.zeros_like(pts)
    sin_angles = np.sin(angles)
    sin_angles[sin_angles < 1e-10] = 1e-10
    
    # For each grid point, the nearest config point contributes
    nearest_idx = G_clipped.argmax(axis=1)
    
    # Vectorized: accumulate gradient contributions
    # For each grid point i, contribute to grad[nearest_idx[i]]:
    #   weight[i] * (-grid[i] / sin_angles[i])
    contrib = -grid / sin_angles[:, np.newaxis]  # (n_grid, 3)
    contrib *= weights[:, np.newaxis]  # (n_grid, 3)
    
    # Scatter-add: for each grid point, add its contribution to the nearest config point
    for i in range(len(grid)):
        j = nearest_idx[i]
        grad[j] += contrib[i]
    
    return grad


def _soft_max_optimize(pts, time_budget, n_grid=100, T_init=0.05, T_min=0.005):
    """
    Optimize using soft-max surrogate with annealing.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    
    # Generate grid
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    T = T_init
    lr = 0.03
    decay_lr = 0.997
    decay_T = 0.993
    min_lr = 1e-4
    
    while time.time() - start < time_budget and lr > min_lr:
        # Compute soft-max objective and gradient
        obj = _soft_max_covering(pts, grid, T)
        grad = _soft_max_gradient(pts, grid, T)
        
        # Gradient descent step
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        # Evaluate true covering radius
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Anneal
        T = max(T_min, T * decay_T)
        lr *= decay_lr
    
    return best_pts, best_r


def _vectorized_repulsive(pts, time_budget):
    """
    Fast vectorized repulsive force optimization.
    Points in crowded regions get pushed away more.
    """
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    start = time.time()
    
    lr = 0.02
    decay = 0.998
    min_lr = 1e-5
    k = 6  # number of nearest neighbors for density
    
    while time.time() - start < time_budget and lr > min_lr:
        G = pts @ pts.T
        G = np.clip(G, -1.0, 1.0)
        np.fill_diagonal(G, 1.0)
        
        dists = 1.0 - G
        np.fill_diagonal(dists, np.inf)
        
        # Find k nearest neighbors for each point
        nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
        local_dists = np.take_along_axis(dists, nn_idx, axis=1)
        local_density = local_dists.mean(axis=1)
        
        # Normalize density: points in sparse regions (high local_density) get more repulsion
        d_min, d_max = local_density.min(), local_density.max()
        if d_max - d_min > 1e-10:
            local_density_norm = (local_density - d_min) / (d_max - d_min)
        else:
            local_density_norm = np.ones(N) * 0.5
        
        # Compute repulsive forces vectorized
        # For each pair (i,j), force on i is proportional to (pts[i] - pts[j]) / (1 + dist^2)
        # weighted by (1 - local_density_norm[i]) so crowded points get less repulsion
        
        # Build pairwise distance matrix
        D2 = np.sum(pts[:, np.newaxis, :] ** 2, axis=2) + np.sum(pts[np.newaxis, :, :] ** 2, axis=2) - 2 * (pts @ pts.T)
        D2 = np.clip(D2, 0, None)
        np.fill_diagonal(D2, np.inf)
        
        # Repulsion magnitude: 1/(1 + D2^2) * (1 - local_density_norm[i])
        repulsion_mag = 1.0 / (1.0 + D2 * D2)
        repulsion_mag *= (1.0 - local_density_norm[:, np.newaxis])
        np.fill_diagonal(repulsion_mag, 0)
        
        # Force vector: sum over j of repulsion_mag[i,j] * (pts[i] - pts[j])
        # = repulsion_mag @ pts - (repulsion_mag.sum(axis=1, keepdims=True) * pts)
        force = repulsion_mag @ pts - repulsion_mag.sum(axis=1, keepdims=True) * pts
        
        # Normalize forces
        f_norms = np.linalg.norm(force, axis=1, keepdims=True)
        f_norms[f_norms == 0] = 1.0
        force = force / f_norms
        
        # Update
        pts = pts + lr * force
        pts = _normalize(pts)
        
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        lr *= decay
    
    return best_pts, best_r


def _tangent_space_refine(pts, time_budget, n_grid=150):
    """
    Fine-tune using tangent-space coordinate descent.
    For each point, try small moves in tangent directions to reduce covering radius.
    """
    pts = _normalize(pts).copy()
    start = time.time()
    
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    
    step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    current_ss = 0
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start >= time_budget:
                break
            
            pi = pts[i]
            # Build orthonormal basis for tangent plane at pi
            if abs(pi[2]) < 0.9:
                u = np.array([0, 0, 1.0])
            else:
                u = np.array([1, 0, 0.0])
            e1 = np.cross(pi, u)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(pi, e1)
            e2 /= np.linalg.norm(e2)
            
            for ss in step_sizes[current_ss:current_ss + 2]:
                # Try 4 tangent directions
                for dir_vec in [e1, -e1, e2, -e2, (e1 + e2) / np.sqrt(2), (e1 - e2) / np.sqrt(2),
                                 (-e1 + e2) / np.sqrt(2), (-e1 - e2) / np.sqrt(2)]:
                    cand = pts.copy()
                    cand[i] = pi + ss * dir_vec
                    cand[i] /= np.linalg.norm(cand[i])
                    r = _covering_radius_fast(cand)
                    if r < best_r - 1e-12:
                        pts = cand
                        best_r = r
                        best_pts = pts.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            current_ss += 1
            if current_ss >= len(step_sizes):
                break
    
    return best_pts, best_r


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
    
    # 3. Random configurations with icosahedral symmetry
    rng = np.random.RandomState(42)
    for seed in [42, 123, 456]:
        rng_s = np.random.RandomState(seed)
        ico = np.array([
            [0, 1, (1 + np.sqrt(5)) / 2], [0, -1, (1 + np.sqrt(5)) / 2],
            [0, 1, -(1 + np.sqrt(5)) / 2], [0, -1, -(1 + np.sqrt(5)) / 2],
            [1, (1 + np.sqrt(5)) / 2, 0], [-1, (1 + np.sqrt(5)) / 2, 0],
            [1, -(1 + np.sqrt(5)) / 2, 0], [-1, -(1 + np.sqrt(5)) / 2, 0],
            [(1 + np.sqrt(5)) / 2, 0, 1], [-(1 + np.sqrt(5)) / 2, 0, 1],
            [(1 + np.sqrt(5)) / 2, 0, -1], [-(1 + np.sqrt(5)) / 2, 0, -1],
        ], dtype=float)
        ico = _normalize(ico)
        
        rand_pts = rng_s.randn(38, 3)
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
    
    # Phase 1: Vectorized repulsive force (fast, spreads points)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 30.0:
        opt_pts, opt_r = _vectorized_repulsive(best_pts, min(remaining - 20.0, 40.0))
        if opt_r < best_r:
            best_pts, best_r = opt_pts, opt_r
    
    # Phase 2: Soft-max gradient optimization (refines worst regions)
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 20.0:
        opt_pts, opt_r = _soft_max_optimize(best_pts, min(remaining - 10.0, 50.0), n_grid=80)
        if opt_r < best_r:
            best_pts, best_r = opt_pts, opt_r
    
    # Phase 3: Tangent-space local refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0:
        opt_pts, opt_r = _tangent_space_refine(best_pts, min(remaining - 5.0, 30.0))
        if opt_r < best_r:
            best_pts, best_r = opt_pts, opt_r
    
    # Ensure valid output
    best_pts = _normalize(best_pts)
    if not np.all(np.isfinite(best_pts)) or len(best_pts) != N:
        best_pts = fib
    
    return best_pts
# EVOLVE-BLOCK-END
