"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull

DEADLINE = 138.0  # wall-clock safety margin (evaluator kills at 150 s)


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


def _covering_radius_and_worst(pts, grid):
    """Return covering radius and the worst grid point index."""
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    worst_idx = np.argmax(angles)
    return angles[worst_idx], worst_idx


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


def _spherical_kmeans_init(rng, n_iter=100):
    """Spherical K-Means initialization for uniform point distribution."""
    # Initialize with random points
    centers = rng.randn(N, 3)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    for _ in range(n_iter):
        # Assign each center to nearest center (for re-initialization diversity)
        # Actually, we want to spread points evenly. Use a simple approach:
        # Generate many random points, assign to nearest center, recompute centers
        n_samples = 2000
        samples = rng.randn(n_samples, 3)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        
        # Assign samples to nearest center
        G = samples @ centers.T  # (n_samples, N)
        assignments = np.argmax(G, axis=1)
        
        # Recompute centers as mean of assigned points (spherical mean)
        new_centers = np.zeros_like(centers)
        for k in range(N):
            mask = assignments == k
            if np.sum(mask) > 0:
                new_centers[k] = np.sum(samples[mask], axis=0)
                norm = np.linalg.norm(new_centers[k])
                if norm > 1e-10:
                    new_centers[k] /= norm
                else:
                    new_centers[k] = centers[k]
            else:
                new_centers[k] = centers[k]
        
        # Check convergence
        diff = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if diff < 1e-8:
            break
    
    return _normalize(centers)


def _soft_max_covering(pts, grid, temperature=0.01):
    """Smooth surrogate: soft-max of angular distances via log-sum-exp."""
    G = grid @ pts.T
    G = np.clip(G, -1.0, 1.0)
    max_cos = G.max(axis=1)
    angles = np.arccos(max_cos)
    
    a_max = angles.max()
    if temperature < 1e-10:
        return a_max
    shifted = (angles - a_max) / temperature
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_max = a_max + temperature * log_sum
    return soft_max


def _soft_max_gradient(pts, grid, temperature=0.01):
    """Compute gradient of soft-max covering radius w.r.t. point positions."""
    G = grid @ pts.T
    G_clipped = np.clip(G, -1.0, 1.0)
    max_cos = G_clipped.max(axis=1)
    angles = np.arccos(max_cos)
    
    a_max = angles.max()
    if temperature < 1e-10:
        worst_idx = np.argmax(angles)
        grad = np.zeros_like(pts)
        nearest = G_clipped[worst_idx].argmax()
        worst_pt = grid[worst_idx]
        grad[nearest] = worst_pt
        return grad
    
    shifted = (angles - a_max) / temperature
    shifted = np.clip(shifted, -100, 100)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    grad = np.zeros_like(pts)
    sin_angles = np.sin(angles)
    sin_angles[sin_angles < 1e-10] = 1e-10
    
    nearest_idx = G_clipped.argmax(axis=1)
    
    for i in range(len(grid)):
        w = weights[i]
        j = nearest_idx[i]
        grad[j] += w * (-grid[i] / sin_angles[i])
    
    return grad


def _optimize_softmax(pts, time_budget, n_grid=100, T_init=0.03, T_min=0.003):
    """Optimize using soft-max surrogate with annealing."""
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
    
    T = T_init
    lr = 0.015
    decay_lr = 0.9985
    decay_T = 0.996
    min_lr = 1e-5
    
    while time.time() - start < time_budget and lr > min_lr:
        obj = _soft_max_covering(pts, grid, T)
        grad = _soft_max_gradient(pts, grid, T)
        
        pts = pts - lr * grad
        pts = _normalize(pts)
        
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        T = max(T_min, T * decay_T)
        lr *= decay_lr
    
    return best_pts, best_r


def _repulsive_force_optimize(pts, time_budget, n_grid=100):
    """Repulsive force-based dynamics to escape local minima."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts)
    best_pts = pts.copy()
    start = time.time()
    
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    lr = 0.008
    decay = 0.9995
    min_lr = 1e-6
    
    while time.time() - start < time_budget and lr > min_lr:
        G = pts @ pts.T
        G = np.clip(G, -1.0, 1.0)
        np.fill_diagonal(G, 1.0)
        
        dists = 1.0 - G
        np.fill_diagonal(dists, np.inf)
        
        # Find worst point on grid and move nearest config point toward it
        r, worst_idx = _covering_radius_and_worst(pts, grid)
        worst_pt = grid[worst_idx]
        nearest = (pts @ worst_pt).argmax()
        
        # Force: move nearest point toward worst point, plus repulsion from close neighbors
        forces = np.zeros_like(pts)
        
        # Attraction to worst point for nearest config point
        forces[nearest] += 2.0 * (worst_pt - pts[nearest])
        
        # Repulsion from close neighbors
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 0.1:  # Only repel if close
                    mag = (0.1 - d) * 0.5
                    forces[i] += mag * (pts[i] - pts[j])
        
        pts = pts + lr * forces
        pts = _normalize(pts)
        
        r = _covering_radius_fast(pts)
        if r < best_r:
            best_r = r
            best_pts = pts.copy()
        
        lr *= decay
    
    return best_pts, best_r


def _coordinate_descent_polish(pts, time_budget, n_grid=150):
    """Fine coordinate descent polish."""
    pts = _normalize(pts).copy()
    best_r = _covering_radius_fast(pts, n_grid=n_grid)
    best_pts = pts.copy()
    start = time.time()
    
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, 2 * n_grid)
    theta, phi = np.meshgrid(theta, phi)
    grid_x = np.sin(theta) * np.cos(phi)
    grid_y = np.sin(theta) * np.sin(phi)
    grid_z = np.cos(theta)
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid = _normalize(grid)
    
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in step_sizes:
        if time.time() - start > time_budget:
            break
        improved = True
        while improved and time.time() - start < time_budget:
            improved = False
            for i in range(N):
                for axis in range(3):
                    for sgn in (1.0, -1.0):
                        cand = pts.copy()
                        cand[i, axis] += sgn * step
                        norm = np.linalg.norm(cand[i])
                        if norm < 1e-10:
                            continue
                        cand[i] /= norm
                        r = _covering_radius_fast(cand, n_grid=n_grid)
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
    
    return best_pts, best_r


def construct():
    t0 = time.time()
    
    # Generate multiple initial configurations
    configs = []
    
    # 1. Hardin-Sloane-Smith based (primary - known near-optimal)
    try:
        hss = _build_hss50()
        if len(hss) == 50 and np.all(np.isfinite(hss)):
            configs.append(hss)
    except Exception:
        pass
    
    # 2. Fibonacci spiral
    fib = _fibonacci_spiral()
    configs.append(fib)
    
    # 3. Spherical K-Means with different seeds
    rng = np.random.RandomState(42)
    for seed in [1, 2, 3]:
        rng_s = np.random.RandomState(seed)
        km = _spherical_kmeans_init(rng_s, n_iter=50)
        configs.append(km)
    
    # 4. HSS50 with small random perturbations
    try:
        hss = _build_hss50()
        for scale in [0.01, 0.02]:
            perturbed = hss + scale * rng.randn(50, 3)
            configs.append(_normalize(perturbed))
    except Exception:
        pass
    
    # Evaluate all configs and pick the best few
    scored_configs = []
    for cfg in configs:
        cfg = _normalize(cfg)
        if len(cfg) != 50 or not np.all(np.isfinite(cfg)):
            continue
        r = _covering_radius_fast(cfg, n_grid=150)
        scored_configs.append((r, cfg))
    
    scored_configs.sort(key=lambda x: x[0])
    
    if not scored_configs:
        best_pts = fib
    else:
        best_pts = scored_configs[0][1]
    
    # Optimize the top 3 configurations with different strategies
    remaining = DEADLINE - (time.time() - t0)
    best_overall = best_pts.copy()
    best_overall_r = _covering_radius_fast(best_overall, n_grid=200)
    
    n_starts = min(3, len(scored_configs))
    time_per_start = max(5.0, (remaining - 15.0) / max(n_starts, 1))
    
    for idx in range(n_starts):
        if time.time() - t0 > DEADLINE - 15.0:
            break
        
        cfg = scored_configs[idx][1]
        start_time = time.time()
        
        # Phase 1: Soft-max gradient optimization
        budget1 = min(time_per_start * 0.4, 30.0)
        if budget1 > 3.0:
            pts1, r1 = _optimize_softmax(cfg, budget1, n_grid=80, T_init=0.02, T_min=0.002)
            if r1 < best_overall_r:
                best_overall_r = r1
                best_overall = pts1.copy()
        
        # Phase 2: Repulsive force optimization
        elapsed = time.time() - start_time
        budget2 = max(0.0, time_per_start * 0.3 - elapsed)
        if budget2 > 3.0:
            pts2, r2 = _repulsive_force_optimize(best_overall, budget2, n_grid=80)
            if r2 < best_overall_r:
                best_overall_r = r2
                best_overall = pts2.copy()
        
        # Phase 3: Coordinate descent polish
        elapsed = time.time() - start_time
        budget3 = max(0.0, time_per_start - elapsed - 5.0)
        if budget3 > 3.0:
            pts3, r3 = _coordinate_descent_polish(best_overall, budget3, n_grid=150)
            if r3 < best_overall_r:
                best_overall_r = r3
                best_overall = pts3.copy()
    
    # Final polish on the best configuration
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0:
        final_pts, final_r = _coordinate_descent_polish(best_overall, min(remaining - 5.0, 40.0), n_grid=200)
        if final_r < best_overall_r:
            best_overall_r = final_r
            best_overall = final_pts.copy()
    
    # Ensure valid output
    best_overall = _normalize(best_overall)
    if not np.all(np.isfinite(best_overall)) or len(best_overall) != N:
        best_overall = fib
    
    return best_overall
# EVOLVE-BLOCK-END
