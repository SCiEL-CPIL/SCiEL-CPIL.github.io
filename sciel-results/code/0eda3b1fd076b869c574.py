"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_score(pts):
    """Compute the packing score: min(min pairwise dist/2, min container margin)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check container: all points must be strictly inside unit circle
    norms = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(norms >= 1.0):
        return 0.0
    container_margin = 1.0 - np.max(norms)
    
    # Check pairwise distances
    if N < 2:
        return container_margin
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    if min_pair_dist <= 0:
        return 0.0
    
    return min(min_pair_dist / 2.0, container_margin)

def _hex_lattice_init(n=N):
    """Generate a hexagonal lattice packing as initial configuration."""
    # Try to find a good hexagonal arrangement
    # Hexagonal packing with spacing d: points at (i*d + j*d/2, j*d*sqrt(3)/2)
    # We want to fit n points in unit circle
    
    best_pts = None
    best_score = -1
    
    # Try various spacings
    for d in np.linspace(0.15, 0.35, 50):
        # Generate hexagonal lattice points
        points = []
        k = int(1.0 / d) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * d
                y = j * d * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - d / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n:
            continue
        
        # Select n points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:n]
        pts = points[idx]
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random
        return _random_init(n)
    
    return best_pts

def _ring_init(n=N):
    """Generate points on concentric rings."""
    # Try different ring configurations
    best_pts = None
    best_score = -1
    
    for n_rings in range(1, 10):
        # Distribute points on rings
        # Ring radii roughly proportional to sqrt(k) for uniform density
        if n_rings == 1:
            # All on one ring
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r = 0.5
            pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        else:
            # Center + rings
            pts_list = [[0.0, 0.0]]
            remaining = n - 1
            for ring_idx in range(1, n_rings):
                frac = remaining / (n_rings - ring_idx + 1)
                n_on_ring = max(1, int(round(frac)))
                if ring_idx == n_rings - 1:
                    n_on_ring = remaining
                r = (ring_idx + 0.5) / n_rings * 0.95
                angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
                ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
                pts_list.append(ring_pts)
                remaining -= n_on_ring
            pts = np.vstack(pts_list)
        
        if len(pts) != n:
            continue
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        return _random_init(n)
    
    return best_pts

def _random_init(n=N, seed=42):
    """Generate random points inside unit circle."""
    rng = np.random.default_rng(seed)
    while True:
        pts = rng.uniform(-1, 1, size=(n, 2))
        norms = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[norms < 1.0]
        if len(pts) >= n:
            # Scale to fit
            scale = 0.95 / np.max(norms)
            pts = pts[:n] * scale
            break
    return pts

def _objective_and_grad(x, pts_shape=N):
    """Objective: maximize min distance. We minimize negative of that."""
    pts = x.reshape(pts_shape, 2)
    
    # Compute all pairwise distances
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Container constraint
    norms = np.hypot(pts[:, 0], pts[:, 1])
    container_margin = 1.0 - np.max(norms)
    
    score = min(min_pair / 2.0, container_margin)
    
    # For gradient, we need to be more careful
    # Use a smooth approximation: soft-min
    # But for simplicity, return the negative score
    return -score

def _lbfgs_polish(pts, max_iter=200, tol=1e-12):
    """Polish the configuration using L-BFGS to maximize minimum distance."""
    x0 = pts.flatten()
    
    def objective(x):
        pts = x.reshape(N, 2)
        dists = cdist(pts, pts)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        norms = np.hypot(pts[:, 0], pts[:, 1])
        container_margin = 1.0 - np.max(norms)
        score = min(min_pair / 2.0, container_margin)
        return -score
    
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
        if res.success or res.fun < -best_score:
            candidate = res.x.reshape(N, 2)
            cand_score = _compute_score(candidate)
            if cand_score > best_score:
                best_pts = candidate
                best_score = cand_score
    except Exception:
        pass
    
    return best_pts

def _simulated_annealing(pts, t_start=0.01, t_end=0.001, n_iter=5000, seed=42):
    """Simulated annealing to escape local minima."""
    rng = np.random.default_rng(seed)
    current = pts.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for i in range(n_iter):
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, t, size=2)
        candidate = current.copy()
        candidate[idx] += delta
        
        # Keep in container
        norm = np.hypot(candidate[idx, 0], candidate[idx, 1])
        if norm >= 1.0:
            continue
        
        cand_score = _compute_score(candidate)
        
        if cand_score > current_score or rng.random() < np.exp((cand_score - current_score) / t):
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best

def construct():
    """Construct the optimal packing of 50 equal circles in a unit circle."""
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with safety margin
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal lattice
    if time.time() < deadline - 30:
        inits.append(_hex_lattice_init())
    
    # Ring configurations
    if time.time() < deadline - 30:
        inits.append(_ring_init())
    
    # Random inits with different seeds
    for seed in [42, 123, 456]:
        if time.time() < deadline - 30:
            inits.append(_random_init(seed=seed))
    
    # Also try a structured approach: place points on a grid then optimize
    if time.time() < deadline - 30:
        # Grid-based init
        grid_size = 8
        pts = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = (i / (grid_size - 1) - 0.5) * 1.8
                y = (j / (grid_size - 1) - 0.5) * 1.8
                if np.hypot(x, y) < 0.95:
                    pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            # Select N closest to center
            radii = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(radii, kind='stable')[:N]
            inits.append(pts[idx])
    
    # Optimize each initial configuration
    best_pts = None
    best_score = -1
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        # Simulated annealing first to escape local minima
        if time.time() < deadline - 20:
            sa_pts = _simulated_annealing(init_pts, t_start=0.005, t_end=0.0005, n_iter=2000, seed=init_idx+1)
            sa_score = _compute_score(sa_pts)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()
        
        # L-BFGS polish
        if time.time() < deadline - 5:
            polished = _lbfgs_polish(best_pts if best_pts is not None else init_pts, max_iter=100, tol=1e-12)
            pol_score = _compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_pts = polished.copy()
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 2:
        final = _lbfgs_polish(best_pts, max_iter=300, tol=1e-14)
        final_score = _compute_score(final)
        if final_score > best_score:
            best_pts = final
    
    # Ensure validity
    if best_pts is None:
        best_pts = _random_init(seed=42)
    
    # Final check: ensure all points are strictly inside
    norms = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(norms >= 1.0):
        # Scale down
        scale = 0.99 / np.max(norms)
        best_pts = best_pts * scale
    
    return best_pts

# EVOLVE-BLOCK-END
