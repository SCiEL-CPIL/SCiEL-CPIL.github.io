"""Packing 50 equal circles in a unit square [-1/2, 1/2]^2."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return DEADLINE - _elapsed()

def _radius(pts):
    """Compute the largest common radius for given centers."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Boundary constraint: min over i,k of (1/2 - |c_ik|)
    boundary = 0.5 - np.abs(pts)
    min_boundary = np.min(boundary)
    if min_boundary < 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = 0.5 * min(min_dist, 2.0 * min_boundary)
    return max(0.0, r)

def _smooth_radius(pts, beta=100.0):
    """Smooth surrogate for radius using soft-min via log-sum-exp."""
    pts = np.asarray(pts, dtype=np.float64)
    # Boundary: soft-min of (0.5 - |c_ik|)
    boundary = 0.5 - np.abs(pts)
    # For soft-min, we use -logsumexp(-x)/beta
    max_b = np.max(boundary)
    soft_min_b = max_b - np.log(np.sum(np.exp(-beta * (boundary - max_b)))) / beta
    
    # Pairwise: soft-min of distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = dists[mask]
    max_d = np.max(dists_masked)
    soft_min_d = max_d - np.log(np.sum(np.exp(-beta * (dists_masked - max_d)))) / beta
    
    r = 0.5 * min(soft_min_d, 2.0 * soft_min_b)
    return r

def _negative_smooth_radius(pts, beta=100.0):
    return -_smooth_radius(pts, beta)

def _hexagonal_init(n=N, spacing=0.15):
    """Hexagonal lattice initialization."""
    pts = []
    row = 0
    while len(pts) < n:
        y = -0.5 + 0.05 + row * spacing * np.sqrt(3) / 2
        if y > 0.45:
            break
        offset = 0.0 if row % 2 == 0 else spacing / 2
        col = 0
        while len(pts) < n:
            x = -0.5 + 0.05 + offset + col * spacing
            if x > 0.45:
                break
            pts.append([x, y])
            col += 1
        row += 1
    pts = np.array(pts[:n])
    return pts

def _random_init(n=N, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.uniform(-0.45, 0.45, size=(n, 2))
    return pts

def _grid_init(n=N):
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _sa_optimize(pts0, n_steps=5000, t0=0.1, t1=0.001, seed=0):
    """Simulated annealing to maximize radius."""
    rng = np.random.RandomState(seed)
    pts = pts0.copy()
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    for step in range(n_steps):
        if _remaining() < 10.0:
            break
        T = t0 * (t1 / t0) ** (step / n_steps)
        
        # Random perturbation
        idx = rng.randint(N)
        delta = rng.normal(0, T * 0.5, size=2)
        new_pts = pts.copy()
        new_pts[idx] += delta
        # Clamp to square
        new_pts = np.clip(new_pts, -0.49, 0.49)
        
        new_r = _radius(new_pts)
        delta_r = new_r - _radius(pts)
        
        if delta_r > 0 or rng.random() < np.exp(delta_r / max(T, 1e-10)):
            pts = new_pts
            if new_r > best_r:
                best_r = new_r
                best_pts = pts.copy()
    
    return best_pts, best_r

def _gradient_polish(pts, beta=50.0, maxiter=200):
    """Polish with L-BFGS-B using smooth surrogate."""
    if _remaining() < 5.0:
        return pts, _radius(pts)
    
    def obj(x):
        pts_x = x.reshape(N, 2)
        return _negative_smooth_radius(pts_x, beta)
    
    def bounds():
        return [(-0.49, 0.49)] * (2 * N)
    
    try:
        res = minimize(obj, pts.flatten(), method='L-BFGS-B', bounds=bounds(),
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
        polished = res.x.reshape(N, 2)
        r = _radius(polished)
        return polished, r
    except:
        return pts, _radius(pts)

def _refine_radius(pts):
    """Iterative radius refinement: try to increase radius by small steps."""
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    for _ in range(20):
        if _remaining() < 15.0:
            break
        target_r = best_r * 1.005
        # Use gradient polish with higher beta
        polished, r = _gradient_polish(best_pts, beta=80.0, maxiter=100)
        if r > best_r:
            best_r = r
            best_pts = polished
        else:
            break
    
    return best_pts, best_r

def _best_known_init():
    """Approximation of the Boll et al. 2000 packing structure."""
    # Start from a hexagonal-like arrangement and perturb
    pts = _hexagonal_init(N, spacing=0.145)
    # Add small random perturbation
    rng = np.random.RandomState(123)
    pts += rng.normal(0, 0.01, size=pts.shape)
    pts = np.clip(pts, -0.45, 0.45)
    return pts

def construct():
    global _start_time
    _start_time = time.time()
    
    best_pts = None
    best_r = 0.0
    
    # Multiple initial configurations
    inits = [
        ("grid", _grid_init()),
        ("hex", _hexagonal_init()),
        ("random1", _random_init(seed=1)),
        ("random2", _random_init(seed=42)),
        ("random3", _random_init(seed=99)),
        ("known", _best_known_init()),
    ]
    
    for name, init_pts in inits:
        if _remaining() < 20.0:
            break
        
        # SA phase
        sa_pts, sa_r = _sa_optimize(init_pts, n_steps=3000, t0=0.08, t1=0.001, seed=hash(name) % 10000)
        
        # Gradient polish
        polished, pol_r = _gradient_polish(sa_pts, beta=50.0, maxiter=150)
        
        # Refine
        refined, ref_r = _refine_radius(polished)
        
        if ref_r > best_r:
            best_r = ref_r
            best_pts = refined.copy()
    
    # Final high-precision polish on best
    if best_pts is not None and _remaining() > 10.0:
        final, final_r = _gradient_polish(best_pts, beta=100.0, maxiter=300)
        if final_r > best_r:
            best_r = final_r
            best_pts = final
    
    if best_pts is None:
        return _grid_init()
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.49, 0.49)
    return best_pts
