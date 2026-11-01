"""
Optimized sphere packing: multi-start L-BFGS-B with adaptive soft-min + local refinement.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time as _time

N = 100

_DEADLINE = _time.time() + 135.0  # safety margin before 150s kill

def _compute_radius(centers):
    """Compute the largest common radius for given centers inside [-0.5, 0.5]^3."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))
    
    return min(min_pair / 2.0, min_wall)

def _soft_min_objective(x, beta, iu):
    """
    Objective: minimize -soft_min of all radius-limiting distances.
    """
    centers = x.reshape(N, 3)
    
    # Wall distances
    half = 0.5
    wall_dists = np.min(half - np.abs(centers), axis=1)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_dists = dists[iu] / 2.0
    
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Stable log-sum-exp based soft-min
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_args = -beta * shifted
    exp_args = np.maximum(exp_args, -700.0)
    log_sum = np.log(np.sum(np.exp(exp_args)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    return -soft_min_val

def _build_grid(m1, m2, m3, jitter=0.0, seed=42):
    """Build a grid of size m1 x m2 x m3 and return first N points."""
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    if jitter > 0:
        rng = np.random.RandomState(seed)
        noise = rng.randn(*grid.shape) * jitter
        grid = grid + noise
        np.clip(grid, -0.499, 0.499, out=grid)
    return grid[:N]

def _build_halton(n=100, seed=42):
    """Build a Halton sequence based initialization for good space-filling properties."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.randn(*centers.shape) * 0.005
    np.clip(centers, -0.499, 0.499, out=centers)
    
    return centers

def _lbfgsb_refine(centers, max_time=60.0):
    """
    Refine centers using L-BFGS-B with adaptive beta schedule.
    """
    start = _time.time()
    x = centers.flatten().copy()
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    iu = np.triu_indices(N, k=1)
    
    # Adaptive beta schedule
    betas = [10, 30, 100, 300, 1000, 3000, 5000]
    
    for beta in betas:
        if _time.time() > _DEADLINE - 15 or (_time.time() - start) > max_time:
            break
        
        try:
            res = minimize(
                _soft_min_objective,
                x,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-8, 'disp': False}
            )
            
            new_centers = res.x.reshape(N, 3)
            if np.all(np.abs(new_centers) <= 0.4999) and np.all(np.isfinite(new_centers)):
                x = new_centers.flatten()
            else:
                new_centers = np.clip(res.x.reshape(N, 3), -0.4999, 0.4999)
                x = new_centers.flatten()
                
        except Exception:
            pass
    
    return x.reshape(N, 3)

def _anneal(centers, deadline, T_start=0.01, T_end=0.0001, n_steps=3000):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings with energy = -radius (we want to maximize radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    iu = np.triu_indices(N, k=1)
    rng = np.random.RandomState(42)
    
    n_done = 0
    for step in range(n_steps):
        if _time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.randint(N)
        delta = rng.randn(3) * 0.01
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        np.clip(new_centers[idx], -0.5, 0.5, out=new_centers[idx])
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta_E = current_radius - new_radius  # positive means worse
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                current_centers = new_centers
                current_radius = new_radius
        
        n_done += 1
    
    return best_centers, best_radius

def _local_perturbation_refine(centers, max_time=30.0):
    """
    Per-point local refinement with decreasing perturbation scales.
    """
    start = _time.time()
    centers = centers.copy()
    n = centers.shape[0]
    r = _compute_radius(centers)
    
    for _pass in range(15):
        if _time.time() > _DEADLINE - 10 or (_time.time() - start) > max_time:
            break
        
        for i in range(n):
            if _time.time() > _DEADLINE - 10 or (_time.time() - start) > max_time:
                break
            
            orig = centers[i].copy()
            best_local = orig.copy()
            best_r = r
            
            for scale in [0.01, 0.005, 0.002, 0.001, 0.0005]:
                for _ in range(4):
                    perturb = np.random.randn(3) * scale
                    new_pt = centers[i] + perturb
                    if np.any(np.abs(new_pt) > 0.4999):
                        continue
                    centers[i] = new_pt
                    new_r = _compute_radius(centers)
                    if new_r > best_r + 1e-15:
                        best_r = new_r
                        best_local = new_pt.copy()
                    else:
                        centers[i] = best_local
            
            centers[i] = best_local
            r = best_r
    
    return centers

def _generate_initial_configs():
    """Generate multiple diverse initial configurations."""
    configs = []
    
    # Grid-based configurations
    grid_specs = [
        (5, 5, 4),
        (5, 4, 5),
        (4, 5, 5),
        (4, 4, 6),
        (6, 4, 4),
        (4, 6, 4),
        (5, 5, 4),
        (4, 5, 5),
        (5, 4, 5),
        (4, 4, 6),
    ]
    
    for idx, (m1, m2, m3) in enumerate(grid_specs):
        if m1 * m2 * m3 < N:
            continue
        # Try with and without jitter
        configs.append(_build_grid(m1, m2, m3, jitter=0.0, seed=idx))
        configs.append(_build_grid(m1, m2, m3, jitter=0.005, seed=idx))
        configs.append(_build_grid(m1, m2, m3, jitter=0.01, seed=idx + 100))
    
    # Halton sequence configurations
    for seed in range(5):
        configs.append(_build_halton(N, seed=seed))
    
    # Random configurations
    for seed in range(10):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(-0.45, 0.45, size=(N, 3))
        configs.append(centers)
    
    return configs

def construct():
    start_time = _time.time()
    
    # Generate initial configurations
    initial_configs = _generate_initial_configs()
    
    best_centers = None
    best_r = -np.inf
    
    # Evaluate all initial configs and pick top few for refinement
    scored_configs = []
    for cfg in initial_configs:
        r = _compute_radius(cfg)
        scored_configs.append((r, cfg))
    
    scored_configs.sort(key=lambda x: -x[0])
    
    # Refine top configurations
    max_refine = 5
    for r_init, cfg in scored_configs[:max_refine]:
        if _time.time() > _DEADLINE - 20:
            break
        
        if r_init > best_r:
            best_r = r_init
            best_centers = cfg.copy()
        
        # L-BFGS-B refinement
        try:
            refined = _lbfgsb_refine(cfg, max_time=25.0)
            r_ref = _compute_radius(refined)
            if r_ref > best_r:
                best_r = r_ref
                best_centers = refined.copy()
        except Exception:
            pass
    
    # Simulated annealing on the best
    if best_centers is not None and _time.time() < _DEADLINE - 15:
        try:
            annealed, r_anneal = _anneal(best_centers, _DEADLINE - 15)
            if r_anneal > best_r:
                best_r = r_anneal
                best_centers = annealed.copy()
        except Exception:
            pass
    
    # Local perturbation polish on the best
    if best_centers is not None and _time.time() < _DEADLINE - 10:
        try:
            polished = _local_perturbation_refine(best_centers, max_time=20.0)
            r_pol = _compute_radius(polished)
            if r_pol > best_r:
                best_r = r_pol
                best_centers = polished
        except Exception:
            pass
    
    # Final validation
    if best_centers is None:
        best_centers = _build_grid(5, 5, 4, jitter=0.0)
    
    final_r = _compute_radius(best_centers)
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        best_centers = _build_grid(5, 5, 4, jitter=0.0)
    
    return best_centers
