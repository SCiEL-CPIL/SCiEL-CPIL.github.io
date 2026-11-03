"""Packing 50 equal spheres in a unit cube [-1/2,1/2]^3 via simulated annealing + L-BFGS-B polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _compute_radius(centres):
    """Compute the largest common radius for given centres."""
    if centres.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centres)):
        return 0.0
    # Wall distances
    wall = np.min(np.minimum(0.5 - np.abs(centres), 0.5 + np.abs(centres)), axis=1)
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(wall))

def _negative_radius(centres):
    """Objective to minimize: negative of the radius."""
    return -_compute_radius(centres)

def _bcc_lattice(n):
    """Generate a BCC-like lattice initialization."""
    pts = []
    # Body-centered cubic: corners + body centers
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Corner
                pts.append([i - 0.5, j - 0.5, k - 0.5])
                # Body center
                pts.append([i - 0.5 + 0.5, j - 0.5 + 0.5, k - 0.5 + 0.5])
    pts = np.array(pts[:n])
    # Scale to fit
    return pts

def _random_init(n, seed=0):
    """Random initialization inside the cube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 3))

def _simple_grid(n):
    """Simple cubic grid initialization."""
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def _anneal_and_polish(init, start_time, max_anneal_time=60.0, max_polish_time=20.0):
    """Simulated annealing followed by L-BFGS-B polish."""
    t_start = time.time()
    centres = init.copy()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Simulated annealing
    T = 0.05  # initial temperature
    T_min = 0.001
    cooling = 0.995
    steps_per_temp = 50
    
    while T > T_min and (time.time() - t_start) < max_anneal_time:
        for _ in range(steps_per_temp):
            if time.time() > DEADLINE - 5:
                break
            # Random perturbation
            idx = np.random.randint(N)
            step = np.random.normal(0, 0.02, size=3)
            new_centres = centres.copy()
            new_centres[idx] += step
            # Clamp to cube
            new_centres[idx] = np.clip(new_centres[idx], -0.499, 0.499)
            
            new_radius = _compute_radius(new_centres)
            delta = new_radius - best_radius
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                centres = new_centres
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centres = centres.copy()
        T *= cooling
    
    # L-BFGS-B polish
    if (time.time() - t_start) < max_polish_time and (time.time()) < DEADLINE - 10:
        try:
            bounds = [(-0.4999, 0.4999)] * N * 3
            res = minimize(_negative_radius, best_centres.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success or res.fun < -best_radius:
                polished = res.x.reshape(N, 3)
                polished = np.clip(polished, -0.4999, 0.4999)
                pol_radius = _compute_radius(polished)
                if pol_radius > best_radius:
                    best_centres = polished
                    best_radius = pol_radius
        except Exception:
            pass
    
    return best_centres, best_radius

def construct():
    t0 = time.time()
    
    # Multiple initializations
    inits = [
        _simple_grid(N),
        _bcc_lattice(N),
        _random_init(N, seed=42),
        _random_init(N, seed=123),
        _random_init(N, seed=777),
    ]
    
    best_centres = None
    best_radius = 0.0
    
    for i, init in enumerate(inits):
        if time.time() > DEADLINE - 30:
            break
        max_anneal = max(10.0, 50.0 - i * 5.0)
        max_polish = max(5.0, 15.0 - i * 2.0)
        centres, radius = _anneal_and_polish(init, t0, max_anneal_time=max_anneal, max_polish_time=max_polish)
        if radius > best_radius:
            best_radius = radius
            best_centres = centres.copy()
    
    if best_centres is None:
        best_centres = _simple_grid(N)
    
    # Final validation
    best_centres = np.clip(best_centres, -0.499999, 0.499999)
    return best_centres
