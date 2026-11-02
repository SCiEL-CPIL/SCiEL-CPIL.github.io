"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # seconds, safety margin from 150s limit

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _bary_to_cart(bary):
    """Convert barycentric (n,3) to Cartesian (n,2)."""
    return bary[:, 1:2] * V1 + bary[:, 2:3] * V2

def _cart_to_bary(pts):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    b = np.zeros((len(pts), 3))
    b[:, 2] = pts[:, 1] / S3H
    b[:, 1] = pts[:, 0] - 0.5 * b[:, 2]
    b[:, 0] = 1.0 - b[:, 1] - b[:, 2]
    return b

def _project_to_triangle(pts):
    """Project points to be inside the triangle."""
    b = _cart_to_bary(pts)
    b = np.clip(b, 0, 1)
    # Renormalize
    s = b.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    b = b / s
    return _bary_to_cart(b)

def _min_triangle_area(pts):
    """Compute min triangle area for N points, normalized by TRI_AREA."""
    n = len(pts)
    if n < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (y[j]-y[i])*(x[k]-x[i]))
                if a < min_a:
                    min_a = a
    return min_a / TRI_AREA

def _soft_min_area(pts, T=0.01):
    """Soft minimum of all triangle areas, normalized."""
    n = len(pts)
    x, y = pts[:, 0], pts[:, 1]
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = 0.5 * abs((x[j]-x[i])*(y[k]-y[i]) - (y[j]-y[i])*(x[k]-x[i]))
                areas.append(a)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    # Soft min using log-sum-exp
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum = max_a + T * np.log(np.sum(np.exp(-shifted / T)))
    return log_sum / TRI_AREA

def _bary_objective(bary_flat, T=0.01):
    """Objective in barycentric coords: minimize negative soft-min area."""
    uv = bary_flat.reshape(-1, 2)
    bary = np.zeros((N, 3))
    bary[:, 0] = 1.0 - uv[:, 0] - uv[:, 1]
    bary[:, 1] = uv[:, 0]
    bary[:, 2] = uv[:, 1]
    
    # Check if valid (all barycentric coords non-negative)
    if np.any(bary < -1e-10):
        return 1000.0 + np.sum(bary[bary < 0] ** 2)
    
    pts = _bary_to_cart(bary)
    return -_soft_min_area(pts, T)

def _bary_hard_objective(bary_flat):
    """Objective in barycentric coords: minimize negative min area (hard)."""
    uv = bary_flat.reshape(-1, 2)
    bary = np.zeros((N, 3))
    bary[:, 0] = 1.0 - uv[:, 0] - uv[:, 1]
    bary[:, 1] = uv[:, 0]
    bary[:, 2] = uv[:, 1]
    
    if np.any(bary < -1e-10):
        return 1000.0 + np.sum(bary[bary < 0] ** 2)
    
    pts = _bary_to_cart(bary)
    return -_min_triangle_area(pts)

def _penalized_hard_min(x):
    """Penalized hard min objective in Cartesian coords."""
    pts = x.reshape(-1, 2)
    # Penalty for being outside triangle
    penalty = 0.0
    for px, py in pts:
        # Check three half-planes
        d1 = py  # bottom edge: y >= 0
        d2 = S3H - py  # top edge: y <= S3H
        d3 = py - S3H * (1 - px)  # right slant: y <= S3H*(1-x)
        d4 = py - S3H * px  # left slant: y <= S3H*x
        for d in [d1, d2, d3, d4]:
            if d < 0:
                penalty += d * d
    return -_min_triangle_area(pts) + 100.0 * penalty

def _farthest_point_sampling(rng, n_samples=500):
    """Generate initial configuration using farthest point sampling."""
    bary = np.zeros((N, 3))
    bary[0] = [1/3, 1/3, 1/3]
    pts = _bary_to_cart(bary)
    
    for i in range(1, N):
        best_bary = None
        best_dist = -1
        for _ in range(n_samples):
            u, v = rng.random(), rng.random()
            if u + v <= 1:
                cand = np.array([1-u-v, u, v])
                cand_pt = _bary_to_cart(cand.reshape(1,3))[0]
                min_d = np.min(np.linalg.norm(pts[:i] - cand_pt, axis=1))
                if min_d > best_dist:
                    best_dist = min_d
                    best_bary = cand
        if best_bary is not None:
            bary[i] = best_bary
            pts = np.vstack([pts, _bary_to_cart(best_bary.reshape(1,3))])
    return bary

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution in barycentric coords
    bounds = [(0, 1)] * (2 * N)
    
    # Multiple DE runs with different populations
    de_configs = [
        {'popsize': 30, 'maxiter': 200, 'mutation': (0.5, 1.0), 'recombination': 0.8, 'seed': 42},
        {'popsize': 40, 'maxiter': 150, 'mutation': (0.8, 1.0), 'recombination': 0.9, 'seed': 123},
        {'popsize': 25, 'maxiter': 250, 'mutation': (0.3, 1.0), 'recombination': 0.7, 'seed': 456},
    ]
    
    for cfg in de_configs:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            # Create initial population
            pop = np.random.RandomState(cfg['seed']).random((cfg['popsize'], 2 * N))
            # Ensure valid barycentric: u+v <= 1
            for i in range(pop.shape[0]):
                u, v = pop[i, :N], pop[i, N:]
                s = u + v
                if s > 1:
                    scale = 1.0 / s
                    pop[i, :N] *= scale
                    pop[i, N:] *= scale
            
            res = minimize(
                lambda x: _bary_objective(x, T=0.02),
                pop,
                method='differential_evolution',
                bounds=bounds,
                options={'maxiter': cfg['maxiter'], 'popsize': cfg['popsize'],
                         'mutation': cfg['mutation'], 'recombination': cfg['recombination'],
                         'seed': cfg['seed'], 'polish': False, 'init': pop}
            )
            
            if res.success or res.fun < 1000:
                uv = res.x.reshape(-1, 2)
                bary = np.zeros((N, 3))
                bary[:, 0] = 1.0 - uv[:, 0] - uv[:, 1]
                bary[:, 1] = uv[:, 0]
                bary[:, 2] = uv[:, 1]
                
                if np.all(bary >= -1e-9):
                    pts = _bary_to_cart(bary)
                    pts = _project_to_triangle(pts)
                    score = _min_triangle_area(pts)
                    if score > best_score:
                        best_score = score
                        best_pts = pts.copy()
                        
        except Exception:
            continue
    
    # Phase 2: Simulated Annealing polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        current_pts = best_pts.copy()
        current_score = best_score
        T_start = 0.05
        T_end = 0.001
        n_iter = 5000
        step_size = 0.05
        
        for i in range(n_iter):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            T = T_start * (T_end / T_start) ** (i / n_iter)
            
            # Perturb
            new_pts = current_pts + rng.randn(N, 2) * step_size * (1 + i/n_iter * 0.5)
            new_pts = _project_to_triangle(new_pts)
            
            new_score = _min_triangle_area(new_pts)
            
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
    
    # Phase 3: Local search with Nelder-Mead
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        try:
            res_nm = minimize(
                _penalized_hard_min,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            polished = _project_to_triangle(res_nm.x.reshape(-1, 2))
            score = _min_triangle_area(polished)
            if score > best_score:
                best_score = score
                best_pts = polished
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        bary = _farthest_point_sampling(rng)
        best_pts = _bary_to_cart(bary)
    
    return best_pts
# EVOLVE-BLOCK-END
