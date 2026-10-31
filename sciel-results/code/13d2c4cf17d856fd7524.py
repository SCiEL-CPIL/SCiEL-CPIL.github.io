"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def _pairwise_dists(pts):
    """Compute all pairwise distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    # Get upper triangle
    iu = np.triu_indices(n, 1)
    return dists[iu]

def _score(pts):
    """Compute (max/min)^2."""
    d = _pairwise_dists(pts)
    return (d.max() / d.min())**2

def _soft_objective(pts, beta):
    """Smooth surrogate: use log-sum-exp for max, and -log-sum-exp for min (inverted)."""
    d = _pairwise_dists(pts)
    d = np.maximum(d, 1e-10)
    # Soft max via log-sum-exp
    log_max = np.log(np.sum(np.exp(beta * d))) / beta
    # Soft min: use negative log-sum-exp of 1/d
    inv_d = 1.0 / d
    log_min_inv = np.log(np.sum(np.exp(beta * inv_d))) / beta
    soft_min = 1.0 / log_min_inv
    return (log_max / soft_min)**2

def _exact_objective(pts):
    """Exact objective."""
    d = _pairwise_dists(pts)
    return (d.max() / d.min())**2

def _make_ring(n, radius, angle_offset=0.0):
    """Create n points on a ring of given radius."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _init_single_ring(n=16):
    """Regular 16-gon."""
    return _make_ring(n, 1.0)

def _init_two_rings():
    """8 inner + 8 outer, with rotation offset for symmetry."""
    inner = _make_ring(8, 0.6, angle_offset=0.0)
    outer = _make_ring(8, 1.3, angle_offset=np.pi/8)
    return np.vstack([inner, outer])

def _init_three_rings():
    """1 center + 5 inner + 10 outer."""
    center = np.array([[0.0, 0.0]])
    inner = _make_ring(5, 0.5, angle_offset=0.0)
    outer = _make_ring(10, 1.2, angle_offset=0.0)
    return np.vstack([center, inner, outer])

def _init_hexagonal():
    """Hexagonal lattice pattern, take 16 points."""
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts[:16])
    # Center and normalize
    pts -= pts.mean(axis=0)
    return pts

def _init_random(seed=42):
    """Random points in unit disk."""
    rng = np.random.RandomState(seed)
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2*np.pi, N)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    # Generate diverse initial configurations
    inits = [
        ("ring16", _init_single_ring()),
        ("two_rings", _init_two_rings()),
        ("three_rings", _init_three_rings()),
        ("hexagonal", _init_hexagonal()),
        ("random1", _init_random(42)),
        ("random2", _init_random(123)),
    ]
    
    # Multi-stage optimization with increasing beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    for name, init_pts in inits:
        if time.time() > deadline - 10:
            break
        
        pts = init_pts.copy()
        
        # Progressive optimization with increasing beta
        for beta in betas:
            if time.time() > deadline - 10:
                break
            
            def obj(x):
                p = x.reshape(N, D)
                return _soft_objective(p, beta)
            
            try:
                res = minimize(obj, pts.flatten(), method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-10})
                pts = res.x.reshape(N, D)
            except Exception:
                pass
        
        # Final polish with exact objective
        if time.time() > deadline - 10:
            break
        
        def obj_exact(x):
            p = x.reshape(N, D)
            return _exact_objective(p)
        
        try:
            res = minimize(obj_exact, pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12})
            pts = res.x.reshape(N, D)
        except Exception:
            pass
        
        sc = _score(pts)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: regular 16-gon
        best_pts = _init_single_ring()
    
    return best_pts
# EVOLVE-BLOCK-END
