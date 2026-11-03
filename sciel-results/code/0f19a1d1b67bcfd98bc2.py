import numpy as np
import math
import time
import scipy.optimize as so
import scipy.stats.qmc as qmc

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n

def _energy_and_grad_pen(x_flat, lam=100.0):
    """
    Compute energy and gradient for penalty method.
    x_flat: shape (3N,)
    """
    pts = x_flat.reshape(N, 3)
    # Compute distances
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of log energy
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    grad_log = -row_sums * pts + (W @ pts)
    
    # Penalty term: lam * sum(||x_i||^2 - 1)^2
    norms_sq = sq_norms
    penalty = lam * np.sum((norms_sq - 1.0) ** 2)
    grad_pen = 4.0 * lam * (norms_sq - 1.0)[:, np.newaxis] * pts
    
    total_energy = energy + penalty
    total_grad = grad_log + grad_pen
    
    return total_energy, total_grad.flatten()

def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_init(n, seed=0):
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    pts = sampler.random(n)
    # Map [0,1]^3 to sphere
    # Simple mapping: normalize
    return _project(pts)

def _random_init(n, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    return _project(pts)

def _minimize_from_init(init_pts, deadline):
    """
    Run L-BFGS-B on the penalty objective.
    """
    x0 = init_pts.flatten()
    
    # We need to stop if time runs out
    # L-BFGS-B doesn't have a direct time limit, but we can check in the callback
    # However, for short runs, it's often better to just let it run and check time after.
    # Or use maxiter.
    
    # Estimate iterations: 100 points, 300 dims. 500 iterations should be enough for local min.
    # But we have time constraints.
    
    res = so.minimize(
        _energy_and_grad_pen, 
        x0, 
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Project back to sphere to ensure validity
    best_pts = _project(res.x.reshape(N, 3))
    return best_pts, res.fun

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    initializations = []
    
    # Fibonacci
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # Sobol
    for seed in range(5):
        initializations.append(_sobol_init(N, seed=seed))
        
    # Random
    for seed in range(10):
        initializations.append(_random_init(N, seed=seed))
        
    best_pts = None
    best_energy = float('inf')
    
    # Evaluate initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        # Quick energy eval (without penalty)
        e = _energy_only(init_pts)
        scored_inits.append((e, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    top_candidates = scored_inits[:10]
    
    for i, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 10.0:
            break
            
        # Try L-BFGS-B
        try:
            opt_pts, opt_e = _minimize_from_init(init_pts, deadline)
            # Check validity
            if _valid(opt_pts):
                # Precise energy
                pe = _precise_energy(opt_pts)
                if pe < best_energy:
                    best_energy = pe
                    best_pts = opt_pts
        except Exception as e:
            pass
            
    # Final polish on the best
    if best_pts is not None and time.time() < deadline - 5.0:
        # Run another round of L-BFGS-B from the best point
        try:
            opt_pts, opt_e = _minimize_from_init(best_pts, deadline)
            if _valid(opt_pts):
                pe = _precise_energy(opt_pts)
                if pe < best_energy:
                    best_energy = pe
                    best_pts = opt_pts
        except Exception as e:
            pass
            
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
        
    best_pts = _project(best_pts)
    return best_pts

def _energy_only(pts):
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-16)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))

def _precise_energy(pts):
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True
