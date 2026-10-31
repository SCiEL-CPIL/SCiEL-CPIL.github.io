"""Rewritten program: multi-stage optimization for 16-point distance ratio minimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 110.0  # safety margin before 120s limit


def _pairwise_dists(pts):
    """Compute all pairwise distances from an (N, 2) array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dists


def _score(pts):
    """Compute (max_d / min_d)^2. Returns inf if any points coincide."""
    dists = _pairwise_dists(pts)
    # Exclude self-distances (diagonal)
    mask = ~np.eye(N, dtype=bool)
    d = dists[mask]
    min_d = d.min()
    max_d = d.max()
    if min_d < 1e-12:
        return np.inf
    return (max_d / min_d) ** 2


def _smooth_objective(pts, beta=50.0):
    """Smooth surrogate: soft-max / soft-min ratio squared via log-sum-exp."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    # Exclude diagonal
    sq_dists = sq_dists.copy()
    np.fill_diagonal(sq_dists, np.inf)
    
    # Use log-sum-exp for soft max and soft min of distances
    # Work with squared distances for numerical stability
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    if m_max < 1e-15 or m_min < 1e-15:
        return 1e10
    
    # Soft max: (1/beta) * log(sum(exp(beta * x)))
    exp_max = np.exp(beta * (sq_dists - m_max))
    exp_max[np.isinf(sq_dists)] = 0
    soft_max_sq = m_max + np.log(np.sum(exp_max)) / beta
    
    # Soft min: -(1/beta) * log(sum(exp(-beta * x)))
    exp_min = np.exp(-beta * (sq_dists - m_min))
    exp_min[np.isinf(sq_dists)] = 0
    soft_min_sq = m_min - np.log(np.sum(exp_min)) / beta
    
    if soft_min_sq < 1e-15:
        return 1e10
    
    ratio = soft_max_sq / soft_min_sq
    return ratio


def _hex_init():
    """Hexagonal lattice initialization for 16 points."""
    pts = []
    # Hexagonal packing: rows offset
    # Try to fit 16 points in a hex-like arrangement
    # 4-4-4-4 grid with hex offset
    s = 1.0
    for row in range(4):
        for col in range(4):
            x = col * s * np.sqrt(3) / 2 + (row % 2) * s * np.sqrt(3) / 4
            y = row * s * 1.5
            pts.append([x, y])
    pts = np.array(pts)
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _ring_init():
    """Two concentric rings: 8 + 8."""
    pts = []
    # Inner ring: 8 points
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append([np.cos(angle), np.sin(angle)])
    # Outer ring: 8 points
    R = 2.5
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi / 8  # offset
        pts.append([R * np.cos(angle), R * np.sin(angle)])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    return pts


def _square_init():
    """4x4 grid."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i - 1.5, j - 1.5])
    pts = np.array(pts)
    return pts


def _random_init(seed=42):
    """Random initialization in unit square, centered."""
    rng = np.random.default_rng(seed)
    pts = rng.random((N, D)) * 4 - 2
    pts -= pts.mean(axis=0)
    return pts


def _powell_optimize(pts, time_budget=10.0):
    """Powell optimization on the exact objective."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        p = x.reshape(N, D)
        s = _score(p)
        if not np.isfinite(s):
            return 1e10
        return s
    
    result = minimize(obj, pts.flatten(), method='Powell',
                      options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x.reshape(N, D)


def _lbfgs_smooth_optimize(pts, beta=50.0, time_budget=10.0):
    """L-BFGS-B on the smooth surrogate."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        p = x.reshape(N, D)
        return _smooth_objective(p, beta=beta)
    
    result = minimize(obj, pts.flatten(), method='L-BFGS-B',
                      options={'maxiter': 300, 'gtol': 1e-8})
    return result.x.reshape(N, D)


def _polish(pts, time_budget=8.0):
    """High-precision polish with tight tolerances."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_budget:
            return 1e10
        p = x.reshape(N, D)
        s = _score(p)
        if not np.isfinite(s):
            return 1e10
        return s
    
    result = minimize(obj, pts.flatten(), method='Powell',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x.reshape(N, D)


def construct():
    global DEADLINE
    deadline = time.time() + DEADLINE
    
    # Generate diverse initial candidates
    inits = [
        _hex_init(),
        _ring_init(),
        _square_init(),
        _random_init(42),
        _random_init(123),
        _random_init(777),
    ]
    
    best_pts = None
    best_score = np.inf
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        current = init.copy()
        
        # Stage 1: Powell on exact objective
        current = _powell_optimize(current, time_budget=8.0)
        s = _score(current)
        if s < best_score:
            best_score = s
            best_pts = current.copy()
        
        # Stage 2: L-BFGS on smooth surrogate (moderate beta)
        if time.time() > deadline - 25:
            break
        current = _lbfgs_smooth_optimize(current, beta=30.0, time_budget=8.0)
        s = _score(current)
        if s < best_score:
            best_score = s
            best_pts = current.copy()
        
        # Stage 3: Powell exact again
        if time.time() > deadline - 20:
            break
        current = _powell_optimize(current, time_budget=8.0)
        s = _score(current)
        if s < best_score:
            best_score = s
            best_pts = current.copy()
        
        # Stage 4: L-BFGS high-beta smooth
        if time.time() > deadline - 15:
            break
        current = _lbfgs_smooth_optimize(current, beta=100.0, time_budget=8.0)
        s = _score(current)
        if s < best_score:
            best_score = s
            best_pts = current.copy()
    
    # Final polish on best candidate
    if best_pts is not None and time.time() < deadline - 10:
        polished = _polish(best_pts, time_budget=8.0)
        s = _score(polished)
        if s < best_score:
            best_pts = polished
            best_score = s
    
    # Safety: ensure distinct points
    if best_pts is None:
        best_pts = _square_init()
    
    # Verify distinctness
    dists = _pairwise_dists(best_pts)
    mask = ~np.eye(N, dtype=bool)
    if dists[mask].min() < 1e-9:
        best_pts = _square_init()
    
    return best_pts
