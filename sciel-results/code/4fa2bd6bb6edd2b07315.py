"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DEADLINE = 115.0  # safety margin before 120s kill

def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq = np.sum(diff * diff, axis=-1)
    return sq

def _score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if min_dist ~ 0."""
    sq = _pairwise_sq_dists(pts)
    iu = np.triu_indices_from(sq, k=1)
    d2 = sq[iu]
    mn = d2.min()
    mx = d2.max()
    if mn < 1e-12:
        return np.inf
    return mx / mn

def _soft_score(pts, beta=50.0):
    """Soft surrogate: smooth max/min via log-sum-exp on squared distances."""
    sq = _pairwise_sq_dists(pts)
    iu = np.triu_indices_from(sq, k=1)
    d2 = sq[iu]
    d2 = np.maximum(d2, 1e-12)
    log_max = np.log(np.sum(np.exp(d2 / beta))) - np.log(len(d2)) + np.log(len(d2))
    # Actually: logsumexp(d2/beta)*beta gives soft max of d2
    # soft max of d2 ≈ beta * logsumexp(d2/beta)
    # soft min of d2 ≈ -beta * logsumexp(-d2/beta)
    soft_max = beta * (np.log(np.sum(np.exp(d2 / beta))) - np.log(len(d2)))
    soft_min = -beta * (np.log(np.sum(np.exp(-d2 / beta))) - np.log(len(d2)))
    if soft_min < 1e-12:
        return np.inf
    return soft_max / soft_min

def _hex_init():
    """Regular hexagonal tiling subset of 16 points."""
    # Use a hexagonal lattice: spacing 1, 16 points in a compact arrangement
    # Rows of 4,4,4,4 with alternating offsets
    pts = []
    row_width = 4
    for r in range(4):
        offset = 0.5 if r % 2 == 1 else 0.0
        for c in range(row_width):
            x = c + offset
            y = r * np.sqrt(3) / 2
            pts.append([x, y])
    return np.array(pts, dtype=float)

def _square_init():
    """4x4 grid."""
    pts = []
    for r in range(4):
        for c in range(4):
            pts.append([c, r])
    return np.array(pts, dtype=float)

def _ring_init():
    """Points on concentric rings: 4 inner, 4 middle, 8 outer."""
    pts = []
    # Inner ring: 4 points at radius 0.5
    for k in range(4):
        angle = 2 * np.pi * k / 4
        pts.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
    # Middle ring: 4 points at radius 1.2
    for k in range(4):
        angle = 2 * np.pi * (k + 0.5) / 4
        pts.append([1.2 * np.cos(angle), 1.2 * np.sin(angle)])
    # Outer ring: 8 points at radius 2.0
    for k in range(8):
        angle = 2 * np.pi * k / 8
        pts.append([2.0 * np.cos(angle), 2.0 * np.sin(angle)])
    return np.array(pts, dtype=float)

def _random_init(seed=0):
    """Random points in a unit square."""
    rng = np.random.default_rng(seed)
    return rng.random((16, 2))

def _flatten(pts):
    return pts.flatten()

def _unflatten(x):
    return x.reshape(16, 2)

def _optimize_from(pts0, deadline):
    """Run multi-stage optimization from initial point."""
    best = pts0.copy()
    best_score = _score(best)
    
    # Stage 1: Differential Evolution for global search
    x0 = _flatten(pts0)
    bounds = [(-5, 5)] * 32
    
    def de_obj(x):
        return _soft_score(_unflatten(x), beta=20.0)
    
    try:
        res_de = differential_evolution(
            de_obj, bounds, seed=42, maxiter=100, tol=1e-4,
            popsize=15, polish=False, init='sobol'
        )
        if time.time() > deadline - 30:
            return best, best_score
        pts_de = _unflatten(res_de.x)
        s_de = _score(pts_de)
        if s_de < best_score:
            best = pts_de
            best_score = s_de
    except Exception:
        pass
    
    # Stage 2: Powell refinement
    x_start = _flatten(best)
    try:
        res_pow = minimize(
            lambda x: _soft_score(_unflatten(x), beta=10.0),
            x_start, method='Powell',
            options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-10}
        )
        if time.time() > deadline - 15:
            return best, best_score
        pts_pow = _unflatten(res_pow.x)
        s_pow = _score(pts_pow)
        if s_pow < best_score:
            best = pts_pow
            best_score = s_pow
    except Exception:
        pass
    
    # Stage 3: L-BFGS-B on exact objective
    x_start = _flatten(best)
    try:
        res_lb = minimize(
            lambda x: _score(_unflatten(x)),
            x_start, method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-10}
        )
        if time.time() > deadline - 5:
            return best, best_score
        pts_lb = _unflatten(res_lb.x)
        s_lb = _score(pts_lb)
        if s_lb < best_score:
            best = pts_lb
            best_score = s_lb
    except Exception:
        pass
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Multi-start from different initializations
    inits = [
        ('hex', _hex_init()),
        ('square', _square_init()),
        ('ring', _ring_init()),
        ('rand0', _random_init(0)),
        ('rand1', _random_init(1)),
        ('rand2', _random_init(2)),
    ]
    
    best_pts = None
    best_score = np.inf
    
    for name, pts0 in inits:
        if time.time() > deadline - 20:
            break
        pts, sc = _optimize_from(pts0, deadline)
        if sc < best_score:
            best_score = sc
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init()
    
    return best_pts
# EVOLVE-BLOCK-END
