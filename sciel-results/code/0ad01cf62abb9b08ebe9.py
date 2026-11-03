"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

# Precompute all triangle indices for N=13: C(13,3)=286
_TRIPLETS = np.array(list(itertools.combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _min_triangle_area_raw(pts):
    """Compute minimum triangle area for n points (vectorized)."""
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        # ConvexHull requires at least 3 non-collinear points
        if len(pts) < 3:
            return 1.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        # Fallback: approximate area if hull fails (e.g., collinear)
        # For 13 points in 2D, this is unlikely unless degenerate
        return 1.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_ta = _min_triangle_area_raw(pts)
    return min_ta / hull_a

def _soft_min_area(pts, beta):
    """Soft-min surrogate for minimum triangle area (vectorized)."""
    a = pts[_TRIPLETS[:, 0]]
    b = pts[_TRIPLETS[:, 1]]
    c = pts[_TRIPLETS[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    m = np.min(areas)
    if m < 1e-15:
        return m
    # Log-sum-exp trick for stability
    # log(sum(exp(-beta * (areas - m)))) = log(sum(exp(-beta * areas + beta * m)))
    # = beta * m + log(sum(exp(-beta * areas)))
    # soft_min = m - (1/beta) * log(sum(exp(-beta * (areas - m))))
    #          = m - (1/beta) * (beta * m + log(sum(exp(-beta * areas))))
    #          = m - m - (1/beta) * log(sum(exp(-beta * areas)))
    #          = -(1/beta) * log(sum(exp(-beta * areas)))
    # This is numerically more stable if we shift by max.
    shifted = -beta * (areas - m)
    # shifted values are <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - (1.0 / beta) * log_sum

def _objective_factory(beta):
    """Factory for the smooth objective function to minimize."""
    def obj(x):
        pts = x.reshape(N, 2)
        hull_a = _hull_area(pts)
        if hull_a < 1e-12:
            return 1e6
        soft_min = _soft_min_area(pts, beta)
        # We want to maximize soft_min / hull_a
        # So minimize - (soft_min / hull_a)
        return -(soft_min / hull_a)
    return obj

def _generate_starts():
    """Generate a diverse set of starting points."""
    starts = []
    
    # 1. Regular polygons
    for radius in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for rot in np.linspace(0, 2*np.pi, 5, endpoint=False):
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + rot
            x = 0.5 + radius * np.cos(angles)
            y = 0.5 + radius * np.sin(angles)
            starts.append(np.column_stack((x, y)))
            
    # 2. Halton-like sequences (low discrepancy)
    # Simple Halton base 2 and 3
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    n_halton = 200
    h1 = np.array([halton(i, 2) for i in range(1, n_halton+1)])
    h2 = np.array([halton(i, 3) for i in range(1, n_halton+1)])
    # Scale to [0,1] and center
    h1 = 0.5 + 0.4 * (h1 - 0.5)
    h2 = 0.5 + 0.4 * (h2 - 0.5)
    starts.append(np.column_stack((h1[:N], h2[:N])))
    
    # 3. Random uniform in unit circle
    np.random.seed(42)
    for _ in range(10):
        r = np.sqrt(np.random.rand(N))
        theta = 2 * np.pi * np.random.rand(N)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        starts.append(np.column_stack((x, y)))
        
    # 4. Center + Rings
    for r1 in [0.2, 0.3, 0.4]:
        for r2 in [0.5, 0.6, 0.7]:
            pts = [[0.5, 0.5]]
            n1 = 6
            n2 = 6
            a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
            a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi/n2
            x1 = 0.5 + r1 * np.cos(a1)
            y1 = 0.5 + r1 * np.sin(a1)
            x2 = 0.5 + r2 * np.cos(a2)
            y2 = 0.5 + r2 * np.sin(a2)
            pts.extend(zip(x1, y1))
            pts.extend(zip(x2, y2))
            starts.append(np.array(pts[:N]))
            
    return starts

def _polish_from_start(start_pts, deadline):
    """Optimize from a single start point using increasing beta."""
    current = start_pts.flatten()
    best_pts = start_pts.copy()
    best_score = _score(start_pts)
    
    # Sequence of betas for annealing
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        if time.time() > deadline - 1:
            break
        try:
            obj = _objective_factory(beta)
            # Use L-BFGS-B as it's fast and handles bounds well
            # No explicit bounds needed if we trust the optimizer to stay in reasonable range,
            # but adding loose bounds [0,1] helps prevent explosion
            bounds = [(0.0, 1.0)] * (N * 2)
            res = minimize(
                obj, 
                current, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current = res.x
            cand = current.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    starts = _generate_starts()
    
    best_pts = starts[0].copy()
    best_score = _score(best_pts)
    
    # Evaluate all starts first to find the best initial guess
    for s in starts[1:]:
        sc = _score(s)
        if sc > best_score:
            best_score = sc
            best_pts = s.copy()
            
    # Sort starts by score (descending) to optimize the best ones first
    scored_starts = []
    for s in starts:
        scored_starts.append((_score(s), s))
    scored_starts.sort(key=lambda x: x[0], reverse=True)
    
    # Polish the top starts
    # We have ~40 starts. Polishing takes ~0.1-0.5s each.
    # 140s / 0.5s = 280 optimizations possible. We have plenty of time.
    # But let's be conservative and polish the top 20 or until time runs out.
    
    for sc, s in scored_starts:
        if time.time() > deadline - 5:
            break
        try:
            polished, polished_score = _polish_from_start(s, deadline)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
