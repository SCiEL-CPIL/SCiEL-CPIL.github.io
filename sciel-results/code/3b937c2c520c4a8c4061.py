"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import itertools

def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples of points."""
    n = len(pts)
    # Vectorized computation: for each pair (i,j), compute cross product with all other points k
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    # We want the minimum over all i < j < k
    
    # For efficiency, we can iterate over pairs and compute areas with all other points
    # But for n=14, C(14,3) = 364, which is small. Let's just do it directly.
    
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices pts[i], pts[j], pts[k]
                # = 0.5 * |cross(pts[j] - pts[i], pts[k] - pts[i])|
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, this is the area
    except Exception:
        # If points are collinear or fewer than 3, return 0
        return 0.0

def _score(pts):
    """Score = minimum triangle area / convex hull area."""
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _regular_n_gon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack((center[0] + radius * np.cos(angles),
                           center[1] + radius * np.sin(angles)))
    return pts

def _two_nested_heptagons(phi, r_ratio):
    """
    Generate 14 points as two nested regular heptagons.
    Outer heptagon has radius 1, inner has radius r_ratio.
    Inner heptagon is rotated by phi relative to outer.
    """
    outer = _regular_n_gon(7, radius=1.0)
    inner = _regular_n_gon(7, radius=r_ratio)
    # Rotate inner by phi
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    rot = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
    inner = inner @ rot.T
    return np.vstack([outer, inner])

def _optimize_two_heptagons():
    """Optimize phi and r_ratio for two nested heptagons."""
    best_score = 0.0
    best_pts = None
    
    # Grid search over phi and r_ratio
    # phi in [0, pi/7] by symmetry (rotating by 2*pi/7 gives same config)
    # r_ratio in (0, 1]
    
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Coarse grid
    phi_vals = np.linspace(0, np.pi / 7, 200)
    r_vals = np.linspace(0.1, 1.0, 100)
    
    for r in r_vals:
        for phi in phi_vals:
            if time.time() > deadline:
                break
            pts = _two_nested_heptagons(phi, r)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
    
    if best_pts is None:
        return _regular_n_gon(14), 0.0
    
    # Refine around best
    best_phi = phi_vals[np.argmin(np.abs(np.array([_two_nested_heptagons(p, r) for p in phi_vals]).shape[0]))]
    # Actually, let's just do a local refinement
    best_r = r_vals[0]  # We need to track this properly
    
    # Let's redo with proper tracking
    best_score = 0.0
    best_pts = None
    best_phi = 0.0
    best_r = 0.5
    
    for r in r_vals:
        for phi in phi_vals:
            if time.time() > deadline:
                break
            pts = _two_nested_heptagons(phi, r)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()
                best_phi = phi
                best_r = r
    
    # Local refinement using golden section
    if best_pts is not None and time.time() < deadline:
        # Refine phi for fixed best_r
        def neg_score_phi(phi, r):
            pts = _two_nested_heptagons(phi, r)
            return -_score(pts)
        
        # Golden section search in [max(0, best_phi - 0.1), min(pi/7, best_phi + 0.1)]
        lo = max(0.0, best_phi - 0.1)
        hi = min(np.pi / 7, best_phi + 0.1)
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        
        for _ in range(30):
            if time.time() > deadline:
                break
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = neg_score_phi(x1, best_r)
            f2 = neg_score_phi(x2, best_r)
            if f1 < f2:
                hi = x2
            else:
                lo = x1
        
        best_phi = (lo + hi) / 2
        
        # Refine r for fixed best_phi
        def neg_score_r(r, phi):
            pts = _two_nested_heptagons(phi, r)
            return -_score(pts)
        
        lo_r = max(0.01, best_r - 0.1)
        hi_r = min(1.0, best_r + 0.1)
        
        for _ in range(30):
            if time.time() > deadline:
                break
            x1 = hi_r - gr * (hi_r - lo_r)
            x2 = lo_r + gr * (hi_r - lo_r)
            f1 = neg_score_r(x1, best_phi)
            f2 = neg_score_r(x2, best_phi)
            if f1 < f2:
                hi_r = x2
            else:
                lo_r = x1
        
        best_r = (lo_r + hi_r) / 2
        
        pts = _two_nested_heptagons(best_phi, best_r)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    return best_pts if best_pts is not None else _regular_n_gon(14), best_score

def _stochastic_local_search(pts, max_iter=50000):
    """Local search: perturb points, accept if score improves."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    
    start_time = time.time()
    deadline = 140.0
    
    rng = np.random.RandomState(42)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Perturb a random point
        new_pts = best_pts.copy()
        idx = rng.randint(0, N)
        new_pts[idx] += rng.normal(0, 0.05, 2)
        
        s = _score(new_pts)
        if s > best_score:
            best_score = s
            best_pts = new_pts
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Start with regular 14-gon
    pts_regular = _regular_n_gon(14)
    score_regular = _score(pts_regular)
    
    best_pts = pts_regular.copy()
    best_score = score_regular
    
    # Try two nested heptagons
    if time.time() < deadline:
        pts_hept, score_hept = _optimize_two_heptagons()
        if score_hept > best_score:
            best_score = score_hept
            best_pts = pts_hept.copy()
    
    # Try stochastic local search from best
    if time.time() < deadline:
        pts_sl, score_sl = _stochastic_local_search(best_pts, max_iter=30000)
        if score_sl > best_score:
            best_score = score_sl
            best_pts = pts_sl.copy()
    
    return best_pts

# EVOLVE-BLOCK-END
