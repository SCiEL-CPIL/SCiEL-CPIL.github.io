import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13
DEADLINE = 145  # safety margin before 150s kill

def _min_triangle_area(pts):
    """Compute minimum triangle area among all C(13,3)=286 triples."""
    n = len(pts)
    # Precompute cross products for all pairs
    # For triangle (i,j,k): area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    # Vectorize: compute all pairs differences
    # Use broadcasting: for each pair (i,j), compute cross with all k
    # More efficient: compute all cross products at once
    # Shape: (n, n, 2) differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    # cross product of diff[i,j] and diff[i,k] for all i,j,k
    # cross(a, b) = a_x * b_y - a_y * b_x
    # For triangle (i,j,k): cross(pts[j]-pts[i], pts[k]-pts[i])
    # = diff[i,j,0]*diff[i,k,1] - diff[i,j,1]*diff[i,k,0]
    cross = diff[:, :, 0] * diff[np.newaxis, :, 1] - diff[:, :, 1] * diff[np.newaxis, :, 0]
    # cross[i,j,k] = cross(pts[j]-pts[i], pts[k]-pts[i])
    # We need triangles where i < j < k
    # Set diagonal and symmetric to large value to exclude
    n_pts = len(pts)
    cross = cross.copy()
    # For i=j or j=k or i=k, cross is 0, but we only want i<j<k
    # Set invalid entries to large value
    for i in range(n_pts):
        for j in range(n_pts):
            if i == j:
                cross[i, j, :] = np.inf
            if i > j:
                cross[i, j, :] = np.inf  # only keep i < j
    # Now for each i < j, we need k > j
    # Actually simpler: just compute min over all i<j<k
    min_area = np.inf
    for i in range(n_pts - 2):
        for j in range(i + 1, n_pts - 1):
            for k in range(j + 1, n_pts):
                a = 0.5 * abs(cross[i, j, k])
                if a < min_area:
                    min_area = a
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / hull area. Higher is better."""
    pts = np.asarray(pts).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(pts)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def _neg_score_smooth(pts, alpha=100.0):
    """Smooth approximation of -score for L-BFGS-B.
    Uses soft-min on triangle areas, and hull area in denominator."""
    pts = np.asarray(pts).reshape(-1, 2)
    n_pts = len(pts)
    
    # Compute all triangle areas
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    cross = diff[:, :, 0] * diff[np.newaxis, :, 1] - diff[:, :, 1] * diff[np.newaxis, :, 0]
    
    # Collect all triangle areas for i<j<k
    areas = []
    for i in range(n_pts - 2):
        for j in range(i + 1, n_pts - 1):
            for k in range(j + 1, n_pts):
                areas.append(0.5 * abs(cross[i, j, k]))
    areas = np.array(areas)
    
    # Soft min: -log(sum(exp(-alpha * areas))) / alpha
    # This approximates min(areas) as alpha -> inf
    m = np.min(areas)
    soft_min = -np.log(np.sum(np.exp(-alpha * (areas - m)))) / alpha + m
    
    # Hull area
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    
    return -(soft_min / ha)

def _neg_score_exact(pts):
    """Exact negative score for Nelder-Mead."""
    return -_score(pts)

def _make_initializations():
    """Generate diverse initial point sets."""
    inits = []
    
    # 1. Concentric rings: 7 outer + 6 inner
    for r_scale in [0.3, 0.5, 0.7]:
        outer_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.random.uniform(0, 0.5)
        inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.random.uniform(0, 0.5)
        outer = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
        inner = r_scale * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
        pts = np.vstack([outer, inner])
        # Normalize to unit area
        ha = _hull_area(pts)
        if ha > 1e-10:
            pts = pts / np.sqrt(ha)
        inits.append(pts.copy())
    
    # 2. Jittered regular polygon
    for jitter in [0.0, 0.05, 0.1]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.uniform(-jitter, jitter, N)
        radii = 1.0 + np.random.uniform(-0.1, 0.1, N) * (jitter > 0)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        ha = _hull_area(pts)
        if ha > 1e-10:
            pts = pts / np.sqrt(ha)
        inits.append(pts.copy())
    
    # 3. Two concentric circles with random rotation
    for rot in [0, np.pi/7, np.pi/4]:
        n1, n2 = 7, 6
        a1 = np.linspace(0, 2*np.pi, n1, endpoint=False) + rot
        a2 = np.linspace(0, 2*np.pi, n2, endpoint=False) + rot + np.pi/n2
        r1, r2 = 1.0, 0.5
        pts1 = r1 * np.column_stack([np.cos(a1), np.sin(a1)])
        pts2 = r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        pts = np.vstack([pts1, pts2])
        ha = _hull_area(pts)
        if ha > 1e-10:
            pts = pts / np.sqrt(ha)
        inits.append(pts.copy())
    
    # 4. Random points in disk
    for _ in range(3):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        ha = _hull_area(pts)
        if ha > 1e-10:
            pts = pts / np.sqrt(ha)
        inits.append(pts.copy())
    
    return inits

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    best_pts = None
    best_score = -np.inf
    
    inits = _make_initializations()
    
    for init_idx, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        # Phase 1: L-BFGS-B on smooth objective
        try:
            res1 = minimize(
                _neg_score_smooth,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            pts1 = res1.x.reshape(-1, 2)
            s1 = _score(pts1)
            
            if s1 > best_score:
                best_score = s1
                best_pts = pts1.copy()
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead on exact objective
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:
            break
        
        try:
            res2 = minimize(
                _neg_score_exact,
                init_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            pts2 = res2.x.reshape(-1, 2)
            s2 = _score(pts2)
            
            if s2 > best_score:
                best_score = s2
                best_pts = pts2.copy()
        except Exception:
            pass
        
        # Phase 3: If we found a good point, polish with Nelder-Mead from there
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5 or best_pts is None:
            break
        
        try:
            res3 = minimize(
                _neg_score_exact,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            pts3 = res3.x.reshape(-1, 2)
            s3 = _score(pts3)
            
            if s3 > best_score:
                best_score = s3
                best_pts = pts3.copy()
        except Exception:
            pass
    
    # Final safety: if best_pts is None, return something valid
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit hull area
    ha = _hull_area(best_pts)
    if ha > 1e-10:
        best_pts = best_pts / np.sqrt(ha)
    
    return best_pts

# EVOLVE-BLOCK-END
