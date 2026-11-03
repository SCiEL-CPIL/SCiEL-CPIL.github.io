"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    # i < j < k
    # p_i, p_j, p_k
    # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    
    # Indices for combinations
    idx_i, idx_j, idx_k = np.indices((N, N, N)).reshape(3, -1)
    # Filter i < j < k
    mask = (idx_i < idx_j) & (idx_j < idx_k)
    i = idx_i[mask]
    j = idx_j[mask]
    k = idx_k[mask]
    
    p_i = pts[i]
    p_j = pts[j]
    p_k = pts[k]
    
    # Vectorized cross product in 2D: (x1*y2 - y1*x2)
    # det = (p_j[0] - p_i[0]) * (p_k[1] - p_i[1]) - (p_j[1] - p_i[1]) * (p_k[0] - p_i[0])
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _halton_sequence(n, dim=2):
    """Generate n Halton sequence points in dim dimensions."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, base) for base in [2, 3]])
    return np.array(points)

def _generate_starts(rng, num_starts):
    """Generate diverse starting points."""
    starts = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    starts.append(p_reg)
    
    # 2. Jittered Regular Polygons
    for _ in range(5):
        noise = rng.normal(0, 0.05, size=(N, 2))
        p_jit = p_reg + noise
        starts.append(p_jit)
        
    # 3. Halton Sequence points (scaled to fit in [-1, 1])
    halton_pts = _halton_sequence(N)
    # Normalize to [-1, 1] roughly
    halton_pts = (halton_pts - 0.5) * 2.0
    starts.append(halton_pts)
    
    # 4. Random points
    for _ in range(5):
        p_rand = rng.uniform(-1, 1, size=(N, 2))
        starts.append(p_rand)
        
    # 5. "Ring" points with random radii
    for _ in range(5):
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = rng.uniform(0.8, 1.2, N)
        p_ring = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        starts.append(p_ring)

    return starts[:num_starts]

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate a pool of starting points
    starts = _generate_starts(rng, 50)
    
    best_pts = starts[0].copy()
    best_score = _compute_score(best_pts)
    
    # Iterate through starts and refine with Nelder-Mead
    for p0 in starts:
        if (time.time() - start_time) > time_limit - 2.0:
            break
            
        # Objective: Minimize -score
        def objective(params):
            pts = params.reshape(N, 2)
            score = _compute_score(pts)
            if score == 0.0:
                return 1e9
            return -score
        
        try:
            res = minimize(
                objective, 
                p0, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-12, 'adaptive': True}
            )
            
            # Check improvement
            pts_res = res.x.reshape(N, 2)
            score_res = _compute_score(pts_res)
            
            if score_res > best_score:
                best_score = score_res
                best_pts = pts_res.copy()
                
        except Exception:
            continue
            
    return best_pts

# EVOLVE-BLOCK-END
