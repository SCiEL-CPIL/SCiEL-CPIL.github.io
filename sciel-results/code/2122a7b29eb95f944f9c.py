"""Heilbronn convex regions: place 13 points to maximize min triangle area / hull area."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

N = 13

# EVOLVE-BLOCK-START
def _score(pts):
    """Compute min triangle area divided by convex hull area."""
    n = len(pts)
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0
    if hull_area <= 1e-15:
        return 0.0
    
    # Compute all triangle areas
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area of triangle using cross product
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _score_vectorized(pts):
    """Faster vectorized score computation."""
    n = len(pts)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-15:
        return 0.0
    
    # Vectorized triangle area computation
    # For each triple (i,j,k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_area:
                    min_area = area
    
    return min_area / hull_area


def _fermat_spiral(n, scale=1.0):
    """Generate n points using Fermat/Vogel spiral for uniform distribution."""
    i = np.arange(n, dtype=float)
    r = scale * np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle in radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    # Center
    pts -= np.mean(pts, axis=0)
    return pts


def _concentric_circles(n):
    """Generate n points on concentric circles."""
    pts = []
    # Place points in rings
    # For n=13: center + ring of 6 + ring of 6
    # Or: 1 + 5 + 7, etc.
    # Let's try: center, ring1 (5 pts), ring2 (7 pts)
    pts.append([0.0, 0.0])
    r1 = 0.5
    for j in range(5):
        theta = 2*np.pi*j/5
        pts.append([r1*np.cos(theta), r1*np.sin(theta)])
    r2 = 1.0
    for j in range(7):
        theta = 2*np.pi*j/7 + np.pi/7  # offset
        pts.append([r2*np.cos(theta), r2*np.sin(theta)])
    return np.array(pts)


def _regular_polygon_ring(n_outer, r=1.0, offset=0.0):
    """Regular polygon with n_outer vertices at radius r."""
    thetas = offset + 2*np.pi*np.arange(n_outer)/n_outer
    return np.column_stack([r*np.cos(thetas), r*np.sin(thetas)])


def _hill_climb(pts, max_iters=500, step=0.05, time_limit=100):
    """Simple hill climbing on worst triangles."""
    start = time.time()
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Find worst triangle
        min_area = float('inf')
        worst_tri = None
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    a = current[i]
                    b = current[j]
                    c = current[k]
                    area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if area < min_area:
                        min_area = area
                        worst_tri = (i, j, k)
        
        if worst_tri is None:
            break
        
        i, j, k = worst_tri
        a = current[i]
        b = current[j]
        c = current[k]
        
        # Centroid of worst triangle
        centroid = (a + b + c) / 3.0
        
        # Try moving each vertex of the worst triangle
        improved = False
        for idx in [i, j, k]:
            p = current[idx].copy()
            # Try several perturbations
            for dx in [-step, -step*0.5, 0, step*0.5, step]:
                for dy in [-step, -step*0.5, 0, step*0.5, step]:
                    new_pts = current.copy()
                    new_pts[idx] = [p[0]+dx, p[1]+dy]
                    s = _score(new_pts)
                    if s > current_score + 1e-12:
                        current = new_pts
                        current_score = s
                        if s > best_score:
                            best = current.copy()
                            best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try smaller steps
            step *= 0.7
            if step < 1e-6:
                break
    
    return best, best_score


def _simulated_annealing(pts, max_iters=3000, t0=0.1, t1=0.001, time_limit=100):
    """Simulated annealing optimization."""
    start = time.time()
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    n = len(current)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Temperature
        alpha = it / max_iters
        T = t0 * (t1/t0) ** alpha
        
        # Random perturbation: move 1-3 random points
        n_move = np.random.randint(1, 4)
        indices = np.random.choice(n, size=n_move, replace=False)
        new_pts = current.copy()
        for idx in indices:
            noise = np.random.normal(0, T, size=2)
            new_pts[idx] = current[idx] + noise
        
        new_score = _score(new_pts)
        
        if new_score >= current_score or np.random.random() < np.exp((new_score - current_score) / max(T, 1e-10)):
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_pts = None
    best_score = -1.0
    
    # Multiple initialization strategies
    initializations = []
    
    # 1. Fermat spiral
    initializations.append(_fermat_spiral(N))
    
    # 2. Concentric circles
    initializations.append(_concentric_circles(N))
    
    # 3. Regular 13-gon with center
    pts_reg = []
    pts_reg.append([0.0, 0.0])
    for j in range(12):
        theta = 2*np.pi*j/12
        pts_reg.append([np.cos(theta), np.sin(theta)])
    initializations.append(np.array(pts_reg))
    
    # 4. Two rings: 5 + 8
    pts_two_ring = []
    r1, r2 = 0.55, 1.0
    for j in range(5):
        theta = 2*np.pi*j/5
        pts_two_ring.append([r1*np.cos(theta), r1*np.sin(theta)])
    for j in range(8):
        theta = 2*np.pi*j/8 + np.pi/8
        pts_two_ring.append([r2*np.cos(theta), r2*np.sin(theta)])
    initializations.append(np.array(pts_two_ring))
    
    # 5. Random uniform in unit disk
    np.random.seed(42)
    for _ in range(3):
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N))
        pts_rand = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        pts_rand -= np.mean(pts_rand, axis=0)
        initializations.append(pts_rand)
    
    # Evaluate all initializations
    for init in initializations:
        if time.time() > deadline - 30:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    # Hill climbing from best initialization
    if best_pts is not None:
        hc_pts, hc_score = _hill_climb(best_pts, max_iters=300, step=0.08, time_limit=40)
        if hc_score > best_score:
            best_score = hc_score
            best_pts = hc_pts
    
    # Simulated annealing from best
    if time.time() < deadline - 40:
        sa_pts, sa_score = _simulated_annealing(best_pts, max_iters=2000, t0=0.15, t1=0.005, time_limit=35)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts
    
    # Second round of hill climbing from SA result
    if best_pts is not None and time.time() < deadline - 20:
        hc2_pts, hc2_score = _hill_climb(best_pts, max_iters=200, step=0.03, time_limit=15)
        if hc2_score > best_score:
            best_score = hc2_score
            best_pts = hc2_pts
    
    # Final: try a few more random restarts if time permits
    if time.time() < deadline - 25:
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N))
            pts_r = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
            pts_r -= np.mean(pts_r, axis=0)
            
            s = _score(pts_r)
            if s > best_score * 0.95:  # Only pursue if promising
                hc_pts, hc_score = _hill_climb(pts_r, max_iters=150, step=0.06, time_limit=10)
                if hc_score > best_score:
                    best_score = hc_score
                    best_pts = hc_pts
    
    if best_pts is None:
        # Fallback
        best_pts = _fermat_spiral(N)
    
    return best_pts
# EVOLVE-BLOCK-END
