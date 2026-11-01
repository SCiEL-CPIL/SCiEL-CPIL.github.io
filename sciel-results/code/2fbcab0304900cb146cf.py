"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools

N = 14
DEADLINE = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14
DEADLINE = 140.0

def _min_triangle_area(pts):
    """Compute the minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    # Vectorized: compute all triangle areas
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume = area
    except Exception:
        # Fallback: use polygon area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except:
            # Compute area via shoelace on sorted points
            order = np.lexsort((pts[:,1], pts[:,0]))
            p = pts[order]
            n = len(p)
            area = 0.0
            for i in range(n):
                j = (i+1) % n
                area += p[i,0]*p[j,1] - p[j,0]*p[i,1]
            return abs(area) / 2.0

def _score(pts):
    """Score = min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    min_ta = _min_triangle_area(pts)
    return min_ta / hull_a

def _objective(x):
    """Objective to minimize (negative score)."""
    pts = _unpack(x)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s

def _unpack(x):
    """Unpack parameter vector to point array. Fix p0=(0,0), p1=(1,0)."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    for i in range(N - 2):
        pts[i+2, 0] = x[2*i]
        pts[i+2, 1] = x[2*i+1]
    return pts

def _pack(pts):
    """Pack point array to parameter vector (excluding p0, p1)."""
    x = np.zeros(2*(N-2))
    for i in range(N - 2):
        x[2*i] = pts[i+2, 0]
        x[2*i+1] = pts[i+2, 1]
    return x

def _regular_polygon_init():
    """Regular 14-gon, centered, scaled so p0=(0,0), p1=(1,0)."""
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    # Translate so p0 is at origin
    pts = pts - pts[0]
    # Rotate/scale so p1 is at (1,0)
    # p1 is at angle 2*pi/14
    # Actually let's just do it properly: translate p0 to origin
    # Then we need p1 at (1,0). p1 - p0 direction
    p0 = pts[0]
    p1 = pts[1]
    # Translate
    pts = pts - p0
    # Now p0 is at origin, rotate so p1 is along x-axis
    angle = np.arctan2(p1[1], p1[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ R.T
    # Scale so p1 is at distance 1
    d = np.linalg.norm(p1)
    if d > 1e-12:
        pts = pts / d
    return pts

def _two_ring_init():
    """Two concentric rings: 7 points on outer, 7 on inner."""
    n_outer = 7
    n_inner = 7
    r_outer = 1.0
    r_inner = 0.5
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(np.pi/n_outer, 2*np.pi, n_inner, endpoint=False)  # offset
    pts_outer = np.column_stack((r_outer*np.cos(angles_outer), r_outer*np.sin(angles_outer)))
    pts_inner = np.column_stack((r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)))
    pts = np.vstack([pts_outer, pts_inner])
    # Translate p0 to origin
    pts = pts - pts[0]
    # Rotate so p1 is along x-axis
    p1 = pts[1]
    angle = np.arctan2(p1[1], p1[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ R.T
    # Scale so p1 is at distance 1
    d = np.linalg.norm(pts[1])
    if d > 1e-12:
        pts = pts / d
    return pts

def _three_ring_init():
    """Three rings: 5+5+4 points."""
    r1, r2, r3 = 1.0, 0.65, 0.35
    n1, n2, n3 = 5, 5, 4
    a1 = np.linspace(0, 2*np.pi, n1, endpoint=False)
    a2 = np.linspace(np.pi/n2, 2*np.pi, n2, endpoint=False)
    a3 = np.linspace(0, 2*np.pi, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack((r1*np.cos(a1), r1*np.sin(a1))),
        np.column_stack((r2*np.cos(a2), r2*np.sin(a2))),
        np.column_stack((r3*np.cos(a3), r3*np.sin(a3)))
    ])
    pts = pts - pts[0]
    p1 = pts[1]
    angle = np.arctan2(p1[1], p1[0])
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ R.T
    d = np.linalg.norm(pts[1])
    if d > 1e-12:
        pts = pts / d
    return pts

def _coordinate_descent_polish(pts, max_iter=200, tol=1e-10):
    """Polish points via coordinate descent on each coordinate."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    for iteration in range(max_iter):
        improved = False
        for i in range(N):
            for coord in range(2):
                # Try small perturbations
                orig = best_pts[i, coord]
                for step in [0.01, 0.005, 0.001, 0.0005, -0.0005, -0.001, -0.005, -0.01]:
                    new_pts = best_pts.copy()
                    new_pts[i, coord] = orig + step
                    s = _score(new_pts)
                    if s > best_score + tol:
                        best_pts = new_pts
                        best_score = s
                        improved = True
                        orig = new_pts[i, coord]
        if not improved:
            break
    return best_pts, best_score

def _local_refine(x, time_limit=5.0):
    """Local refinement using scipy minimize from a starting point."""
    start_time = time.time()
    pts = _unpack(x)
    current_score = _score(pts)
    
    def obj(x):
        if time.time() - start_time > time_limit:
            return 1e10
        p = _unpack(x)
        s = _score(p)
        return -s if s > 0 else 1e10
    
    try:
        result = minimize(obj, x, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if result.fun < -current_score:
            return result.x, -result.fun
    except:
        pass
    return x, current_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_x = None
    
    # Define bounds for the free coordinates
    # Points should roughly be in a reasonable range
    bounds = [(-5, 5)] * (2*(N-2))
    
    # Initial configurations to try
    inits = []
    
    # 1. Regular polygon
    try:
        pts_reg = _regular_polygon_init()
        inits.append(_pack(pts_reg))
    except:
        pass
    
    # 2. Two-ring
    try:
        pts_2r = _two_ring_init()
        inits.append(_pack(pts_2r))
    except:
        pass
    
    # 3. Three-ring
    try:
        pts_3r = _three_ring_init()
        inits.append(_pack(pts_3r))
    except:
        pass
    
    # 4. Random inits
    rng = np.random.RandomState(N)
    for _ in range(3):
        x_rand = rng.uniform(-3, 3, 2*(N-2))
        inits.append(x_rand)
    
    # Evaluate initial configurations
    for x_init in inits:
        if time.time() - start_time > DEADLINE - 20:
            break
        pts = _unpack(x_init)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_x = x_init.copy()
    
    # Differential evolution from best init
    if time.time() - start_time < DEADLINE - 30 and best_x is not None:
        try:
            result = differential_evolution(
                _objective, bounds, x0=best_x,
                seed=N, maxiter=50, popsize=15,
                tol=1e-6, mutation=(0.5, 1.5), recombination=0.9,
                polish=False, updating='deferred',
                seed=N
            )
            if result.fun < -best_score:
                best_x = result.x
                best_score = -result.fun
        except Exception as e:
            pass
    
    # Local refinement
    if time.time() - start_time < DEADLINE - 10 and best_x is not None:
        refined_x, refined_score = _local_refine(best_x, time_limit=10.0)
        if refined_score > best_score:
            best_score = refined_score
            best_x = refined_x
    
    # Coordinate descent polish
    if time.time() - start_time < DEADLINE - 5 and best_x is not None:
        pts = _unpack(best_x)
        polished_pts, polished_score = _coordinate_descent_polish(pts, max_iter=100)
        if polished_score > best_score:
            best_score = polished_score
            best_x = _pack(polished_pts)
    
    # Final check
    if time.time() - start_time > DEADLINE - 2:
        # Return immediately
        pass
    
    if best_x is None:
        # Fallback: regular polygon
        pts = _regular_polygon_init()
        return pts
    
    pts = _unpack(best_x)
    return pts
# EVOLVE-BLOCK-END

