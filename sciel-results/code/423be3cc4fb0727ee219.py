"""Heilbronn triangles: maximize smallest triangle area among 11 points in unit equilateral triangle."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 11
T_START = time.time()
TIME_LIMIT = 145.0  # safety margin before 150s kill

# Triangle vertices
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3)/2]])

def barycentric_to_cartesian(bary):
    """Convert barycentric coords (n,3) to Cartesian (n,2)."""
    return bary @ V

def cartesian_to_barycentric(pts):
    """Convert Cartesian (n,2) to barycentric (n,3)."""
    # Solve: pts = bary @ V
    # Using the fact that for equilateral triangle, we can use linear transform
    n = pts.shape[0]
    bary = np.zeros((n, 3))
    bary[:, 0] = 1.0 - pts[:, 0] - pts[:, 1] / np.sqrt(3)
    bary[:, 1] = pts[:, 1] / np.sqrt(3)
    bary[:, 2] = pts[:, 0] - pts[:, 1] / np.sqrt(3)
    return bary

def triangle_area(a, b, c):
    """Area of triangle with vertices a, b, c."""
    return 0.5 * abs(np.cross(b - a, c - a))

def all_triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                areas.append(triangle_area(pts[i], pts[j], pts[k]))
    return np.array(areas)

def min_area_value(pts):
    """The objective: smallest triangle area."""
    return np.min(all_triangle_areas(pts))

def soft_min(areas, beta):
    """Soft-min approximation using log-sum-exp."""
    # soft_min = -logsumexp(-areas/beta) * beta
    return -beta * logsumexp(-areas / beta)

def objective_bary(bary, beta):
    """Objective in barycentric coordinates: maximize soft-min of triangle areas."""
    # Clip barycentric coords to be non-negative (soft penalty)
    bary_clipped = np.maximum(bary, 0.0)
    # Normalize so they sum to 1 (project to simplex)
    bary_clipped = bary_clipped / bary_clipped.sum(axis=1, keepdims=True)
    
    pts = barycentric_to_cartesian(bary_clipped)
    areas = all_triangle_areas(pts)
    # We want to maximize the soft-min, so minimize negative
    return -soft_min(areas, beta)

def objective_bary_grad(bary, beta):
    """Objective with numerical gradient (L-BFGS-B will compute if not provided)."""
    return objective_bary(bary, beta)

def make_initial_bary(strategy, rng):
    """Generate initial barycentric coordinates based on strategy."""
    if strategy == 'random':
        bary = rng.random((N, 3))
        bary = bary / bary.sum(axis=1, keepdims=True)
    elif strategy == 'grid':
        # Place points on a structured grid in barycentric space
        bary = np.zeros((N, 3))
        # Corners and midpoints and center
        corners = np.array([[1,0,0],[0,1,0],[0,0,1]])
        midpoints = np.array([[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
        center = np.array([[1/3,1/3,1/3]])
        # Add some more structured points
        extra = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])
        all_pts = np.vstack([corners, midpoints, center, extra])
        bary = all_pts[:N]
    elif strategy == 'structured':
        # Try a more careful initial placement
        # Use known good configurations as starting points
        # For 11 points, try a pattern based on known results
        bary = np.zeros((N, 3))
        # 3 corners
        bary[0] = [0.9, 0.05, 0.05]
        bary[1] = [0.05, 0.9, 0.05]
        bary[2] = [0.05, 0.05, 0.9]
        # 3 midpoints of edges
        bary[3] = [0.45, 0.45, 0.1]
        bary[4] = [0.45, 0.1, 0.45]
        bary[5] = [0.1, 0.45, 0.45]
        # Center
        bary[6] = [1/3, 1/3, 1/3]
        # 4 more points in a ring
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        for i, ang in enumerate(angles):
            r = 0.2
            bary[7+i] = [1/3 + r*np.cos(ang), 1/3 + r*np.sin(ang), 1/3 - r*(np.cos(ang)+np.sin(ang))/2]
            # Normalize
            bary[7+i] = np.maximum(bary[7+i], 0)
            bary[7+i] /= bary[7+i].sum()
    else:
        bary = rng.random((N, 3))
        bary = bary / bary.sum(axis=1, keepdims=True)
    return bary

def construct():
    """Main entry point: find 11 points maximizing smallest triangle area."""
    global T_START
    deadline = T_START + TIME_LIMIT
    rng = np.random.RandomState(42)
    
    best_pts = None
    best_val = -np.inf
    
    # Multiple restarts with different strategies and betas
    strategies = ['structured', 'grid', 'random', 'random', 'random', 'random', 'structured', 'grid']
    betas = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    attempt = 0
    for strat_idx, strat in enumerate(strategies):
        if time.time() > deadline - 5.0:
            break
        for beta in betas[:min(3, len(betas))]:
            if time.time() > deadline - 5.0:
                break
            bary0 = make_initial_bary(strat, rng)
            
            # Optimize in barycentric space
            try:
                res = minimize(
                    objective_bary,
                    bary0,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            bary_opt = res.x
            # Ensure valid barycentric coords
            bary_opt = np.maximum(bary_opt, 0)
            bary_opt = bary_opt / bary_opt.sum(axis=1, keepdims=True)
            
            pts = barycentric_to_cartesian(bary_opt)
            val = min_area_value(pts)
            
            if val > best_val:
                best_val = val
                best_pts = pts.copy()
            
            # Polish with very small beta for precision
            if time.time() > deadline - 3.0:
                break
            try:
                res2 = minimize(
                    objective_bary,
                    bary_opt,
                    args=(0.0001,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                bary_opt2 = res2.x
                bary_opt2 = np.maximum(bary_opt2, 0)
                bary_opt2 = bary_opt2 / bary_opt2.sum(axis=1, keepdims=True)
                pts2 = barycentric_to_cartesian(bary_opt2)
                val2 = min_area_value(pts2)
                if val2 > best_val:
                    best_val = val2
                    best_pts = pts2.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return seed-like random points
        pts = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1:
                u, v = rng.random(), rng.random()
            pts[i] = [u + 0.5*v, np.sqrt(3)/2 * v]
        return pts
    
    # Final validation: ensure all points are inside triangle
    # Clip any points that might be slightly outside
    for i in range(N):
        p = best_pts[i]
        # Check barycentric coordinates
        b = cartesian_to_barycentric(p.reshape(1,2))[0]
        if np.any(b < -1e-10):
            # Project back inside
            b = np.maximum(b, 0)
            b = b / b.sum()
            best_pts[i] = barycentric_to_cartesian(b.reshape(1,3))[0]
    
    return best_pts
