import numpy as np
from scipy.optimize import minimize
from collections import deque
import time

N = 11

def get_hex_cluster_positions():
    """Get positions of N hexagons in a compact hex grid cluster via BFS."""
    visited = set()
    queue = deque([(0, 0)])
    visited.add((0, 0))
    axial = []
    while len(axial) < N and queue:
        q, r = queue.popleft()
        axial.append((q, r))
        for dq, dr in [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]:
            nq, nr = q+dq, r+dr
            if (nq, nr) not in visited:
                visited.add((nq, nr))
                queue.append((nq, nr))
    
    sqrt3 = np.sqrt(3)
    pts = []
    for q, r in axial:
        x = 2.0 * (q + r/2.0)
        y = sqrt3 * r
        pts.append([x, y])
    return np.array(pts)

def enclosing_hexagon_side(pts, cx, cy, theta):
    """Compute minimal side length of enclosing regular hexagon."""
    angles = theta + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    diff = pts - np.array([cx, cy])
    projections = diff @ normals.T
    max_proj = np.max(projections)
    
    return 2.0 * max_proj / np.sqrt(3)

def objective(params, pts):
    cx, cy, theta = params
    return enclosing_hexagon_side(pts, cx, cy, theta)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    pts = get_hex_cluster_positions()
    centroid = np.mean(pts, axis=0)
    
    best_side = float('inf')
    best_params = None
    
    # Try multiple initial rotations
    for theta_init in np.linspace(0, np.pi/3, 13):
        if time.time() > deadline:
            break
        x0 = [centroid[0], centroid[1], theta_init]
        try:
            res = minimize(objective, x0, args=(pts,), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_side = enclosing_hexagon_side(pts, centroid[0], centroid[1], 0.0)
        best_params = [centroid[0], centroid[1], 0.0]
    
    cx, cy, theta = best_params
    
    inner = []
    for i in range(N):
        inner.append([float(pts[i][0]), float(pts[i][1]), 0.0])
    
    return inner, (float(cx), float(cy)), float(best_side), float(theta)
