"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Precompute for speed
def _triangle_area(p1, p2, p3):
    """Area of triangle formed by 3 points (2D vectors)."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def _min_area(points):
    """Minimum area over all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a

def _barycentric_inside(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    # Using barycentric coordinates relative to A, B, C
    # p = a*A + b*B + c*C with a+b+c=1, a,b,c >= 0
    # Solve: p = B*b + C*c (since A=origin)
    # [B C] [b;c] = p
    M = np.array([[B[0], C[0]], [B[1], C[1]]])
    try:
        bc = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        return False
    b, c = bc
    a = 1.0 - b - c
    return (a >= -eps) and (b >= -eps) and (c >= -eps)

def _clip_to_triangle(p):
    """Clip a point to the interior of the triangle (small margin)."""
    # Project onto triangle using barycentric coordinates
    M = np.array([[B[0], C[0]], [B[1], C[1]]])
    try:
        bc = np.linalg.solve(M, p)
    except np.linalg.LinAlgError:
        return A.copy()
    b, c = bc
    a = 1.0 - b - c
    # Clip to [0,1] with small margin
    margin = 1e-4
    a = max(margin, min(1.0 - 2*margin, a))
    b = max(margin, min(1.0 - 2*margin, b))
    c = max(margin, min(1.0 - 2*margin, c))
    # Renormalize
    s = a + b + c
    a, b, c = a/s, b/s, c/s
    return b * B + c * C

def _penalty(points, mu=1e4):
    """Quadratic penalty for points outside the triangle."""
    pen = 0.0
    for p in points:
        if not _barycentric_inside(p, eps=-1e-6):
            # Distance from triangle (approximate: barycentric violation)
            M = np.array([[B[0], C[0]], [B[1], C[1]]])
            try:
                bc = np.linalg.solve(M, p)
            except np.linalg.LinAlgError:
                pen += mu * 1.0
                continue
            b, c = bc
            a = 1.0 - b - c
            viol = max(0, -a) + max(0, -b) + max(0, -c)
            pen += mu * viol * viol
    return pen

def _objective_flat(x, mu=1e4):
    """Objective: minimize -min_area + penalty. x is flat array of 2N coords."""
    pts = x.reshape(N, 2)
    # Clip to triangle for evaluation
    pts_clipped = np.array([_clip_to_triangle(p) for p in pts])
    ma = _min_area(pts_clipped)
    pen = _penalty(pts, mu)
    return -ma + pen

def _nelder_mead_start(x0, max_iter=2000, mu=1e4):
    """Run Nelder-Mead from a single start."""
    res = minimize(_objective_flat, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12,
                            'adaptive': True})
    return res

def _make_lattice(starts=20, n_grid=3):
    """Generate lattice-based starting configurations inside the triangle."""
    starts_list = []
    # Grid points in barycentric coordinates
    n = n_grid
    for i in range(n):
        for j in range(n - i):
            k = n - 1 - i - j
            a, b, c = i/(n-1), j/(n-1), k/(n-1)
            p = a*A + b*B + c*C
            # Add small margin
            p = A + 0.95 * (p - A)
            starts_list.append(p)
    return starts_list

def _random_config(rng, margin=0.05):
    """Generate a random configuration of N points inside the triangle."""
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                p = u * B + v * C
                # Apply margin from boundary
                p = A + (1 - 2*margin) * (p - A)
                pts.append(p)
                break
    return np.array(pts)

def _grid_config():
    """Generate a structured grid configuration."""
    pts = []
    # Place points in a pattern: 3 rows
    # Row 0 (bottom): 4 points
    # Row 1 (middle): 3 points
    # Row 2 (top): 2 points
    # Plus 2 more distributed
    h = np.sqrt(3)/2
    # Bottom row y = h*0.15
    y0 = h * 0.15
    x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in x_positions[:4]:
        pts.append([x, y0])
    # Middle row y = h*0.5
    y1 = h * 0.5
    for x in [0.2, 0.5, 0.8]:
        pts.append([x, y1])
    # Top row y = h*0.85
    y2 = h * 0.85
    for x in [0.35, 0.65]:
        pts.append([x, y2])
    # 2 more: center and one offset
    pts.append([0.5, h*0.35])
    pts.append([0.5, h*0.65])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = -float('inf')
    
    # Generate multiple starting configurations
    starts = []
    
    # 1. Grid-based structured start
    starts.append(_grid_config())
    
    # 2. Random starts
    for _ in range(15):
        starts.append(_random_config(rng))
    
    # 3. Lattice subset starts
    lattice = _make_lattice(starts=25, n_grid=4)
    # Pick 11 points from lattice with some randomness
    for _ in range(5):
        if len(lattice) >= N:
            idx = rng.choice(len(lattice), size=N, replace=False)
            starts.append(np.array([lattice[i] for i in idx]))
    
    # Add a few more random starts with different margins
    for margin in [0.02, 0.08, 0.12]:
        for _ in range(3):
            starts.append(_random_config(rng, margin=margin))
    
    # Run Nelder-Mead from each start
    for s_idx, s_pts in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
        
        x0 = s_pts.flatten()
        
        # Try with a few different penalty strengths
        for mu in [1e3, 1e4, 1e5]:
            if time.time() > deadline - 3.0:
                break
            
            try:
                res = _nelder_mead_start(x0, max_iter=1500, mu=mu)
                x_opt = res.x
                pts_opt = x_opt.reshape(N, 2)
                pts_clipped = np.array([_clip_to_triangle(p) for p in pts_opt])
                
                score = _min_area(pts_clipped)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts_clipped.copy()
            except Exception:
                pass
    
    # Final polish: high-precision Nelder-Mead on the best solution
    if best_pts is not None and time.time() < deadline - 5.0:
        x_best = best_pts.flatten()
        for mu in [1e5, 1e6]:
            if time.time() > deadline - 2.0:
                break
            try:
                res = _nelder_mead_start(x_best, max_iter=3000, mu=mu)
                x_opt = res.x
                pts_opt = x_opt.reshape(N, 2)
                pts_clipped = np.array([_clip_to_triangle(p) for p in pts_opt])
                
                score = _min_area(pts_clipped)
                if score > best_score:
                    best_score = score
                    best_pts = pts_clipped.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: return a simple grid
        best_pts = _grid_config()
    
    # Ensure all points are strictly inside
    final_pts = np.array([_clip_to_triangle(p) for p in best_pts])
    
    return final_pts
