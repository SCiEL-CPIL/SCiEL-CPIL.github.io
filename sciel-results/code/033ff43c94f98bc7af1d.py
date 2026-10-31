"""Heilbronn triangles: maximize smallest triangle area among 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute triangle indices for vectorized area computation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]

_H = np.sqrt(3) / 2

def _min_area(pts):
    """Vectorized: compute minimum triangle area among all C(11,3)=165 triangles."""
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    areas = 0.5 * np.abs((pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1])
                       - (pk[:, 0] - pi[:, 0]) * (pj[:, 1] - pi[:, 1]))
    return areas.min()

def _clamp_to_triangle(pts):
    """Clamp points to be inside the equilateral triangle (0,0),(1,0),(0.5,sqrt(3)/2)."""
    # Constraints: y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    # i.e., x >= 0, x <= 1, y >= 0, y <= sqrt(3)*x, y <= sqrt(3)*(1-x)
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    
    # y >= 0
    y = np.maximum(y, 0)
    # y <= sqrt(3) * x  =>  x >= y / sqrt(3)
    x = np.maximum(x, y / np.sqrt(3))
    # y <= sqrt(3) * (1-x)  =>  x <= 1 - y / sqrt(3)
    x = np.minimum(x, 1 - y / np.sqrt(3))
    # x >= 0, x <= 1
    x = np.clip(x, 0, 1)
    
    # Re-clamp y after x changes
    y = np.minimum(y, np.sqrt(3) * x)
    y = np.minimum(y, np.sqrt(3) * (1 - x))
    y = np.maximum(y, 0)
    
    return np.column_stack([x, y])

def _random_points(rng):
    """Generate random points inside the triangle."""
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                break
        x = u + 0.5 * v
        y = _H * v
        pts[i] = [x, y]
    return pts

def _sa_optimize(start_pts, rng, time_limit):
    """Simulated Annealing optimization."""
    pts = start_pts.copy()
    current_area = _min_area(pts)
    best_pts = pts.copy()
    best_area = current_area
    
    T0 = 0.001  # initial temperature
    T1 = 1e-7   # final temperature
    n_steps = 200000
    alpha = (T1 / T0) ** (1.0 / n_steps)
    T = T0
    
    deadline = time.time() + time_limit
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Random perturbation
        idx = rng.randint(0, N - 1)
        dx = (rng.random() - 0.5) * 0.05
        dy = (rng.random() - 0.5) * 0.05
        new_pts = pts.copy()
        new_pts[idx, 0] += dx
        new_pts[idx, 1] += dy
        new_pts = _clamp_to_triangle(new_pts)
        
        new_area = _min_area(new_pts)
        delta = new_area - current_area
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            pts = new_pts
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_pts = pts.copy()
        
        T *= alpha
    
    return best_pts, best_area

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0
    
    # Multi-start with SA
    n_starts = 0
    while time.time() < deadline:
        start_pts = _random_points(rng)
        # Give each SA run about 5 seconds
        remaining = deadline - time.time()
        if remaining < 2:
            break
        sa_time = min(5.0, remaining - 1)
        pts, area = _sa_optimize(start_pts, rng, sa_time)
        n_starts += 1
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: random points
        best_pts = _random_points(rng)
    
    return best_pts
# EVOLVE-BLOCK-END
