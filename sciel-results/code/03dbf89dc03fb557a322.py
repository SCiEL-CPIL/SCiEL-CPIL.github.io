"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time

N = 11
H = np.sqrt(3) / 2.0

def _is_inside_triangle_fast(p, eps=1e-9):
    """
    Vectorized check if all points in p (N, 2) are inside the reference triangle.
    Triangle: A(0,0), B(1,0), C(0.5, H)
    """
    x = p[:, 0]
    y = p[:, 1]
    
    # Conditions:
    # 1. y >= -eps
    # 2. sqrt(3)*x + y <= sqrt(3) + eps
    # 3. y <= sqrt(3)*x + eps
    
    cond1 = y >= -eps
    cond2 = (np.sqrt(3)*x + y - np.sqrt(3)) <= eps
    cond3 = (y - np.sqrt(3)*x) <= eps
    
    return np.all(cond1 & cond2 & cond3)

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    min_a = 1e9
    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            for k in range(j+1, n):
                xk, yk = points[k]
                a = 0.5 * abs((xj-xi)*(yk-yi) - (xk-xi)*(yj-yi))
                if a < min_a:
                    min_a = a
    return min_a

def _grid_initialization():
    """
    Create a structured initial configuration.
    """
    pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, H],
        [0.5, 0.0], [0.75, H/2], [0.25, H/2],
        [0.5, H/3], [0.25, H/6], [0.75, H/6],
        [0.375, H/3], [0.625, H/3]
    ])
    return pts

def construct():
    """
    Place 11 points on or inside the unit equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    Returns an array of shape (11, 2).
    """
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Initial points
    best_pts = _grid_initialization()
    best_score = _min_triangle_area(best_pts)
    
    # Simulated Annealing
    T = 0.1
    T_min = 1e-5
    cooling = 0.99
    max_iter = 50000
    
    current_pts = best_pts.copy()
    current_score = best_score
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # Propose move: pick random point, move it randomly
        idx = rng.integers(0, N)
        # Step size proportional to T
        step = T * 0.5
        move = rng.normal(0, step, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] += move
        
        # Check validity
        if not _is_inside_triangle_fast(new_pts):
            continue
        
        new_score = _min_triangle_area(new_pts)
        
        # Accept if better, or with probability exp((new-old)/T)
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            delta = new_score - current_score
            # delta is negative
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_pts = new_pts
                current_score = new_score
                
        T *= cooling
        
    return best_pts

# EVOLVE-BLOCK-END
