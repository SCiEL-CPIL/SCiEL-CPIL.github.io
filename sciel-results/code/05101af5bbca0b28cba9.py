import numpy as np
from scipy.optimize import minimize
import time

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2

def _min_triangle_area(pts):
    """Calculate the minimum area of any triangle formed by the points."""
    min_area = float('inf')
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle with vertices pts[i], pts[j], pts[k]
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p):
    """Check if point p is inside the unit equilateral triangle."""
    x, y = p
    if y < -1e-10 or x < -1e-10 or x > 1 + 1e-10:
        return False
    # Line from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3) x
    # Line from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3) (x - 1)
    if y > S3 * x + 1e-10:
        return False
    if y > -S3 * (x - 1) + 1e-10:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120 # Safety margin
    
    # Initial guess: Centroid + vertices + midpoints? 
    # Let's try a structured grid first.
    # Or random restarts.
    
    best_score = 0
    best_pts = None
    
    # 1. Try a few deterministic structured layouts
    # Layout 1: Grid in barycentric coords
    # Layout 2: Random
    
    # Let's use random restarts with local optimization.
    # Objective: maximize min_area. 
    # Minimize negative min_area? No, min_area is not differentiable easily.
    # But we can use a smooth approximation or just use a simple local search.
    # Since N=11 is small, we can use a simple hill climbing or Nelder-Mead.
    
    # Let's try Nelder-Mead on the negative of the min area.
    # However, Nelder-Mead might get stuck in local minima.
    
    # Better approach: 
    # 1. Generate random points inside triangle.
    # 2. Calculate min area.
    # 3. Try to move points to increase min area.
    
    # Let's implement a simple local search.
    
    def random_points_in_triangle():
        pts = []
        for _ in range(N):
            while True:
                x = np.random.rand()
                y = np.random.rand() * S3H
                if _inside_triangle([x, y]):
                    pts.append([x, y])
                    break
        return np.array(pts)
    
    def local_search(pts, max_iter=50, step_size=0.05):
        current_pts = pts.copy()
        current_score = _min_triangle_area(current_pts)
        
        for _ in range(max_iter):
            improved = False
            for i in range(N):
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = current_pts.copy()
                        new_pts[i] = [current_pts[i][0] + dx, current_pts[i][1] + dy]
                        if _inside_triangle(new_pts[i]):
                            new_score = _min_triangle_area(new_pts)
                            if new_score > current_score:
                                current_pts = new_pts
                                current_score = new_score
                                improved = True
            if not improved:
                break
        return current_pts, current_score

    # Run a few random restarts
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        init_pts = random_points_in_triangle()
        # Optional: add a small jitter to avoid symmetry breaking issues
        init_pts += np.random.normal(0, 0.01, init_pts.shape)
        # Project back inside? Simple rejection or clamp.
        # For simplicity, if a point goes out, we might need to handle it.
        # Let's just re-sample if any point is outside.
        if not all(_inside_triangle(p) for p in init_pts):
             # Re-sample the specific points
             for i in range(N):
                 if not _inside_triangle(init_pts[i]):
                     while True:
                         x = np.random.rand()
                         y = np.random.rand() * S3H
                         if _inside_triangle([x, y]):
                             init_pts[i] = [x, y]
                             break
        
        # Local search
        res_pts, res_score = local_search(init_pts, max_iter=20, step_size=0.02)
        
        if res_score > best_score:
            best_score = res_score
            best_pts = res_pts
            
    # If best_pts is None (shouldn't happen), return a default
    if best_pts is None:
        # Return a grid
        best_pts = np.array([
            [0, 0], [1, 0], [0.5, S3H],
            [1/3, S3H/3], [2/3, S3H/3], [0.5, S3H/2],
            [0.25, S3H/4], [0.75, S3H/4], [0.5, 0],
            [1/6, S3H/6], [5/6, S3H/6]
        ])
        
    # Final check: ensure all points are strictly inside or on boundary
    # The evaluator might reject points outside.
    # Our local search ensures inside.
    
    return best_pts
