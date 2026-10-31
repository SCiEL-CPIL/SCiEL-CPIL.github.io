"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(
                    (p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                    (p3[0] - p1[0]) * (p2[1] - p1[1])
                )
                if area < min_area:
                    min_area = area
    return min_area

def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle (0,0), (1,0), (0.5, sqrt(3)/2)."""
    x, y = p
    h = np.sqrt(3) / 2
    if y < -eps or y > h + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    # Check against the three edges
    # Edge 1: from (0,0) to (1,0): y >= 0
    if y < -eps:
        return False
    # Edge 2: from (1,0) to (0.5, h): line equation
    # Slope = (h - 0)/(0.5 - 1) = h/(-0.5) = -2h
    # y - 0 = -2h(x - 1) => y = -2h(x - 1) => 2h(x-1) + y <= 0
    if 2 * h * (x - 1) + y > eps:
        return False
    # Edge 3: from (0.5, h) to (0,0): line equation
    # Slope = (0 - h)/(0 - 0.5) = -h/(-0.5) = 2h
    # y - 0 = 2h(x - 0) => y = 2hx => y - 2hx <= 0
    if y - 2 * h * x > eps:
        return False
    return True

def _clamp_to_triangle(p):
    """Clamp a point to be inside the triangle."""
    x, y = p
    h = np.sqrt(3) / 2
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, h)
    # Check edge constraints
    # 2h(x-1) + y <= 0 => y <= 2h(1-x)
    if 2 * h * (x - 1) + y > 0:
        y = max(0, 2 * h * (1 - x))
    # y - 2hx <= 0 => y <= 2hx
    if y - 2 * h * x > 0:
        y = min(y, 2 * h * x)
    return np.array([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    h = np.sqrt(3) / 2
    triangle_area = 0.5 * 1 * h  # area of unit base equilateral triangle
    
    # Start with a reasonable initial configuration
    # Use a grid-like or symmetric initial guess
    def initial_points():
        # Place points in a structured way
        pts = []
        # Corners and center
        pts.append([0.0, 0.0])
        pts.append([1.0, 0.0])
        pts.append([0.5, h])
        pts.append([0.5, h/3])
        pts.append([0.25, h/6])
        pts.append([0.75, h/6])
        pts.append([0.33, h/3])
        pts.append([0.67, h/3])
        pts.append([0.5, 2*h/3])
        pts.append([0.25, h/3 + h/12])
        pts.append([0.75, h/3 + h/12])
        return np.array(pts)
    
    points = initial_points()
    # Ensure all points are inside
    for i in range(len(points)):
        points[i] = _clamp_to_triangle(points[i])
    
    best_points = points.copy()
    best_area = _min_triangle_area(best_points)
    
    # Try to improve using local optimization
    # We'll use a simple hill-climbing / Nelder-Mead approach on the 22 coordinates
    
    def objective(x):
        pts = x.reshape(N, 2)
        # Check if all points are inside
        for i in range(N):
            if not _is_inside_triangle(pts[i]):
                return -1.0  # penalty
        return -_min_triangle_area(pts)  # minimize negative area
    
    # Use differential evolution for global search
    # Bounds for each coordinate
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, h))  # y
    
    # First try a quick local optimization from the initial point
    x0 = points.flatten()
    
    # Try Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_area:  # better (less negative means larger area)
            new_pts = res.x.reshape(N, 2)
            all_inside = all(_is_inside_triangle(new_pts[i]) for i in range(N))
            if all_inside:
                new_area = _min_triangle_area(new_pts)
                if new_area > best_area:
                    best_area = new_area
                    best_points = new_pts.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_points
    
    # Try differential evolution for a more global search
    try:
        # Use a smaller population for speed
        de_result = differential_evolution(
            objective, bounds, 
            maxiter=50, popsize=10, tol=1e-8, 
            seed=42, polish=False,
            init='latinhypercube'
        )
        if de_result.fun < -best_area:
            new_pts = de_result.x.reshape(N, 2)
            all_inside = all(_is_inside_triangle(new_pts[i]) for i in range(N))
            if all_inside:
                new_area = _min_triangle_area(new_pts)
                if new_area > best_area:
                    best_area = new_area
                    best_points = new_pts.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return best_points
    
    # Try multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # Generate random points inside triangle
        pts = []
        for _ in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    p = [u * 0.5 + v * 0.5, v * h]
                    # Actually let's use proper triangle sampling
                    # For triangle (0,0), (1,0), (0.5,h):
                    # Use barycentric coordinates
                    break
            
            # Proper sampling using barycentric coordinates
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1 = 1 - r1
                r2 = 1 - r2
            # P = r1*A + r2*B + (1-r1-r2)*C where A=(0,0), B=(1,0), C=(0.5,h)
            px = r1 * 0 + r2 * 1 + (1 - r1 - r2) * 0.5
            py = r1 * 0 + r2 * 0 + (1 - r1 - r2) * h
            pts.append([px, py])
        
        pts = np.array(pts)
        for i in range(N):
            pts[i] = _clamp_to_triangle(pts[i])
        
        x0 = pts.flatten()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < -best_area:
                new_pts = res.x.reshape(N, 2)
                all_inside = all(_is_inside_triangle(new_pts[i]) for i in range(N))
                if all_inside:
                    new_area = _min_triangle_area(new_pts)
                    if new_area > best_area:
                        best_area = new_area
                        best_points = new_pts.copy()
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
