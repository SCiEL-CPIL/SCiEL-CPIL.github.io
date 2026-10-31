"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

def _tri_area(p1, p2, p3):
    """Area of triangle with vertices p1, p2, p3 (each shape (2,))."""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))

def _min_triangle_area(points):
    """Compute the minimum area over all C(11,3)=165 triangles."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _tri_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _min_triangle_area_vec(points):
    """Vectorized computation of min triangle area."""
    n = len(points)
    # Precompute all pairwise differences
    # For each triple (i,j,k), area = 0.5 * |cross(pj-pi, pk-pi)|
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d1 = points[j] - points[i]
            for k in range(j + 1, n):
                d2 = points[k] - points[i]
                a = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
                if a < min_area:
                    min_area = a
    return min_area

def _objective(x):
    """Negative of min triangle area for optimization (we minimize negative)."""
    points = x.reshape(-1, 2)
    return -_min_triangle_area_vec(points)

def _inside_triangle(points, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    # Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    # For point (x,y) to be inside:
    #   y >= 0
    #   x >= 0
    #   y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    #   y <= sqrt(3) * (1 - x)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    x = points[:, 0]
    y = points[:, 1]
    h = np.sqrt(3) / 2
    return np.all(y >= -eps) and np.all(x >= -eps) and np.all(y <= h + eps) \
           and np.all(y <= np.sqrt(3) * x + eps) and np.all(y <= np.sqrt(3) * (1 - x) + eps)

def _project_inside(points):
    """Project points back into the triangle if they went outside."""
    # Simple approach: clip to bounding box and then ensure triangle constraints
    # This is a rough projection; for better results we'd need proper projection
    for i in range(len(points)):
        x, y = points[i]
        # Clip to bounding box
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, np.sqrt(3) / 2)
        # Enforce triangle constraints
        # y <= sqrt(3)*x
        if y > np.sqrt(3) * x:
            # Move to the line y = sqrt(3)*x
            y = np.sqrt(3) * x
        # y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x):
            y = np.sqrt(3) * (1 - x)
        points[i] = [x, y]
    return points

def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    best_points = None
    best_score = -np.inf
    
    # Initial guess: use a structured grid or known good configuration
    # Start with a rough grid inside the triangle
    h = np.sqrt(3) / 2
    
    # Generate initial points on a grid
    def make_grid(n=4):
        pts = []
        for i in range(n):
            for j in range(n - i):
                if len(pts) < N:
                    # Barycentric-like placement
                    u = (i + 0.5) / n
                    v = (j + 0.5) / (n - i)
                    x = u * (1 - v) + 0.5 * v
                    y = h * v
                    pts.append([x, y])
        # Pad or trim to N points
        while len(pts) < N:
            pts.append([0.5, h / 3])
        return np.array(pts[:N])
    
    # Try multiple restarts with different initial configurations
    rng = np.random.default_rng(42)
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        
        if restarts == 1:
            # First restart: structured grid
            x0 = make_grid(4).flatten()
        elif restarts == 2:
            # Second: near-uniform random
            x0 = _make_random_init(rng).flatten()
        elif restarts == 3:
            # Third: perturbed grid
            x0 = (make_grid(4) + 0.05 * rng.standard_normal((N, 2))).flatten()
        elif restarts == 4:
            # Fourth: different grid density
            x0 = make_grid(5).flatten()
        else:
            # Subsequent: random perturbations of best
            if best_points is not None:
                x0 = (best_points + 0.1 * rng.standard_normal((N, 2))).flatten()
            else:
                x0 = _make_random_init(rng).flatten()
        
        # Ensure initial point is inside triangle
        pts0 = x0.reshape(-1, 2)
        if not _inside_triangle(pts0):
            pts0 = _project_inside(pts0)
            x0 = pts0.flatten()
        
        # Run optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxls': 50}
            )
            
            if result.success or result.fun < best_score:
                pts = result.x.reshape(-1, 2)
                # Project back inside if needed
                if not _inside_triangle(pts):
                    pts = _project_inside(pts)
                
                score = _min_triangle_area_vec(pts)
                if score > best_score and _inside_triangle(pts):
                    best_score = score
                    best_points = pts.copy()
                    
        except Exception:
            pass
        
        # Check if we should continue
        if time.time() > deadline - 20:
            break
    
    if best_points is None:
        # Fallback: return the grid
        best_points = make_grid(4)
    
    return best_points

def _make_random_init(rng):
    """Generate random points inside the triangle."""
    h = np.sqrt(3) / 2
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = h * v
                pts.append([x, y])
                break
    return np.array(pts)
# EVOLVE-BLOCK-END
