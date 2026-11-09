import numpy as np
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

def construct():
    # Define the triangle vertices
    # A=(0,0), B=(1,0), C=(0.5, H)
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    def is_inside_triangle(p):
        if p[1] < -1e-9:
            return False
        if p[1] > np.sqrt(3) * p[0] + 1e-9:
            return False
        if p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            return False
        return True

    def project_to_triangle(p):
        # Simple projection: if outside, move towards centroid
        cx, cy = 0.5, H / 3
        if not is_inside_triangle(p):
            # Move 90% of the way to centroid
            return np.array([0.1 * p[0] + 0.9 * cx, 0.1 * p[1] + 0.9 * cy])
        return p

    def min_triangle_area(pts):
        # Vectorized calculation of all triangle areas
        areas = np.zeros((N, N, N))
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    v1 = b - a
                    v2 = c - a
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas[i, j, k] = area
                    areas[j, i, k] = area
                    areas[i, k, j] = area
                    areas[j, k, i] = area
                    areas[k, i, j] = area
                    areas[k, j, i] = area
        return np.min(areas)

    def compute_gradient(pts):
        # Calculate the subgradient of the minimum area function
        # The minimum area is achieved by one or more triangles.
        # We find all triangles with area close to the minimum and sum their gradients.
        
        min_area = min_triangle_area(pts)
        
        # Threshold to include triangles that are "active" (close to min)
        threshold = min_area * 1.001 + 1e-8
        
        grad = np.zeros((N, 2))
        
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    v1 = b - a
                    v2 = c - a
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    
                    if area <= threshold:
                        # Gradient of area with respect to points
                        # Area = 0.5 * |(b-a) x (c-a)|
                        # Let cross = v1[0]*v2[1] - v1[1]*v2[0]
                        # If cross > 0, grad_a = 0.5 * (-(c-b), -(c-b)_y) ? No.
                        # Let's use the formula:
                        # dA/da = 0.5 * (b - c) rotated 90 deg?
                        # Actually, it's easier to use:
                        # grad_x_i = 0.5 * (y_j - y_k, -(x_j - x_k)) if orientation is CCW
                        # Let's determine orientation.
                        cross = v1[0]*v2[1] - v1[1]*v2[0]
                        if cross >= 0:
                            # CCW
                            ga = 0.5 * np.array([-(c[1] - b[1]), c[0] - b[0]])
                            gb = 0.5 * np.array([c[1] - a[1], -(c[0] - a[0])])
                            gc = 0.5 * np.array([a[1] - b[1], -(a[0] - b[0])])
                        else:
                            # CW
                            ga = 0.5 * np.array([c[1] - b[1], -(c[0] - b[0])])
                            gb = 0.5 * np.array([-(c[1] - a[1]), c[0] - a[0]])
                            gc = 0.5 * np.array([-(a[1] - b[1]), a[0] - b[0]])
                        
                        grad[i] += ga
                        grad[j] += gb
                        grad[k] += gc
                        
        # Normalize gradient to unit length to control step size
        norm = np.linalg.norm(grad)
        if norm > 1e-9:
            grad /= norm
            
        return grad

    def gradient_ascent(pts, max_iter=500, step_size=0.01):
        current_score = min_triangle_area(pts)
        
        for i in range(max_iter):
            grad = compute_gradient(pts)
            
            # Adaptive step size
            new_pts = pts.copy()
            new_pts += step_size * grad
            
            # Project points back to triangle
            for idx in range(N):
                if not is_inside_triangle(new_pts[idx]):
                    new_pts[idx] = project_to_triangle(new_pts[idx])
            
            new_score = min_triangle_area(new_pts)
            
            if new_score > current_score:
                pts = new_pts
                current_score = new_score
                # Optionally increase step size slightly if we made good progress
                # step_size = min(step_size * 1.1, 0.05)
            else:
                # Decrease step size if we didn't improve
                step_size *= 0.9
                if step_size < 1e-6:
                    break
                    
        return pts, current_score

    def local_search(pts, max_iter=1000, step_size=0.001):
        current_score = min_triangle_area(pts)
        
        for i in range(max_iter):
            improved = False
            # Try moving each point in small steps
            for idx in range(N):
                # Try 8 directions + center
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pts = pts.copy()
                        new_pts[idx] += np.array([dx, dy])
                        
                        if is_inside_triangle(new_pts[idx]):
                            new_score = min_triangle_area(new_pts)
                            if new_score > current_score + 1e-10:
                                pts = new_pts
                                current_score = new_score
                                improved = True
            if not improved:
                break
                
        return pts, current_score

    # Multi-start strategy with structured initializations
    best_pts = None
    best_score = -1.0
    
    # Initialization 1: Grid-like distribution
    # We can map a square grid to the triangle using barycentric coordinates
    # Or simply use a hexagonal packing which is often optimal for packing problems.
    
    # Let's create a few initial configurations
    inits = []
    
    # Init 1: Uniform random (but seeded for reproducibility in multi-start context? 
    # Actually, deterministic is better for "new approach" claim, but multi-start with 
    # different deterministic grids is better.)
    
    # Create a grid of points in a unit square, then map to triangle
    # We need 11 points.
    # A 4x4 grid has 16 points. We can pick 11 of them.
    # Or a hexagonal grid.
    
    # Let's use a hexagonal grid centered in the triangle.
    # Spacing s.
    # We want to cover the triangle.
    
    # Simpler: Use barycentric coordinates.
    # Generate a grid in (u,v) space where u+v+w=1, u,v,w >= 0.
    # For N=11, we can use a set of predefined barycentric coordinates.
    
    # Let's try a few specific configurations:
    
    # Config 1: Vertices + midpoints + center + some extra
    # Vertices: (0,0), (1,0), (0.5, H)
    # Midpoints: (0.5, 0), (0.75, H/2), (0.25, H/2)
    # Center: (0.5, H/3)
    # This is 7 points. Need 4 more.
    
    # Let's just use a randomized multi-start with a good local optimizer.
    # The "new approach" is the Gradient Ascent + Local Search hybrid.
    # We will run it from several random starts to ensure robustness.
    
    rng = np.random.default_rng(123)
    
    for start_idx in range(5):
        # Generate initial points
        if start_idx == 0:
            # Symmetric initialization
            pts = np.zeros((N, 2))
            # Place vertices
            pts[0] = A
            pts[1] = B
            pts[2] = C
            # Place midpoints
            pts[3] = (A + B) / 2
            pts[4] = (B + C) / 2
            pts[5] = (C + A) / 2
            # Place centroid
            pts[6] = (A + B + C) / 3
            # Place 4 more points randomly inside
            for i in range(7, N):
                while True:
                    x = rng.random()
                    y = rng.random() * H
                    if is_inside_triangle(np.array([x, y])):
                        pts[i] = [x, y]
                        break
        else:
            # Random initialization
            pts = np.zeros((N, 2))
            for i in range(N):
                while True:
                    x = rng.random()
                    y = rng.random() * H
                    if is_inside_triangle(np.array([x, y])):
                        pts[i] = [x, y]
                        break
            
        # Run Gradient Ascent
        pts_ga, score_ga = gradient_ascent(pts, max_iter=200, step_size=0.05)
        
        # Run Local Search to polish
        pts_ls, score_ls = local_search(pts_ga, max_iter=500, step_size=0.005)
        
        if score_ls > best_score:
            best_score = score_ls
            best_pts = pts_ls
            
    return best_pts
