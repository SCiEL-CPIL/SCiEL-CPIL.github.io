"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
import random

def _triangle_area(p1, p2, p3):
    """Compute the area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area among all triangles formed by triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(points[i], points[j], points[k])
                if a < min_area:
                    min_area = a
    return min_area

def _in_triangle(p, eps=1e-10):
    """Check if point p is inside or on the boundary of the unit equilateral triangle."""
    x, y = p
    # The triangle has vertices (0,0), (1,0), (0.5, sqrt(3)/2)
    # y >= 0
    if y < -eps:
        return False
    # y <= sqrt(3)/2
    if y > np.sqrt(3) / 2 + eps:
        return False
    # y <= sqrt(3) * x  (left edge from (0,0) to (0.5, sqrt(3)/2))
    if y > np.sqrt(3) * x + eps:
        return False
    # y <= sqrt(3) * (1 - x)  (right edge from (1,0) to (0.5, sqrt(3)/2))
    if y > np.sqrt(3) * (1 - x) + eps:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s out of 120s
    
    # Use a deterministic good initial configuration:
    # Start with a grid-like arrangement and refine
    
    best_points = None
    best_score = 0.0
    
    # Initial guess: use a structured pattern
    # For 11 points in a triangle, a good starting point is to use a triangular lattice
    # or a combination of center + vertices + midpoints
    
    # Let's start with a reasonable initial configuration
    # The equilateral triangle has vertices A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    h = np.sqrt(3) / 2
    
    # Initial configuration: mix of vertices, edge midpoints, and interior points
    initial_configs = []
    
    # Config 1: Triangular lattice-like
    # 1 point at center, 3 at vertices, 3 at edge midpoints, 4 more
    center = (0.5, h / 3)  # Centroid
    mid_ab = (0.5, 0)
    mid_bc = (0.75, h / 2)
    mid_ca = (0.25, h / 2)
    
    # Create a 4x4 triangular grid and pick 11 points
    # Or use a simpler approach: start with random points and optimize
    
    # Strategy: Use Nelder-Mead or similar local optimization to maximize min triangle area
    
    def objective(points_flat):
        """Objective function to minimize (negative of min triangle area)."""
        pts = points_flat.reshape(-1, 2)
        # Penalty if any point is outside the triangle
        for p in pts:
            if not _in_triangle(p, eps=1e-6):
                return 1.0  # Large penalty
        return -_min_triangle_area(pts)
    
    def run_local_optimization(initial_points, method='Nelder-Mead'):
        """Run local optimization from a given initial configuration."""
        x0 = np.array(initial_points).flatten()
        result = minimize(objective, x0, method=method, options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.success or result.fun < 0:
            pts = result.x.reshape(-1, 2)
            # Verify all points are inside
            if all(_in_triangle(p) for p in pts):
                return pts
        return None
    
    # Generate multiple initial configurations and try to optimize each
    rng = random.Random(42)
    
    # Initial config 1: Structured
    structured = [
        [0.0, 0.0],           # Vertex A
        [1.0, 0.0],           # Vertex B
        [0.5, h],             # Vertex C
        [0.5, 0.0],           # Mid AB
        [0.75, h/2],          # Mid BC
        [0.25, h/2],          # Mid CA
        [0.5, h/3],           # Centroid
        [0.33, h/6],          # Interior
        [0.67, h/6],          # Interior
        [0.5, h/2],           # Interior
        [0.25, h/3],          # Interior
    ][:N]
    initial_configs.append(structured)
    
    # Initial config 2: Random
    for trial in range(5):
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u + v < 1.0:
                x = u + 0.5 * v
                y = h * v
                pts.append([x, y])
        initial_configs.append(pts)
    
    # Try each initial configuration
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break
        
        # Run local optimization
        optimized = run_local_optimization(init_pts)
        if optimized is not None:
            score = _min_triangle_area(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized
        
        # Also try with a few perturbations
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            perturbed = np.array(init_pts) + np.random.randn(N, 2) * 0.05
            # Clip to stay roughly in triangle
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0, h)
            perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
            # Filter out points outside triangle
            valid_pts = [p for p in perturbed if _in_triangle(p)]
            if len(valid_pts) == N:
                optimized = run_local_optimization(valid_pts)
                if optimized is not None:
                    score = _min_triangle_area(optimized)
                    if score > best_score:
                        best_score = score
                        best_points = optimized
    
    # If no optimization succeeded, return the best initial config
    if best_points is None:
        # Use the structured config as fallback
        best_points = np.array(initial_configs[0])
    
    return np.array(best_points)
# EVOLVE-BLOCK-END
