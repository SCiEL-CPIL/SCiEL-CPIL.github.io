"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

# Define the equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a, b, c):
    """Check if point p is inside triangle abc using barycentric coordinates."""
    # Convert to barycentric coordinates
    v0 = c - a
    v1 = b - a
    v2 = p - a
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    # Compute barycentric coordinates
    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    # Check if point is in the triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def compute_min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross product|
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """Objective function to minimize (negative of min triangle area)."""
    points = params.reshape(-1, 2)
    
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p, A, B, C):
            return 1.0  # Large penalty for invalid points
    
    min_area = compute_min_triangle_area(points)
    return -min_area

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize minimum triangle area."""
    
    # Use a grid-based initial guess with some symmetry
    # The optimal configuration is likely symmetric
    # Start with a regular hexagon-like arrangement plus center and corners
    
    # Initial guess: use a known good configuration or generate symmetric points
    # Let's use a combination of center, midpoints, and vertices (slightly inset)
    
    # Symmetric initial configuration
    center = np.array([0.5, np.sqrt(3) / 6])
    
    # Create initial points with some structure
    # Use a hexagonal lattice pattern
    points = []
    
    # Add center
    points.append(center)
    
    # Add 6 points in a hexagonal pattern around center
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        r = 0.3
        p = center + r * np.array([np.cos(angle), np.sin(angle)])
        if point_in_triangle(p, A, B, C):
            points.append(p)
    
    # Add remaining points near vertices (inset)
    for vertex in [A, B, C]:
        # Move slightly inward
        p = vertex + 0.1 * (center - vertex)
        if point_in_triangle(p, A, B, C) and len(points) < N:
            points.append(p)
    
    # Fill remaining points
    while len(points) < N:
        # Add random points inside triangle
        u = np.random.random()
        v = np.random.random()
        if u + v < 1:
            p = A + u * (B - A) + v * (C - A)
            points.append(p)
    
    points = np.array(points[:N])
    
    # Define bounds for each coordinate
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, np.sqrt(3) / 2))  # y
    
    # Use differential evolution for global optimization
    # This is a good method for non-convex optimization problems
    result = differential_evolution(
        objective, 
        bounds, 
        seed=42,
        maxiter=1000,
        tol=1e-8,
        polish=True,
        workers=-1
    )
    
    points = result.x.reshape(-1, 2)
    
    # Ensure all points are strictly inside the triangle
    for i in range(N):
        p = points[i]
        if not point_in_triangle(p, A, B, C):
            # Project back to triangle
            # Find the closest point in the triangle
            # Simple approach: move toward center
            center = np.array([0.5, np.sqrt(3) / 6])
            points[i] = 0.9 * p + 0.1 * center
            # Repeat until inside
            for _ in range(10):
                if point_in_triangle(points[i], A, B, C):
                    break
                points[i] = 0.9 * points[i] + 0.1 * center
    
    return points
# EVOLVE-BLOCK-END
