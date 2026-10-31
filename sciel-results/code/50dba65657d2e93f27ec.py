import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11

tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

def _calculate_min_area(points):
    """Compute minimum triangle area for 11 points."""
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) * 0.5
    return np.min(areas)

def _uv_to_xy(u, v):
    """Convert barycentric-like coords (u,v) to Cartesian in unit equilateral triangle."""
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2.0) * v
    return x, y

def _xy_to_uv(x, y):
    """Convert Cartesian to (u,v) coords."""
    v = y / (np.sqrt(3) / 2.0)
    u = x - 0.5 * v
    return u, v

def _is_valid(u, v):
    """Check if points are inside the triangle."""
    return np.all(u >= -1e-9) and np.all(v >= -1e-9) and np.all(u + v <= 1 + 1e-9)

def _objective_de(uv_params):
    """Objective for differential evolution: minimize -min_area."""
    u = uv_params[0::2]
    v = uv_params[1::2]
    
    # Soft penalty for constraint violation
    violation = np.maximum(0, u + v - 1.0)
    if np.any(violation > 1e-6):
        return 1.0 + np.sum(violation)
    
    x, y = _uv_to_xy(u, v)
    points = np.column_stack((x, y))
    
    min_area = _calculate_min_area(points)
    return -min_area

def _project_to_triangle(points):
    """Project points to be inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)."""
    pts = points.copy()
    h = np.sqrt(3) / 2.0
    for i in range(len(pts)):
        x, y = pts[i]
        # Constraint 1: y >= 0
        if y < 0:
            y = 0
        # Constraint 2: y <= sqrt(3)*x  => x >= y/sqrt(3)
        if x < y / np.sqrt(3):
            x = y / np.sqrt(3)
        # Constraint 3: y <= sqrt(3)*(1-x) => x <= 1 - y/sqrt(3)
        if x > 1 - y / np.sqrt(3):
            x = 1 - y / np.sqrt(3)
        pts[i] = [x, y]
    return pts

def _initial_guess():
    """Generate a symmetric initial guess for 11 points."""
    # Vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2.0])
    
    # Centroid
    c = (v1 + v2 + v3) / 3.0
    
    # Midpoints of edges
    m1 = (v1 + v2) / 2.0
    m2 = (v2 + v3) / 2.0
    m3 = (v3 + v1) / 2.0
    
    # We have 7 points so far. Need 4 more.
    # Let's place them symmetrically around the centroid.
    # Radius r = 1/3 of the altitude? Altitude is sqrt(3)/2 ~ 0.866.
    # Distance from centroid to vertex is 2/3 * h = sqrt(3)/3 ~ 0.577.
    # Distance from centroid to midpoint is 1/3 * h = sqrt(3)/6 ~ 0.288.
    # Let's try points at distance d from centroid, at angles 0, 90, 180, 270?
    # Or just scale the vertices towards the center.
    
    # Symmetric set:
    # 1-3: Vertices
    # 4: Centroid
    # 5-7: Midpoints
    # 8-11: Points on the medians, between centroid and vertices? Or between centroid and midpoints?
    
    # Let's try a hexagonal-like symmetry around the center.
    # The "best" packing for 11 points often involves the center, 6 points on a hexagon, and 4 others?
    # Actually, a common good start is:
    # 3 vertices, 3 midpoints, 1 center, and 4 points forming a smaller triangle or square in the middle?
    
    # Let's use a simple radial distribution from the center.
    # Center is at (0.5, sqrt(3)/6).
    center_x, center_y = 0.5, np.sqrt(3)/6.0
    
    # 4 points at distance 0.15 from center, at angles 0, 90, 180, 270 (rotated to fit triangle)
    # Angles 0, 60, 120, 180, 240, 300 form a hexagon.
    # Let's pick 4 points that are symmetric with respect to the vertical axis.
    # Angles: 0, 120, 240 (triangle) and 60, 180, 300 (inverted triangle)?
    # That would be 6 points.
    # Let's just pick 4 points:
    # One on each median, halfway between centroid and vertex?
    # Median 1 (to v1): c + 0.5*(v1-c)
    # Median 2 (to v2): c + 0.5*(v2-c)
    # Median 3 (to v3): c + 0.5*(v3-c)
    # That's 3 points. We need 1 more. Maybe center of the triangle formed by midpoints? That's the centroid again.
    
    # Let's try:
    # P1-P3: Vertices
    # P4: Centroid
    # P5-P7: Midpoints
    # P8: Midpoint of segment c-m1
    # P9: Midpoint of segment c-m2
    # P10: Midpoint of segment c-m3
    # P11: ?
    
    # Actually, let's just use a grid-like or random symmetric start and let SA fix it.
    # A better start:
    # 3 vertices.
    # 3 midpoints.
    # 1 center.
    # 4 points at distance 0.2 from center, angles 0, 90, 180, 270?
    # Let's try:
    points = []
    points.append(v1)
    points.append(v2)
    points.append(v3)
    points.append(c)
    points.append(m1)
    points.append(m2)
    points.append(m3)
    
    # Add 4 more points symmetrically.
    # Let's place them on the circle of radius 0.2 around center.
    # To maintain symmetry with the triangle (3-fold), we should use angles that respect that.
    # But we only have 4 points.
    # Let's just scatter them slightly randomly but deterministically for reproducibility.
    # Or better: 2 points on the axis of symmetry, 2 off-axis.
    
    # Let's use the points that are the centroids of the 4 sub-triangles if we divide the big triangle into 4 smaller ones?
    # No, that's just the midpoints and vertices.
    
    # Let's try a "spiral" or "grid" approach for the remaining 4.
    # Actually, a known good configuration for 10 points is vertices, midpoints, center, and 6 points on a hexagon.
    # For 11, maybe add one more?
    
    # Let's just initialize the last 4 points at the center with small offsets.
    offsets = [
        [0.05, 0.0],
        [-0.05, 0.0],
        [0.0, 0.05],
        [0.0, -0.05]
    ]
    for ox, oy in offsets:
        points.append(np.array([center_x + ox, center_y + oy]))
        
    return np.array(points)

def construct():
    """Main function: Simulated Annealing optimization."""
    
    # 1. Initialization
    points = _initial_guess()
    points = _project_to_triangle(points)
    
    current_score = _calculate_min_area(points)
    best_score = current_score
    best_points = points.copy()
    
    # 2. Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.995  # Cooling rate
    steps = 5000  # Number of SA steps
    
    rng = np.random.RandomState(42)
    
    for step in range(steps):
        # Generate a perturbed point
        # Use Gaussian noise. Step size can be adaptive or fixed.
        # Let's use a step size that shrinks with temperature.
        step_size = 0.05 * (T / 0.01) ** 0.5  # Scale step with T
        
        new_points = points + rng.normal(0, step_size, size=(N, 2))
        new_points = _project_to_triangle(new_points)
        
        new_score = _calculate_min_area(new_points)
        
        # Metropolis Acceptance
        if new_score > current_score:
            # Always accept if better
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Accept worse with probability exp((new - current) / T)
            # Since we maximize, delta = new - current is negative.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    points = new_points
                    current_score = new_score
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
    # 3. Final Polish: Small local search
    # Use a few iterations of very small noise to settle
    T = 1e-4
    for _ in range(1000):
        step_size = 0.01 * (T / 1e-4) ** 0.5
        new_points = best_points + rng.normal(0, step_size, size=(N, 2))
        new_points = _project_to_triangle(new_points)
        new_score = _calculate_min_area(new_points)
        
        if new_score > best_score:
            best_score = new_score
            best_points = new_points
        else:
            delta = new_score - best_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    best_points = new_points
                    best_score = new_score
        T *= 0.99
        if T < 1e-8:
            break
            
    return best_points
