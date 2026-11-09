import numpy as np

N = 11
H = np.sqrt(3) / 2

def construct():
    # Define the triangle vertices
    # A=(0,0), B=(1,0), C=(0.5, H)
    
    def is_inside_triangle(p):
        # Barycentric coordinates check or simple inequality
        # y >= 0
        if p[1] < -1e-9:
            return False
        # y <= sqrt(3) * x
        if p[1] > np.sqrt(3) * p[0] + 1e-9:
            return False
        # y <= sqrt(3) * (1 - x)
        if p[1] > np.sqrt(3) * (1 - p[0]) + 1e-9:
            return False
        return True

    def project_to_triangle(p):
        # Simple projection: if outside, move towards centroid
        # Centroid is (0.5, H/3)
        cx, cy = 0.5, H / 3
        if not is_inside_triangle(p):
            # Move 90% of the way to centroid
            return np.array([0.1 * p[0] + 0.9 * cx, 0.1 * p[1] + 0.9 * cy])
        return p

    def min_triangle_area(pts):
        # Vectorized calculation of all triangle areas
        # Number of triangles is N*(N-1)*(N-2)/6
        # For N=11, this is 11*10*9/6 = 165 triangles. Very fast.
        areas = []
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    # Area = 0.5 * |cross(b-a, c-a)|
                    v1 = b - a
                    v2 = c - a
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    areas.append(area)
        return np.min(areas)

    # Initialize points
    # Start with a symmetric configuration
    # We can use a grid mapped to the triangle or just random points inside.
    # A good starting point is often a hexagonal packing or similar.
    # Let's start with points distributed on a grid inside the triangle.
    pts = np.zeros((N, 2))
    
    # Simple stratified sampling for initialization
    # We can just use random points inside the triangle to start SA
    # But better: start with a known good structure if possible, or uniform random.
    # Let's use uniform random points inside the triangle for the initial state.
    for i in range(N):
        while True:
            x = np.random.rand()
            y = np.random.rand() * H
            if is_inside_triangle(np.array([x, y])):
                pts[i] = [x, y]
                break

    current_score = min_triangle_area(pts)
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995
    max_iter = 50000
    rng = np.random.default_rng(42)

    for i in range(max_iter):
        # Generate a new candidate by perturbing one random point
        idx = rng.integers(0, N)
        new_pts = pts.copy()
        
        # Perturbation size decreases with temperature
        step_size = T * 0.5 + 0.001
        delta = rng.normal(0, step_size, size=2)
        new_pts[idx] += delta
        
        # Project back to triangle if outside
        if not is_inside_triangle(new_pts[idx]):
            new_pts[idx] = project_to_triangle(new_pts[idx])
        
        new_score = min_triangle_area(new_pts)
        
        # Acceptance probability
        if new_score > current_score:
            # Always accept better solutions
            pts = new_pts
            current_score = new_score
        else:
            # Accept worse solutions with probability exp((new - current) / T)
            # Note: We are maximizing, so delta = new - current is negative.
            delta_score = new_score - current_score
            if delta_score > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta_score / T)
            
            if rng.random() < accept_prob:
                pts = new_pts
                current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            break
            
    # Final local polish: try small moves around the best point
    # Simple hill climbing
    for _ in range(1000):
        improved = False
        for idx in range(N):
            # Try moving in 8 directions
            for dx in [-0.001, 0, 0.001]:
                for dy in [-0.001, 0, 0.001]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pts = pts.copy()
                    new_pts[idx] += np.array([dx, dy])
                    if is_inside_triangle(new_pts[idx]):
                        new_score = min_triangle_area(new_pts)
                        if new_score > current_score:
                            pts = new_pts
                            current_score = new_score
                            improved = True
        if not improved:
            break
            
    return pts
