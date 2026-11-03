import numpy as np
import time

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

# Pre-compute all triangle indices
tri_indices = []
for i in range(N-2):
    for j in range(i+1, N-1):
        for k in range(j+1, N):
            tri_indices.append((i, j, k))
tri_indices = np.array(tri_indices)

def min_area_vectorized(pts):
    """
    Compute minimum triangle area for N points, vectorized.
    pts: (N, 2) array
    """
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def project_points_to_triangle(pts):
    """
    Project each point to the triangle if outside.
    """
    v0 = B - A
    v1 = C - A
    
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    
    denom = d00 * d11 - d01 * d01
    
    v2 = pts - A
    
    d02 = np.sum(v0 * v2, axis=1)
    d12 = np.sum(v1 * v2, axis=1)
    
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    u = 1 - v - w
    
    inside = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)
    
    projected = pts.copy()
    
    for i in range(N):
        if not inside[i]:
            p = pts[i]
            
            def dist_to_segment(p, a, b):
                ab = b - a
                ap = p - a
                t = np.dot(ap, ab) / np.dot(ab, ab)
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj), proj
            
            d1, proj1 = dist_to_segment(p, A, B)
            d2, proj2 = dist_to_segment(p, B, C)
            d3, proj3 = dist_to_segment(p, C, A)
            
            min_d = min(d1, d2, d3)
            if min_d == d1:
                projected[i] = proj1
            elif min_d == d2:
                projected[i] = proj2
            else:
                projected[i] = proj3
    
    return projected

def random_point_in_triangle(rng):
    u = rng.random()
    v = rng.random()
    if u + v > 1:
        u = 1 - u
        v = 1 - v
    return u * A + v * B + (1 - u - v) * C

def construct():
    rng = np.random.default_rng(42)
    
    # Simulated Annealing parameters
    start_temp = 0.01
    end_temp = 1e-6
    n_iterations = 200000
    start_step = 0.1
    end_step = 0.001
    
    # Initialize with random points
    pts = np.zeros((N, 2))
    for i in range(N):
        pts[i] = random_point_in_triangle(rng)
    
    current_area = min_area_vectorized(pts)
    best_pts = pts.copy()
    best_area = current_area
    
    # Simulated Annealing Loop
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    for i in range(n_iterations):
        if time.time() - start_time > max_time:
            break
            
        # Interpolate temperature and step size
        alpha = i / n_iterations
        temp = start_temp * (end_temp / start_temp) ** alpha
        step = start_step * (end_step / start_step) ** alpha
        
        # Pick a random point to move
        idx = rng.integers(N)
        
        # Generate a random displacement
        displacement = rng.normal(0, step, size=2)
        
        # New point
        new_pts = pts.copy()
        new_pts[idx] += displacement
        
        # Project to triangle
        new_pts = project_points_to_triangle(new_pts)
        
        # Calculate new area
        new_area = min_area_vectorized(new_pts)
        
        # Acceptance criterion
        delta_area = new_area - current_area
        
        if delta_area > 0:
            # Accept improvement
            pts = new_pts
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_pts = pts.copy()
        else:
            # Accept worse solution with probability exp(delta_area / temp)
            # Note: delta_area is negative, temp is positive
            prob = np.exp(delta_area / temp)
            if rng.random() < prob:
                pts = new_pts
                current_area = new_area
    
    return best_pts
