import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

def construct():
    # Vertices of the equilateral triangle
    # A = (0, 0), B = (1, 0), C = (0.5, sqrt(3)/2)
    
    def is_inside(pt):
        x, y = pt
        if y < -1e-9: return False
        if y > np.sqrt(3) * x + 1e-9: return False
        if y > -np.sqrt(3) * (x - 1) + 1e-9: return False
        return True

    def min_area(points):
        min_a = float('inf')
        for i, j, k in itertools.combinations(range(N), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            area = 0.5 * abs(cross)
            if area < min_a:
                min_a = area
        return min_a

    def objective(flat_pts):
        pts = flat_pts.reshape(N, 2)
        x = pts[:, 0]
        y = pts[:, 1]
        
        # Penalties for being outside
        pen1 = np.sum(np.maximum(0, -y)**2)
        pen2 = np.sum(np.maximum(0, y - np.sqrt(3)*x)**2)
        pen3 = np.sum(np.maximum(0, y + np.sqrt(3)*(x-1))**2)
        
        penalty = 10000 * (pen1 + pen2 + pen3)
        
        min_a = min_area(pts)
        
        return -min_a + penalty

    # Generate initial guesses
    initial_guesses = []
    
    # Helper to generate random points inside triangle
    def gen_random_pts(seed, n_points, include_vertices=True):
        np.random.seed(seed)
        pts = []
        if include_vertices:
            pts.append([0.0, 0.0])
            pts.append([1.0, 0.0])
            pts.append([0.5, np.sqrt(3)/2])
        while len(pts) < n_points:
            x = np.random.rand()
            y = np.random.rand() * np.sqrt(3)/2
            if is_inside([x, y]):
                pts.append([x, y])
        return np.array(pts).flatten()

    # Multiple random starts
    for seed in [42, 123, 999, 31415, 27182]:
        initial_guesses.append(gen_random_pts(seed, N))
        
    # A structured start: vertices + centroid + midpoints of edges?
    # Let's try a grid-like start
    np.random.seed(0)
    pts_struct = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [0.5, np.sqrt(3)/6], # Centroid
        [0.25, np.sqrt(3)/12],
        [0.75, np.sqrt(3)/12],
        [0.5, np.sqrt(3)/4],
        [0.125, np.sqrt(3)/24],
        [0.375, np.sqrt(3)/24],
        [0.625, np.sqrt(3)/24],
        [0.875, np.sqrt(3)/24]
    ])
    # Ensure all are inside (they are by construction)
    initial_guesses.append(pts_struct.flatten())

    best_points = None
    best_score = -float('inf')
    
    for guess in initial_guesses:
        try:
            # Powell method is effective for this type of problem
            res = minimize(objective, guess, method='Powell', 
                           options={'maxiter': 1000, 'maxfev': 10000})
            
            if res.fun < 100: # Check if it found a valid solution (low penalty)
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = gen_random_pts(42, N)

    return best_points
