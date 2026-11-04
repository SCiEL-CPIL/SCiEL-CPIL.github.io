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
        # 1. y >= 0
        pen1 = np.sum(np.maximum(0, -y)**2)
        # 2. y <= sqrt(3)*x  =>  y - sqrt(3)*x <= 0
        pen2 = np.sum(np.maximum(0, y - np.sqrt(3)*x)**2)
        # 3. y <= -sqrt(3)*(x-1) => y + sqrt(3)*(x-1) <= 0
        pen3 = np.sum(np.maximum(0, y + np.sqrt(3)*(x-1))**2)
        penalty = 100 * (pen1 + pen2 + pen3)
        
        if penalty > 1e-3:
            return 1000.0 + penalty
            
        # Calculate min area
        min_a = min_area(pts)
        return -min_a + penalty

    # Multi-start initialization
    best_score = -float('inf')
    best_points = None
    
    starts = []
    
    # Start 1: Random
    np.random.seed(42)
    random_pts = []
    while len(random_pts) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            random_pts.append([x, y])
    starts.append(np.array(random_pts).flatten())
    
    # Start 2: Hexagonal-inspired
    # Manually placed points
    hex_pts = [
        [0.5, 0.288],
        [0.25, 0.144],
        [0.75, 0.144],
        [0.5, 0.432],
        [0.1, 0.3],
        [0.9, 0.3],
        [0.35, 0.5],
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, 0.7],
        [0.2, 0.6]
    ]
    valid_hex = []
    for p in hex_pts:
        if is_inside(p):
            valid_hex.append(p)
    while len(valid_hex) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            valid_hex.append([x, y])
    starts.append(np.array(valid_hex[:N]).flatten())

    # Start 3: Grid
    grid_pts = []
    for i in range(4):
        for j in range(4):
            x = 0.1 + 0.2*i
            y = 0.1 + 0.2*j * 0.866 # approx hex grid height
            if is_inside([x, y]):
                grid_pts.append([x, y])
    while len(grid_pts) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            grid_pts.append([x, y])
    starts.append(np.array(grid_pts[:N]).flatten())
    
    # Start 4: Vertices and Midpoints
    edge_pts = [
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, 0.8],
        [0.5, 0.05],
        [0.25, 0.4],
        [0.75, 0.4],
        [0.5, 0.3],
        [0.3, 0.2],
        [0.7, 0.2],
        [0.4, 0.5],
        [0.6, 0.5]
    ]
    valid_edge = []
    for p in edge_pts:
        if is_inside(p):
            valid_edge.append(p)
    while len(valid_edge) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            valid_edge.append([x, y])
    starts.append(np.array(valid_edge[:N]).flatten())
    
    # Start 5: Another random with different seed
    np.random.seed(123)
    random_pts2 = []
    while len(random_pts2) < N:
        x = np.random.rand()
        y = np.random.rand() * np.sqrt(3)/2
        if is_inside([x, y]):
            random_pts2.append([x, y])
    starts.append(np.array(random_pts2).flatten())

    # Run optimization
    for start in starts:
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < 1000:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    if best_points is None:
        return np.array(random_pts)
        
    return best_points
