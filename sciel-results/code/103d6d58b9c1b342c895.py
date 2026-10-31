import numpy as np
from scipy.optimize import minimize

def get_areas(points):
    # points: (N, 2)
    # Calculate all triangle areas
    n = len(points)
    # Use vectorization for speed
    # i, j, k indices
    # We can just loop over i, j, k for N=11, it's small.
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    return np.min(areas)

def objective(x):
    # x is flattened points
    pts = x.reshape(-1, 2)
    # Check if points are inside triangle
    # If not, penalize heavily
    # Triangle: (0,0), (1,0), (0.5, sqrt(3)/2)
    # Conditions:
    # y >= 0
    # y <= sqrt(3)/2 * x  (for x < 0.5) -> y <= sqrt(3)*x
    # y <= -sqrt(3)/2 * x + sqrt(3)/2 (for x > 0.5) -> y <= -sqrt(3)*x + sqrt(3)/2
    # Actually, simpler:
    # h = sqrt(3)/2
    # y >= 0
    # y <= h * x (left side)
    # y <= h * (1-x) (right side)
    
    h = np.sqrt(3)/2
    penalty = 0
    for p in pts:
        x, y = p
        if y < 0: penalty += 1e6 * abs(y)
        if y > h * x: penalty += 1e6 * (y - h * x)
        if y > h * (1 - x): penalty += 1e6 * (y - h * (1 - x))
        # Also x between 0 and 1
        if x < 0: penalty += 1e6 * abs(x)
        if x > 1: penalty += 1e6 * abs(x-1)
        
    if penalty > 0:
        return penalty
    
    min_area = get_areas(pts)
    return -min_area

def construct():
    N = 11
    h = np.sqrt(3)/2
    
    # Initial guess 1: Grid + 1
    # Generate 10 grid points
    grid_pts = []
    for i in range(4):
        for j in range(4-i):
            # i is row from top? No, let's do from bottom
            # Row r (0 to 3)
            # y = r * h / 3
            # x = (j + 0.5*r) / 3 ?
            # Let's check vertices.
            # r=0: y=0. x=0, 1/3, 2/3, 1. (4 pts)
            # r=1: y=h/3. x=1/6, 1/2, 5/6. (3 pts)
            # r=2: y=2h/3. x=1/3, 2/3. (2 pts)
            # r=3: y=h. x=1/2. (1 pt)
            pass
            
    # Let's construct explicitly
    pts = np.array([
        [0.0, 0.0], [1/3, 0.0], [2/3, 0.0], [1.0, 0.0],
        [1/6, h/3], [1/2, h/3], [5/6, h/3],
        [1/3, 2*h/3], [2/3, 2*h/3],
        [1/2, h]
    ])
    # Add 11th point at centroid of top triangle
    # Top triangle vertices: [1/3, 2*h/3], [2/3, 2*h/3], [1/2, h]
    # Centroid:
    # cx = (1/3 + 2/3 + 1/2) / 3 = (1 + 0.5) / 3 = 1.5/3 = 0.5
    # cy = (2*h/3 + 2*h/3 + h) / 3 = (5*h/3) / 3 = 5*h/9
    new_pt = np.array([0.5, 5*h/9])
    pts = np.vstack([pts, new_pt])
    
    # Optimize
    x0 = pts.flatten()
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    best_pts = res.x.reshape(-1, 2)
    best_val = -res.fun
    
    # Try other starts?
    # Maybe random starts with repulsion
    
    # Return best
    return best_pts
