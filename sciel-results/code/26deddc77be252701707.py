import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    def get_L(points):
        # points: (N, 2) array
        # Calculate the minimal L for flat-top hexagon
        # Constraints:
        # |x| <= L
        # |y| <= L * sqrt(3)/2
        # |sqrt(3)/2 * x + 1/2 * y| <= L * sqrt(3)/2
        
        x = points[:, 0]
        y = points[:, 1]
        
        L1 = np.max(np.abs(x))
        L2 = (2.0 / np.sqrt(3)) * np.max(np.abs(y))
        
        # term = sqrt(3)/2 * x + 1/2 * y
        term = (np.sqrt(3)/2) * x + 0.5 * y
        L3 = (2.0 / np.sqrt(3)) * np.max(np.abs(term))
        
        return np.max([L1, L2, L3])

    def objective(params):
        # params: 2*N flat array
        points = params.reshape(N, 2)
        
        # Calculate L
        L = get_L(points)
        
        # Calculate distance penalties
        # We need min distance 1 (in scaled units)
        min_dist = 1.0
        
        # Efficient distance calculation
        # N is small (11)
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                dists[i, j] = d
                dists[j, i] = d
        
        # Penalty for violations
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < min_dist:
                    penalty += (min_dist - d)**2
        
        return L + 100.0 * penalty

    # Initial guess
    # Let's try a specific lattice configuration first to get a good starting point
    # Triangular lattice basis
    u = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(3)/2])
    
    # Generate 11 points from a lattice
    # Center (0,0)
    # 6 neighbors
    # 4 more from the next ring
    # Next ring points in lattice coords (i,j) where i+j=0, |i|=2, |j|=2 etc?
    # Let's just use a simple rectangular block of lattice points
    # 3x4 block = 12 points. Remove one corner.
    
    # Lattice points (i, j) -> x = i + 0.5*j, y = sqrt(3)/2 * j
    # Let's pick i in [-1, 1], j in [-1, 1] -> 9 points
    # Add (2, 0), (-2, 0) -> 11 points
    # This forms a hexagon of radius 2 (in lattice steps)
    
    lattice_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = i + 0.5 * j
            y = np.sqrt(3)/2 * j
            lattice_points.append([x, y])
            
    # We have 9 points. Add 2 more.
    # (2, 0) -> x=2, y=0
    # (-2, 0) -> x=-2, y=0
    lattice_points.append([2.0, 0.0])
    lattice_points.append([-2.0, 0.0])
    
    init_params = np.array(lattice_points).flatten()
    
    # Optimize
    best_L = float('inf')
    best_params = None
    
    # Run a few restarts with small perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(10):
        if restart == 0:
            current_init = init_params
        else:
            current_init = init_params + rng.normal(0, 0.1, 2*N)
            
        res = minimize(objective, current_init, method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
        
        if res.fun < best_L:
            best_L = res.fun
            best_params = res.x
    
    # Extract points
    points = best_params.reshape(N, 2)
    
    # Final L calculation
    final_L = get_L(points)
    
    # The outer hexagon side length S
    S = final_L * np.sqrt(3) + 1.0
    
    # The inner hexagons are at these positions.
    # We assumed they are aligned with the outer hexagon (angle 0).
    # The outer hexagon is centered at (0,0) and aligned (angle 0).
    
    inner = []
    for p in points:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    center = (0.0, 0.0)
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle
