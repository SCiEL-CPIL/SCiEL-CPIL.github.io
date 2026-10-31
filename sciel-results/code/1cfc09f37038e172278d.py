import numpy as np
import time
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

def construct():
    deadline = time.time() + 140  # Safety margin
    
    # Precompute triangle indices for area calculation
    # We need to calculate area for all C(11,3) = 165 triangles
    # Area of triangle with points p1, p2, p3 is 0.5 * |det(p2-p1, p3-p1)|
    
    def get_areas(points):
        """
        points: shape (11, 2)
        returns array of 165 areas
        """
        # Unroll for speed? Or use vectorized operations.
        # Vectorized:
        # p1, p2, p3 combinations
        # This might be slow if done naively in a loop.
        # Let's use a helper that takes a flat array of 22 coords.
        
        # Actually, inside the constraint function, we will have a flat array `x` of length 22.
        # We can reshape it to (11, 2).
        
        pts = x[:22].reshape(N, 2)
        
        # Calculate all pairwise differences?
        # Area formula: 0.5 * | x1(y2-y3) + x2(y3-y1) + x3(y1-y2) |
        
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    # Cross product z-component
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                    areas.append(0.5 * abs(cross))
        return np.array(areas)

    # This loop is 165 iterations. It's fast enough.
    
    def objective(x):
        # x[0] is A, x[1:22] are coordinates
        A = x[0]
        return -A # Minimize -A to maximize A

    def constraints(x):
        A = x[0]
        pts = x[1:22].reshape(N, 2)
        
        # Calculate areas
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                    areas.append(0.5 * abs(cross))
        areas = np.array(areas)
        
        # Constraints: areas - A >= 0
        # scipy expects 'fun' to return 0 for equality, >= 0 for inequality?
        # Actually, scipy.optimize.minimize constraints format:
        # {'type': 'ineq', 'fun': lambda x: ...} where fun(x) >= 0
        # So we return areas - A
        
        cons_area = areas - A
        
        # Linear constraints for triangle bounds
        # 1. y >= 0  =>  y - 0 >= 0
        # 2. y <= sqrt(3) x  =>  y - sqrt(3) x <= 0  =>  sqrt(3) x - y >= 0
        # 3. y <= sqrt(3) (1-x) => y + sqrt(3) x - sqrt(3) <= 0 => sqrt(3) - y - sqrt(3) x >= 0
        
        xs = pts[:, 0]
        ys = pts[:, 1]
        
        cons_y_pos = ys
        cons_y_slope1 = np.sqrt(3) * xs - ys
        cons_y_slope2 = np.sqrt(3) - ys - np.sqrt(3) * xs
        
        # Combine all inequality constraints into a single array
        # The function must return an array of values >= 0
        return np.concatenate([cons_area, cons_y_pos, cons_y_slope1, cons_y_slope2])

    # Bounds for variables
    # A is unbounded above (but practically limited), let's say [0, 1]
    # x in [0, 1], y in [0, H]
    bounds = [(0, 1)]  # A
    for _ in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, H))  # y

    # Initial guesses
    # We will try several initial configurations.
    # 1. Grid
    # 2. Random
    # 3. Hexagonal-like
    
    best_A = 0
    best_x = None
    
    def run_optimization(x0):
        if time.time() > deadline - 10:
            return None, 0
            
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=[{'type': 'ineq', 'fun': constraints}],
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        if res.success:
            A = res.x[0]
            # Verify solution
            pts = res.x[1:22].reshape(N, 2)
            # Check if inside triangle
            # Check min area
            min_area = np.min([0.5 * abs((pts[j][0]-pts[i][0])*(pts[k][1]-pts[i][1]) - (pts[k][0]-pts[i][0])*(pts[j][1]-pts[i][1])) 
                               for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
            # If the reported A is close to min_area and valid, accept
            if min_area > best_A:
                return res.x, min_area
        return None, 0

    # Generate initial guesses
    inits = []
    
    # 1. Uniform Grid
    xs = np.linspace(0.1, 0.9, 4)
    ys = np.linspace(0.1, H*0.8, 3)
    pts = []
    for x in xs:
        for y in ys:
            if y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x) and y >= 0:
                pts.append([x, y])
    # Pad or trim to 11
    while len(pts) < N:
        pts.append([0.5, H/3])
    pts = np.array(pts[:N])
    A_init = 0.01
    x0 = np.concatenate([[A_init], pts.flatten()])
    inits.append(x0)
    
    # 2. Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        pts = []
        while len(pts) < N:
            u, v = rng.random(2)
            if u + v < 1:
                x = u + 0.5*v
                y = H*v
                # Check triangle
                if y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x):
                    pts.append([x, y])
        pts = np.array(pts)
        A_init = 0.01
        x0 = np.concatenate([[A_init], pts.flatten()])
        inits.append(x0)
        
    # 3. Hexagonal packing approximation
    # Center and rings
    pts = [[0.5, H/3]]
    # Ring 1
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.15
        x = 0.5 + r * np.cos(angle)
        y = H/3 + r * np.sin(angle) * 0.8 # Squash to fit triangle better?
        if y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x) and y >= 0:
            pts.append([x, y])
    # Ring 2
    for i in range(4):
        angle = i * np.pi / 2 + np.pi/4
        r = 0.3
        x = 0.5 + r * np.cos(angle)
        y = H/3 + r * np.sin(angle) * 0.6
        if y <= np.sqrt(3)*x and y <= np.sqrt(3)*(1-x) and y >= 0:
            pts.append([x, y])
    
    while len(pts) < N:
        pts.append([0.5, H/3])
    pts = np.array(pts[:N])
    A_init = 0.01
    x0 = np.concatenate([[A_init], pts.flatten()])
    inits.append(x0)

    # Run optimization on each init
    for x0 in inits:
        if time.time() > deadline - 5:
            break
        res_x, area = run_optimization(x0)
        if area > best_A:
            best_A = area
            best_x = res_x

    # If no optimization succeeded, fall back to a good heuristic
    if best_x is None:
        # Return the best initial guess
        # Or a known good configuration
        # Let's use the grid one
        pts = inits[0][1:].reshape(N, 2)
    else:
        pts = best_x[1:].reshape(N, 2)
        
    return pts

