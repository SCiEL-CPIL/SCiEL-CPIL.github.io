import numpy as np
from scipy.optimize import differential_evolution

def construct():
    n = 8
    
    def score(params):
        r1, r2, beta = params
        
        # Construct two squares
        # Square 1: radius r1, angle 0
        # Square 2: radius r2, angle beta
        
        # Avoid zero radius
        if r1 < 1e-6 or r2 < 1e-6:
            return 1e10
            
        points = []
        for k in range(4):
            angle = k * np.pi / 2
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
            
        for k in range(4):
            angle = beta + k * np.pi / 2
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
            
        points = np.array(points)
        
        # Calculate distances
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-10:
                    return 1e10
                dists.append(d)
                
        dists = np.array(dists)
        D = np.max(dists)
        if D < 1e-10:
            return 1e10
            
        val = np.sum(np.log(dists)) - 28 * np.log(D)
        return -val

    # Bounds for differential evolution
    # r1, r2 can be anything, but relative scale matters.
    # Let's search r1, r2 in [0.1, 10.0] and beta in [0, pi/2]
    # Due to symmetry, beta in [0, pi/2] covers all relative rotations.
    # Also, swapping r1 and r2 is equivalent to changing beta? 
    # If we swap r1 and r2, the set of points is the same if we adjust beta?
    # Not exactly, but the space is symmetric.
    
    bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, np.pi / 2)]
    
    # Use differential evolution for global search
    # seed for reproducibility
    res = differential_evolution(score, bounds, seed=42, maxiter=100, tol=1e-8, polish=True)
    
    if res.success and res.fun < 1e9:
        r1, r2, beta = res.x
        points = []
        for k in range(4):
            angle = k * np.pi / 2
            points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
        for k in range(4):
            angle = beta + k * np.pi / 2
            points.append([r2 * np.cos(angle), r2 * np.sin(angle)])
        points = np.array(points)
        
        # Scale to diameter 2? 
        # The evaluator scales to diameter 2. 
        # We can return the points as is.
        
        # Check distinctness
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(points[i] - points[j]) < 1e-5:
                    valid = False
        if not valid:
            # Fallback to regular octagon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return [[np.cos(a), np.sin(a)] for a in angles]
            
        return points.tolist()
    else:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return [[np.cos(a), np.sin(a)] for a in angles]
