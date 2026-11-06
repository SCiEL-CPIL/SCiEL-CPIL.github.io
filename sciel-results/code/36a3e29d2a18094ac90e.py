import numpy as np
from scipy.optimize import minimize

def construct():
    n = 8
    
    # Define the 8 points based on parameters u and v
    # Points are the orbit of (u, v) under D4 symmetry
    # (u, v), (-u, v), (-u, -v), (u, -v)
    # (v, u), (-v, u), (-v, -u), (v, -u)
    
    def get_points(params):
        u, v = params
        # Ensure distinctness by keeping u > v > 0 or similar? 
        # The optimizer might find u < v or negative values.
        # The set of points is the same for (u,v), (v,u), (-u,v) etc due to symmetry.
        # We can just use the raw u, v.
        p1 = np.array([u, v])
        p2 = np.array([-u, v])
        p3 = np.array([-u, -v])
        p4 = np.array([u, -v])
        p5 = np.array([v, u])
        p6 = np.array([-v, u])
        p7 = np.array([-v, -u])
        p8 = np.array([v, -u])
        return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

    def objective(params):
        points = get_points(params)
        
        # Check for duplicates
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    return 1e10 # Large penalty
        
        # Calculate pairwise distances
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
            
        # Maximize sum(log(d)) - 14*log(D)
        # Minimize negative of that
        val = np.sum(np.log(dists)) - 14 * np.log(D)
        return -val

    # Initial guesses for u, v
    # 1. Regular Octagon: u = cos(pi/8), v = sin(pi/8) (approx 0.92, 0.38)
    # Actually, regular octagon vertices are at angles 0, 45, 90...
    # If we use the orbit (u,v), the regular octagon corresponds to u = cos(pi/8), v = sin(pi/8) ?
    # Let's check: (cos, sin), (-cos, sin)...
    # Angle of (cos, sin) is pi/8.
    # Angle of (sin, cos) is pi/2 - pi/8 = 3pi/8.
    # The angles covered are pi/8, 3pi/8, 5pi/8, 7pi/8...
    # This is not the regular octagon angles (0, pi/4, pi/2...).
    # The regular octagon has points at 0, 45, 90...
    # To get point at 0: (u, 0) -> v=0. But then points duplicate.
    # So the D4 orbit (u,v) with u!=v does NOT include the regular octagon?
    # Wait. The regular octagon has C8 symmetry, which is a subgroup of... no, C8 is not D4.
    # D4 has 8 symmetries. C8 has 8 symmetries.
    # The orbit of a generic point under D4 has 8 points.
    # The regular octagon is the orbit of a point at angle 0 under C8.
    # Is the regular octagon a D4 orbit?
    # If we take a point at angle 0 (1,0), its D4 orbit is (1,0), (-1,0), (0,1), (0,-1). Only 4 points.
    # So the regular octagon is NOT a single D4 orbit of a generic point.
    # It is the union of two D4 orbits? Or it has higher symmetry.
    # If we restrict to D4 symmetric sets of 8 distinct points, we cannot represent the regular octagon directly as a single orbit (u,v) unless we allow u,v to be complex? No.
    # The regular octagon consists of 4 points on axes and 4 on diagonals.
    # Points: (1,0), (0,1), (-1,0), (0,-1) and (1/sqrt2, 1/sqrt2)...
    # These are two orbits: one of (1,0) and one of (1/sqrt2, 1/sqrt2).
    # My parameterization (u,v) generates ONE orbit.
    # So my parameterization excludes the regular octagon!
    
    # I need a parameterization that includes the regular octagon.
    # Let's use 4 parameters: 
    # 4 points on axes: (a, 0), (0, a), (-a, 0), (0, -a)
    # 4 points on diagonals: (b, b), (-b, b), (-b, -b), (b, -b)
    # This has D4 symmetry.
    # Regular octagon: a = 1, b = 1/sqrt2.
    # This covers the regular octagon.
    
    def get_points_2(params):
        a, b = params
        p1 = np.array([a, 0])
        p2 = np.array([0, a])
        p3 = np.array([-a, 0])
        p4 = np.array([0, -a])
        p5 = np.array([b, b])
        p6 = np.array([-b, b])
        p7 = np.array([-b, -b])
        p8 = np.array([b, -b])
        return np.array([p1, p2, p3, p4, p5, p6, p7, p8])

    def objective_2(params):
        points = get_points_2(params)
        
        # Check for duplicates
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(points[i] - points[j]) < 1e-10:
                    return 1e10
        
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
            
        val = np.sum(np.log(dists)) - 14 * np.log(D)
        return -val

    # Initial guess for regular octagon: a=1, b=1/sqrt2
    # We can scale arbitrarily, so let's start with a=1, b=0.7
    starts = [
        [1.0, 1.0/np.sqrt(2)], # Regular Octagon
        [1.0, 0.5],            # Perturbed
        [1.0, 0.8],            # Perturbed
        [0.5, 1.0],            # Swapped
        [1.0, 0.01],           # Near degenerate
        [1.0, 1.0],            # Two squares same size? (Points: (1,0),(0,1)... (1,1)...) 
                               # Distances: (1,0) to (1,1) is 1. (1,0) to (0,1) is sqrt(2).
                               # This is a valid configuration.
    ]

    best_val = -np.inf
    best_params = None

    for start in starts:
        try:
            res = minimize(objective_2, start, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to regular octagon
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return [[np.cos(a), np.sin(a)] for a in angles]

    points = get_points_2(best_params)
    
    # Verify distinctness
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-5:
                valid = False
    if not valid:
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return [[np.cos(a), np.sin(a)] for a in angles]

    return points.tolist()
