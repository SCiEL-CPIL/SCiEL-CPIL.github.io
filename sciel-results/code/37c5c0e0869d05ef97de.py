import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])

# Precompute indices for all triangles
TRIPLES = list(combinations(range(N), 3))

def penalty(points):
    pen = 0.0
    for p in points:
        x, y = p
        if y < 0:
            pen += y*y
        d1 = np.sqrt(3)*x - y
        if d1 < 0:
            pen += d1*d1
        d2 = np.sqrt(3)*(1-x) - y
        if d2 < 0:
            pen += d2*d2
    return pen

def min_triangle_area(points):
    min_area = float('inf')
    for i, j, k in TRIPLES:
        p1, p2, p3 = points[i], points[j], points[k]
        a = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
        if a < min_area:
            min_area = a
    return min_area

def objective(x_flat):
    points = x_flat.reshape(N, 2)
    pen = penalty(points)
    if pen > 1e-5:
        return 1e6 + pen
    return -min_triangle_area(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_score = -np.inf
    best_points = None
    
    bounds = [(0, 1), (0, np.sqrt(3)/2)] * N
    
    # Structured Inits
    def get_inits():
        inits = []
        # Init 1: Vertices, Midpoints, Centroid, + 4 interior
        pts = [V0, V1, V2, (V0+V1)/2, (V1+V2)/2, (V2+V0)/2, (V0+V1+V2)/3]
        # Add 4 more points, e.g., midpoints of the sub-triangles
        # Or just randomize a bit
        rng = np.random.RandomState(1)
        for _ in range(4):
             u, v = rng.random(), rng.random()
             if u+v < 1:
                 p = u*V1 + v*V2
                 pts.append(p)
        inits.append(np.array(pts).flatten())
        
        # Init 2: Grid
        # 3x3 grid approx
        pts2 = []
        for i in range(3):
            for j in range(3):
                if i+j < 3: # Rough triangle mask
                    u = i/2.0
                    v = j/2.0
                    if u+v <= 1:
                         p = u*V1 + v*V2
                         pts2.append(p)
        # This gives 6 points. Need 5 more.
        # Just fill with random
        rng2 = np.random.RandomState(2)
        while len(pts2) < N:
             u, v = rng2.random(), rng2.random()
             if u+v < 1:
                 p = u*V1 + v*V2
                 pts2.append(p)
        inits.append(np.array(pts2).flatten())
        
        return inits

    inits = get_inits()
    
    # Run Local Optimization from Inits
    for x0 in inits:
        if time.time() > deadline - 10: break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < 1e5:
                pts = res.x.reshape(N, 2)
                if penalty(pts) < 1e-6:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
        except:
            pass

    # Run DE + Polish Loop
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        seed = attempt * 1000
        try:
            res_de = differential_evolution(
                objective, 
                bounds, 
                seed=seed,
                maxiter=50,
                popsize=10,
                tol=1e-3,
                atol=1e-5,
                init='sobol'
            )
            
            if res_de.fun < 1e5:
                # Polish
                res_polish = minimize(
                    objective, 
                    res_de.x, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9}
                )
                
                if res_polish.fun < res_de.fun:
                    res_de = res_polish
                
                if res_de.fun < 1e5:
                    pts = res_de.x.reshape(N, 2)
                    if penalty(pts) < 1e-6:
                        score = -res_de.fun
                        if score > best_score:
                            best_score = score
                            best_points = pts.copy()
        except:
            pass

    if best_points is None:
        # Fallback
        rng = np.random.RandomState(0)
        pts = []
        while len(pts) < N:
            u, v = rng.random(), rng.random()
            if u+v < 1:
                p = u*V1 + v*V2
                pts.append(p)
        best_points = np.array(pts)

    return best_points
