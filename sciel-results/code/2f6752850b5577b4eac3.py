"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def is_inside(p):
        # Check if point p is inside the triangle
        # Using barycentric coordinates or cross products
        # Cross product sign check
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, A, B)
        d2 = sign(p, B, C)
        d3 = sign(p, C, A)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)

    def calculate_min_area(points):
        min_area = np.inf
        # Vectorized calculation is complex due to min, loop is fine for N=11
        # 165 triangles
        for i, j, k in itertools.combinations(range(N), 3):
            p1 = points[i]
            p2 = points[j]
            p3 = points[k]
            # Area = 0.5 * |det(p2-p1, p3-p1)|
            v1 = p2 - p1
            v2 = p3 - p1
            area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
            if area < min_area:
                min_area = area
        return min_area

    def objective(params):
        points = params.reshape(N, 2)
        
        # Penalty for being outside
        penalty = 0.0
        for p in points:
            if not is_inside(p):
                # Calculate distance to boundary roughly for penalty
                # Simple penalty: distance to closest vertex or edge?
                # Just use a large constant penalty for now, or distance to triangle
                # Distance to triangle:
                # Project to edges if outside
                # For simplicity, just add a large penalty
                penalty += 10.0
        
        if penalty > 0:
            return penalty
        
        min_area = calculate_min_area(points)
        return -min_area

    def generate_initial_guess(strategy):
        if strategy == 0:
            # Random
            points = []
            while len(points) < N:
                u, v = np.random.random(2)
                if u + v < 1:
                    p = u * B + v * C # Barycentric (1-u-v)A + uB + vC
                    points.append(p)
            return np.array(points).flatten()
        elif strategy == 1:
            # Triangular lattice 10 points + 1 center
            # Lattice of side 3 (10 points)
            pts = []
            for i in range(4):
                for j in range(4-i):
                    # Barycentric coords (1-i-j, i, j)
                    p = (1-i-j)*A + i*B + j*C
                    pts.append(p)
            # Add center
            center = (A+B+C)/3
            pts.append(center)
            return np.array(pts).flatten()
        elif strategy == 2:
            # Boundary points
            pts = []
            # Vertices
            pts.append(A)
            pts.append(B)
            pts.append(C)
            # Midpoints
            pts.append((A+B)/2)
            pts.append((B+C)/2)
            pts.append((C+A)/2)
            # Centroids of 4 sub-triangles?
            # Or just random fill for rest
            while len(pts) < N:
                u, v = np.random.random(2)
                if u + v < 1:
                    p = u * B + v * C
                    pts.append(p)
            return np.array(pts).flatten()
        else:
            # Perturbed lattice
            base = generate_initial_guess(1)
            noise = np.random.normal(0, 0.05, base.shape)
            return base + noise

    best_result = None
    best_score = -np.inf
    
    strategies = [0, 1, 2, 3, 4] # 5 restarts
    
    for s in strategies:
        if time.time() > deadline:
            break
            
        x0 = generate_initial_guess(s)
        
        # Nelder-Mead is good for non-smooth
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success or res.fun < best_score:
                # Check if points are actually inside
                pts = res.x.reshape(N, 2)
                valid = True
                for p in pts:
                    if not is_inside(p):
                        valid = False
                        break
                if valid:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_result = res.x
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to seed-like random
        points = []
        while len(points) < N:
            u, v = np.random.random(2)
            if u + v < 1:
                p = u * B + v * C
                points.append(p)
        return np.array(points)
        
    return best_result.reshape(N, 2)
# EVOLVE-BLOCK-END
