import numpy as np
import scipy.optimize as opt
import time
import itertools

N = 11
# Vertices of the unit-area equilateral triangle
# Area = sqrt(3)/4 * side^2 = 1 => side = 2 / sqrt(sqrt(3)) ... wait.
# Task says "unit-area triangle".
# The evaluator uses an equilateral triangle.
# "The notebook uses an equilateral triangle and divides by its area."
# "construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
# The vertices given in the prompt are (0,0), (1,0), (1/2, sqrt(3)/2).
# Let's check the area of this triangle.
# Base = 1. Height = sqrt(3)/2. Area = 0.5 * 1 * sqrt(3)/2 = sqrt(3)/4.
# Is this unit area? No.
# "Place 11 points on or inside a unit-area triangle... The notebook uses an equilateral triangle and divides by its area."
# This implies the *score* is normalized by the area of the triangle used.
# However, the `construct` function must return points in the specific coordinate system: (0,0), (1,0), (1/2, sqrt(3)/2).
# So we work in this coordinate system. The "unit-area" part might just be context for the metric, or the evaluator scales the result.
# "The evaluator measures the raw value".
# If the evaluator calculates area in this coordinate system, the max possible min-area is bounded by the triangle's area.
# We just maximize the raw area in this coordinate system.

A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])

def bary_to_cart(u, v):
    # P = u*A + v*B + (1-u-v)*C
    # A is 0
    # P = v*B + (1-u-v)*C
    # x = v*1 + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)*sqrt(3)/2
    x = 0.5 + 0.5 * v - 0.5 * u
    y = (1.0 - u - v) * (np.sqrt(3) / 2.0)
    return np.array([x, y])

def min_triangle_area(points):
    n = len(points)
    min_area = np.inf
    # Vectorized calculation for speed
    # We can compute all areas at once?
    # i, j, k
    # Let's use a loop over i, and vectorize j, k for that i.
    for i in range(n):
        pi = points[i]
        # Vectors from i to all other points
        diffs = points - pi # (n, 2)
        # We need j > i, k > j
        # This is tricky to vectorize fully without creating huge arrays.
        # N=11 is small. Triple loop is 165 iterations.
        # 165 * (simple math) is very fast.
        for j in range(i + 1, n):
            pj = points[j]
            # Vector from i to j
            v_ij = pj - pi
            for k in range(j + 1, n):
                pk = points[k]
                v_ik = pk - pi
                # Cross product
                cross = v_ij[0] * v_ik[1] - v_ij[1] * v_ik[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    # x is 2N vector: [u1, v1, u2, v2, ...]
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for being outside
    # u >= 0, v >= 0, u+v <= 1
    penalty = 0.0
    # Use squared penalty for smoothness
    # np.sum(np.maximum(0, -u)**2 + np.maximum(0, -v)**2 + np.maximum(0, u+v-1)**2)
    penalty = np.sum(np.maximum(0, -u)**2) + np.sum(np.maximum(0, -v)**2) + np.sum(np.maximum(0, u + v - 1)**2)
    
    if penalty > 1e-8:
        # Still calculate area to provide gradient-like info? 
        # Or just return large penalty?
        # If we return just penalty, the optimizer might ignore the area part.
        # Let's return -area + large_penalty
        pass
        
    points = np.array([bary_to_cart(u[i], v[i]) for i in range(N)])
    m_area = min_triangle_area(points)
    
    # We want to maximize m_area, so minimize -m_area
    # Add penalty to discourage going outside
    return -m_area + 1000.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    best_score = -np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Helper to generate a random point inside triangle
    def random_bary():
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                return u, v

    # Initializer 1: Random
    def init_random():
        u = rng.random(N)
        v = rng.random(N)
        # Rejection sampling for each point
        valid_u = []
        valid_v = []
        for i in range(N):
            while True:
                ui = rng.random()
                vi = rng.random()
                if ui + vi <= 1:
                    valid_u.append(ui)
                    valid_v.append(vi)
                    break
        x = np.zeros(2*N)
        x[0::2] = valid_u
        x[1::2] = valid_v
        return x

    # Initializer 2: Jittered Grid
    def init_grid():
        # Place points in a roughly triangular pattern
        # 11 points.
        # Row 0: 1
        # Row 1: 2
        # Row 2: 3
        # Row 3: 3
        # Row 4: 2
        # Total 11?
        # Let's just use a grid of 4x4 and pick 11?
        # Or use the barycentric coordinates of a triangular lattice.
        # Lattice points (i, j) in barycentric space.
        # Let's just use random for now, it's robust.
        return init_random()

    restart_count = 0
    while time.time() < deadline:
        x0 = init_random()
        
        # Try a structured start every 5th restart
        if restart_count % 5 == 0 and restart_count > 0:
             # Use best points from previous runs as start?
             if best_points is not None:
                 # Convert best_points to barycentric
                 # x = 0.5 + 0.5v - 0.5u => 0.5u - 0.5v = 0.5 - x => u - v = 1 - 2x
                 # y = (1-u-v)sqrt(3)/2 => 1-u-v = 2y/sqrt(3) => u+v = 1 - 2y/sqrt(3)
                 # u = ( (1-2x) + (1-2y/sqrt(3)) ) / 2
                 # v = ( (1-2y/sqrt(3)) - (1-2x) ) / 2
                 u_best = []
                 v_best = []
                 for p in best_points:
                     x, y = p
                     s3 = np.sqrt(3)
                     sum_uv = 1 - 2*y/s3
                     diff_uv = 1 - 2*x
                     u = (sum_uv + diff_uv) / 2
                     v = (sum_uv - diff_uv) / 2
                     # Clip to valid range
                     u = np.clip(u, 0, 1)
                     v = np.clip(v, 0, 1)
                     if u + v > 1:
                         # Project
                         scale = 1 / (u + v)
                         u *= scale
                         v *= scale
                     u_best.append(u)
                     v_best.append(v)
                 
                 x0 = np.zeros(2*N)
                 x0[0::2] = u_best
                 x0[1::2] = v_best
                 # Add small noise
                 x0 += rng.normal(0, 0.01, size=2*N)
        
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            # Check time
            if time.time() > deadline:
                break
                
            x_opt = res.x
            u_opt = x_opt[0::2]
            v_opt = x_opt[1::2]
            
            # Clip to valid range
            u_clipped = np.clip(u_opt, 0, 1)
            v_clipped = np.clip(v_opt, 0, 1)
            sum_uv = u_clipped + v_clipped
            mask = sum_uv > 1
            if np.any(mask):
                scale = 1 / sum_uv[mask]
                u_clipped[mask] *= scale
                v_clipped[mask] *= scale
            
            points = np.array([bary_to_cart(u_clipped[i], v_clipped[i]) for i in range(N)])
            
            score = min_triangle_area(points)
            
            if score > best_score:
                best_score = score
                best_points = points
                
        except Exception as e:
            pass
            
        restart_count += 1
        
    if best_points is None:
        # Fallback
        points = []
        for _ in range(N):
            while True:
                u, v = rng.random(), rng.random()
                if u + v <= 1:
                    points.append(bary_to_cart(u, v))
                    break
        best_points = np.array(points)
        
    return best_points
