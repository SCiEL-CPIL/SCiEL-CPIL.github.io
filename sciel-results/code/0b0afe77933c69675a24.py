import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triangles (i < j < k)
indices = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            indices.append((i, j, k))
indices = np.array(indices)
idx_i = indices[:, 0]
idx_j = indices[:, 1]
idx_k = indices[:, 2]

def min_triangle_area(points):
    """
    points: numpy array of shape (N, 2)
    returns: minimum area of any triangle formed by 3 points
    """
    Pi = points[idx_i]
    Pj = points[idx_j]
    Pk = points[idx_k]
    
    # Vectors
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Determinant (2x2)
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(dets)
    
    return np.min(areas)

def objective(x):
    """
    x: flattened array of 22 coordinates (x1, y1, ..., x11, y11)
    returns: negative of the minimum area
    """
    points = x.reshape(N, 2)
    area = min_triangle_area(points)
    return -area

def constraints(x):
    """
    List of constraint dictionaries for SLSQP
    """
    points = x.reshape(N, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    
    cons = []
    
    # y >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=0: x[i] - 0, 'args': ()}) # This is per point, need vectorized or list of 11
    # Better to define vectorized constraints or a list of 33 constraints
    
    # Let's define the 33 inequality constraints explicitly for SLSQP
    # It accepts a list of dictionaries.
    
    # 1. y_i >= 0  => y_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i + 1]})
        
    # 2. y_i <= sqrt(3) * x_i  => sqrt(3) * x_i - y_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.sqrt(3) * x[2*i] - x[2*i + 1]})
        
    # 3. y_i <= -sqrt(3) * x_i + sqrt(3)  => -sqrt(3) * x_i - y_i + sqrt(3) >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: -np.sqrt(3) * x[2*i] - x[2*i + 1] + np.sqrt(3)})
        
    return cons

# Note: SLSQP might be slow with 33 individual constraints.
# Alternative: Use penalty method with Nelder-Mead?
# Or use 'trust-constr' which handles constraints more efficiently?
# Let's stick to SLSQP but maybe combine constraints?
# SLSQP can take a single function returning an array of constraint values.
# Let's try to vectorize the constraints into a single function returning an array.

def vectorized_constraints(x):
    points = x.reshape(N, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    
    c1 = ys # y >= 0
    c2 = np.sqrt(3) * xs - ys # y <= sqrt(3)x
    c3 = -np.sqrt(3) * xs - ys + np.sqrt(3) # y <= -sqrt(3)x + sqrt(3)
    
    return np.concatenate([c1, c2, c3])

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    best_score = -np.inf
    best_points = None
    
    # Generate initial guesses
    # 1. Random points inside triangle
    def random_points_in_triangle(n, seed):
        rng = np.random.default_rng(seed)
        points = []
        for _ in range(n):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            # Barycentric coordinates u, v, w=1-u-v
            # Point = u*A + v*B + w*C
            # A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
            # x = v*1 + w*0.5 = v + (1-u-v)*0.5 = 0.5 + 0.5v - 0.5u
            # y = v*0 + w*sqrt(3)/2 = (1-u-v)*sqrt(3)/2
            x = 0.5 + 0.5 * v - 0.5 * u
            y = (1 - u - v) * np.sqrt(3) / 2
            points.append([x, y])
        return np.array(points)

    # 2. Grid-like points
    def grid_points_in_triangle(n, seed):
        # Simple heuristic: spread points
        rng = np.random.default_rng(seed)
        # Just use random for now, SLSQP will fix the shape
        return random_points_in_triangle(n, seed)

    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initial guess
        x0 = random_points_in_triangle(N, seed=100 + i).flatten()
        
        # Run optimization
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=[(0, 1)] * N + [(0, np.sqrt(3)/2)] * N, # Rough bounds, constraints are tighter
                constraints=[{'type': 'ineq', 'fun': vectorized_constraints}],
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                # Check if result is valid (inside triangle)
                pts = res.x.reshape(N, 2)
                # Verify constraints manually
                valid = True
                for p in pts:
                    if p[1] < -1e-5 or p[0] < -1e-5 or p[0] > 1 + 1e-5:
                        valid = False
                    if p[1] > np.sqrt(3) * p[0] + 1e-5:
                        valid = False
                    if p[1] > -np.sqrt(3) * p[0] + np.sqrt(3) + 1e-5:
                        valid = False
                if valid:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_points = pts
                        
        except Exception as e:
            pass

    if best_points is None:
        # Fallback to random
        best_points = random_points_in_triangle(N, seed=0)
        
    return best_points

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Vertices of the equilateral triangle
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triangles (i < j < k)
_indices_list = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _indices_list.append((i, j, k))
_indices_arr = np.array(_indices_list)
_idx_i = _indices_arr[:, 0]
_idx_j = _indices_arr[:, 1]
_idx_k = _indices_arr[:, 2]

def _min_triangle_area(points):
    Pi = points[_idx_i]
    Pj = points[_idx_j]
    Pk = points[_idx_k]
    v1 = Pj - Pi
    v2 = Pk - Pi
    dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _objective(x):
    points = x.reshape(N, 2)
    area = _min_triangle_area(points)
    return -area

def _vectorized_constraints(x):
    points = x.reshape(N, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    c1 = ys 
    c2 = np.sqrt(3) * xs - ys 
    c3 = -np.sqrt(3) * xs - ys + np.sqrt(3) 
    return np.concatenate([c1, c2, c3])

def construct():
    start_time = time.time()
    timeout = 140.0 

    best_score = -np.inf
    best_points = None

    def random_points_in_triangle(n, seed):
        rng = np.random.default_rng(seed)
        points = []
        for _ in range(n):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = 0.5 + 0.5 * v - 0.5 * u
            y = (1 - u - v) * np.sqrt(3) / 2
            points.append([x, y])
        return np.array(points)

    num_restarts = 15

    for i in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        x0 = random_points_in_triangle(N, seed=100 + i).flatten()
        
        try:
            res = minimize(
                _objective, 
                x0, 
                method='SLSQP', 
                bounds=[(0, 1)] * N + [(0, np.sqrt(3)/2)] * N,
                constraints=[{'type': 'ineq', 'fun': _vectorized_constraints}],
                options={'maxiter': 300, 'ftol': 1e-6}
            )
            
            if res.fun < best_score:
                pts = res.x.reshape(N, 2)
                # Quick validity check
                valid = True
                for p in pts:
                    if p[1] < -1e-4 or p[0] < -1e-4 or p[0] > 1 + 1e-4:
                        valid = False
                    if p[1] > np.sqrt(3) * p[0] + 1e-4:
                        valid = False
                    if p[1] > -np.sqrt(3) * p[0] + np.sqrt(3) + 1e-4:
                        valid = False
                if valid:
                    best_score = res.fun
                    best_points = pts
        except Exception:
            pass

    if best_points is None:
        best_points = random_points_in_triangle(N, seed=0)
        
    return best_points
# EVOLVE-BLOCK-END
