"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle."""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for speed
    # points shape (n, 2)
    # Area = 0.5 * | (pj - pi) x (pk - pi) |
    # We can compute all areas for a fixed i by vectorizing j and k
    for i in range(n):
        pi = points[i]
        # Vectors from pi to all other points
        vecs = points - pi  # shape (n, 2)
        # Cross product of vecs[j] and vecs[k] for all j, k
        # cross_jk = vecs[j,0]*vecs[k,1] - vecs[j,1]*vecs[k,0]
        # We want min over j < k, j != i, k != i
        # To avoid checking j != i and k != i explicitly in the min, we can set areas involving i to inf
        # Or simply compute all and ignore i
        
        # Efficiently compute all pairwise cross products
        # x_j * y_k - y_j * x_k
        # This is a matrix multiplication / outer product style
        # Let X = vecs[:, 0], Y = vecs[:, 1]
        # Cross matrix C where C[j,k] = X[j]*Y[k] - Y[j]*X[k]
        X = vecs[:, 0]
        Y = vecs[:, 1]
        C = np.outer(X, Y) - np.outer(Y, X)
        
        # We need min of 0.5 * |C[j,k]| for j < k, j != i, k != i
        # Since C is antisymmetric, C[j,k] = -C[k,j]
        # We can take the upper triangle
        # Set diagonal and i-th row/col to inf to ignore them
        C_abs = np.abs(C)
        # Set i-th row and column to inf
        C_abs[i, :] = float('inf')
        C_abs[:, i] = float('inf')
        # Set diagonal to inf (j==k area is 0, but we want distinct points)
        np.fill_diagonal(C_abs, float('inf'))
        
        # Find min in upper triangle
        # To be safe and fast, just find global min of C_abs (excluding i and diag)
        # Since it's symmetric, min in upper == min in lower
        current_min = np.min(C_abs)
        if current_min < min_area:
            min_area = current_min
            if min_area < 1e-9: # Early exit if degenerate
                return 0.0

    return (0.5 * min_area) / AREA


def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9


def _params_to_points(params):
    """Convert 2*N parameters to N points inside the triangle using barycentric coordinates."""
    # params shape (2*N,)
    # For each point, we use 2 params u, v such that u >= 0, v >= 0, u+v <= 1
    # To map from R^2 to the triangle domain, we can use sigmoid or tanh to bound u, v
    # Let's use u = sigmoid(p1), v = sigmoid(p2) * (1 - u)
    # This ensures u in (0,1) and v in (0, 1-u)
    points = []
    for i in range(N):
        u = 1.0 / (1.0 + np.exp(-params[2*i]))
        v = 1.0 / (1.0 + np.exp(-params[2*i+1])) * (1.0 - u)
        p = V0 * (1 - u - v) + V1 * u + V2 * v
        points.append(p)
    return np.array(points)


def _objective(params):
    """Objective function for minimization: negative of min triangle area."""
    points = _params_to_points(params)
    # Check if all points are strictly inside (with small margin)
    # The mapping guarantees they are inside, but let's be safe
    min_area = _min_triangle_area(points)
    return -min_area


def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return np.array(pts)


def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)


def _points_to_params(points):
    """Inverse of _params_to_points. For initial params for Nelder-Mead."""
    # We need to find u, v for each point such that p = (1-u-v)V0 + u V1 + v V2
    # V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    # x = u + 0.5v
    # y = sqrt(3)/2 * v
    # v = 2y/sqrt(3)
    # u = x - 0.5v
    params = []
    for p in points:
        x, y = p
        v = 2 * y / np.sqrt(3)
        u = x - 0.5 * v
        # Inverse sigmoid
        # u = sigmoid(p1) => p1 = log(u/(1-u))
        # v = sigmoid(p2)*(1-u) => sigmoid(p2) = v/(1-u) => p2 = log( (v/(1-u)) / (1 - v/(1-u)) )
        # Handle boundaries
        eps = 1e-6
        u = np.clip(u, eps, 1-eps)
        v = np.clip(v, eps, 1-eps)
        p1 = np.log(u / (1 - u))
        s2 = v / (1 - u)
        s2 = np.clip(s2, eps, 1-eps)
        p2 = np.log(s2 / (1 - s2))
        params.append([p1, p2])
    return np.array(params).flatten()


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy: Multi-start Nelder-Mead
    # 1. Start with structured initial
    # 2. Start with several random initializations
    
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(10):
        initial_configs.append(_generate_random_points(rng))
    
    # Run Nelder-Mead on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Convert to params
        init_params = _points_to_params(init_pts)
        
        # Nelder-Mead settings
        # maxiter is important to prevent hanging
        # We want to give it enough iterations to converge
        # Since we have multiple starts, we can limit iterations per start
        max_iter_per_start = 500 if idx == 0 else 200
        
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': max_iter_per_start, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if res.success or res.fun < 0: # If it improved
                best_params = res.x
                best_points_candidate = _params_to_points(best_params)
                current_min = _min_triangle_area(best_points_candidate)
                
                if current_min > best_min:
                    best_min = current_min
                    best_points = best_points_candidate.copy()
        except Exception as e:
            continue

    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
# EVOLVE-BLOCK-END
