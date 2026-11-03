import numpy as np
import time
from scipy.spatial import Delaunay

N = 130
DEADLINE = 140.0  # 10s safety margin

def _covering_radius(points):
    """
    Compute the covering radius (max circumradius of Delaunay triangles) for points on the unit sphere.
    """
    if len(points) < 3:
        return np.inf
        
    # Delaunay triangulation
    try:
        tri = Delaunay(points)
    except Exception:
        return np.inf
        
    # Get triangle vertices
    # tri.simplices has shape (M, 3)
    simplices = tri.simplices
    
    # Compute circumradius for each triangle
    # R = (a*b*c) / (4 * Area)
    # Area = 0.5 * | (b-a) x (c-a) |
    
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]
    
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)
    
    # Area
    cross = np.cross(p1 - p0, p2 - p0)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    
    # Avoid division by zero
    eps = 1e-12
    area = np.maximum(area, eps)
    
    radii = (a * b * c) / (4.0 * area)
    
    return np.max(radii)

def _optimize(points, Y, beta, gamma, steps, lr):
    """
    Gradient descent on soft-max of soft-min distances.
    """
    points = points.copy()
    Y = Y.copy()
    
    for step in range(steps):
        # Compute distances D (shape |Y| x N)
        # D[y, i] = || Y[y] - points[i] ||^2
        # Use broadcasting
        # Y: (M, 3), points: (N, 3)
        # D = np.sum(Y**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * Y @ points.T
        D = np.sum(Y**2, axis=1)[:, None] + np.sum(points**2, axis=1)[None, :] - 2 * (Y @ points.T)
        D = np.maximum(D, 0) # numerical stability
        
        # Soft-min weights w (shape M x N)
        # w[y, i] = exp(-beta * D[y, i]) / sum_j exp(-beta * D[y, j])
        exp_neg_beta_D = np.exp(-beta * D)
        sum_exp = np.sum(exp_neg_beta_D, axis=1, keepdims=True)
        w = exp_neg_beta_D / sum_exp
        
        # Soft-min values m (shape M)
        # m[y] = -1/beta * log(sum_j exp(-beta * D[y, j]))
        m = -1.0 / beta * np.log(sum_exp.flatten())
        
        # Soft-max weights v (shape M)
        # v[y] = exp(gamma * m[y]) / sum_z exp(gamma * m[z])
        exp_gamma_m = np.exp(gamma * m)
        sum_exp_v = np.sum(exp_gamma_m)
        v = exp_gamma_m / sum_exp_v
        
        # Gradients
        # g_k = -2 * sum_y (v[y] * w[y, k]) * Y[y] + 2 * p_k * sum_y (v[y] * w[y, k])
        # A = W^T * diag(V)  -> (N x M) * (M x M) ? No.
        # w is (M, N). v is (M,).
        # We want sum_y v[y] * w[y, k] * Y[y]
        # This is (W^T * diag(v)) @ Y^T ?
        # W^T is (N, M). diag(v) is (M, M).
        # Let's do: weighted_W = w * v[:, None]  (M, N)
        # Then sum_y weighted_W[y, k] * Y[y] is (weighted_W.T @ Y) -> (N, 3)
        weighted_W = w * v[:, None]
        
        S1 = weighted_W.T @ Y  # (N, 3)
        S2 = np.sum(weighted_W, axis=0)  # (N,)
        
        gradients = -2.0 * S1 + 2.0 * (S2[:, None] * points)
        
        # Update
        points -= lr * gradients
        
        # Project to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        points /= norms
        
    return points

def construct():
    start_time = time.time()
    
    # Sample points Y on the sphere for the surrogate objective
    # Use a dense set, e.g., Fibonacci spiral with more points
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initializations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(P_fib)
    
    # 2. Rotated Fibonacci
    # Rotate around z-axis
    angle = np.pi / 7
    R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    inits.append(P_fib @ R_z.T)
    
    # 3. Random
    np.random.seed(42)
    P_rand = np.random.randn(N, 3)
    P_rand /= np.linalg.norm(P_rand, axis=1, keepdims=True)
    inits.append(P_rand)
    
    # 4. Jittered Fibonacci
    P_jit = P_fib + 0.1 * np.random.randn(N, 3)
    P_jit /= np.linalg.norm(P_jit, axis=1, keepdims=True)
    inits.append(P_jit)
    
    best_points = None
    best_radius = np.inf
    
    beta = 20.0
    gamma = 20.0
    steps = 300
    lr = 0.01
    
    for idx, init in enumerate(inits):
        if time.time() > DEADLINE:
            break
            
        # Run optimization
        optimized = _optimize(init, Y, beta, gamma, steps, lr)
        
        # Calculate true covering radius
        radius = _covering_radius(optimized)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized.copy()
            
            # Optional: If we have time, run a second polish pass with higher beta/gamma?
            # Or just accept.
            
    if best_points is None:
        # Fallback to Fibonacci
        best_points = P_fib
        
    return best_points
