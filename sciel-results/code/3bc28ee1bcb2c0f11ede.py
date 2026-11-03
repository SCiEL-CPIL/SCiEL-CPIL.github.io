import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius(points):
    """
    Computes the exact covering radius using ConvexHull.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-10
    if not np.all(valid):
        return np.pi
        
    u = n / norms[:, np.newaxis]
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    return np.max(radii)

def objective_and_grad(x, beta, k):
    """
    Objective: Soft-max of the k-th nearest neighbor distance for each point.
    x: (N, 3) array of points (not necessarily normalized).
    """
    points = x.reshape(N, 3)
    # Normalize points to sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Compute pairwise distances
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Set diagonal to infinity to exclude self
    np.fill_diagonal(dists, np.inf)
    
    # Get k-th nearest neighbor distance for each point
    # argsort gives indices, we take the k-th one (index k-1)
    # dists shape (N, N)
    # We want the k-th smallest distance for each row
    # Use partition for efficiency if k is small, but N=130 is small enough for sort
    sorted_dists = np.sort(dists, axis=1)
    kth_dists = sorted_dists[:, k-1]
    
    # Soft-max of kth_dists
    # To avoid overflow, subtract max
    max_val = np.max(kth_dists)
    exp_vals = np.exp(beta * (kth_dists - max_val))
    obj = (max_val + np.log(np.sum(exp_vals))) / beta
    
    # Gradient computation
    # This is complex. Let's use numerical gradient or a simpler analytical approximation.
    # For a structural change, let's try to compute the gradient analytically if possible,
    # or rely on a finite difference if needed, but L-BFGS-B needs gradients for speed.
    # Actually, let's use a simpler potential that is easier to differentiate.
    
    # Alternative: Sum of squared distances to k-th NN
    # obj = sum(kth_dists**2)
    # Gradient of sum of squared distances is easier.
    
    # Let's stick to the soft-max but compute gradient via finite differences? No, too slow.
    # Let's use the gradient of the sum of exponentials.
    # d/dx sum(exp(beta * d_i)) = sum(exp(beta * d_i) * beta * d(d_i)/dx)
    
    # d(d_i)/dx is the gradient of the angular distance to the k-th NN.
    # This is only non-zero for the k-th NN and changes if the k-th NN changes.
    
    # Given the complexity, let's use a different objective:
    # Minimize the sum of squared distances between all pairs? No, that's packing.
    # Minimize the maximum distance is key.
    
    # Let's use a "Repulsive Force" approach with a gradient descent loop, 
    # but with a more sophisticated force law.
    
    # Force: For each point i, calculate vector sum of (p_i - p_j) / dist(p_i, p_j)^2 
    # weighted by a function that emphasizes large gaps.
    
    # Let's return obj and a dummy gradient to use a derivative-free method like Powell?
    # Powell is slower but robust.
    
    return obj, None

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_radius = np.pi
    
    # Initializations
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand]
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            np.random.seed(restart * 100 + 7)
            points = init_fib + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Use Powell's method (derivative-free) on the soft-max objective
        # This avoids the need for complex gradients
        
        def objective(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pts = pts / norms
            
            dots = pts @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            np.fill_diagonal(dists, np.inf)
            
            # Use k=4 nearest neighbors to approximate local covering radius
            k = 4
            sorted_dists = np.sort(dists, axis=1)
            kth_dists = sorted_dists[:, k-1]
            
            # Soft max
            beta = 5.0
            max_val = np.max(kth_dists)
            exp_vals = np.exp(beta * (kth_dists - max_val))
            return (max_val + np.log(np.sum(exp_vals))) / beta
        
        res = minimize(objective, points.flatten(), method='Powell', options={'maxiter': 2000, 'disp': False})
        
        if res.success or res.fun < best_radius:
            final_points = res.x.reshape(N, 3)
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
            
            # Verify with exact convex hull radius
            exact_radius = compute_covering_radius(final_points)
            
            if exact_radius < best_radius:
                best_radius = exact_radius
                best_points = final_points.copy()
        
        if time.time() - start_time > time_limit:
            break

    if best_points is None:
        best_points = init_fib
        
    return best_points
