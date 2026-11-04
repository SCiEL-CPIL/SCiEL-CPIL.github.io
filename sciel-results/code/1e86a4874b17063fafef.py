import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def construct():
    # Parameters: 13 edge weights for the cycle graph
    # w[0] is edge between 0 and 1, w[1] between 1 and 2, ..., w[11] between 11 and 12, w[12] between 12 and 0
    
    def get_points_from_weights(w):
        # Ensure weights are positive
        w = np.abs(w) + 1e-6
        
        # Construct the weighted Laplacian of the cycle graph
        # L is N x N
        L = np.zeros((N, N))
        for i in range(N):
            # Diagonal
            L[i, i] = w[i] + w[(i - 1) % N]
            # Off-diagonals
            L[i, (i + 1) % N] = -w[i]
            L[i, (i - 1) % N] = -w[(i - 1) % N]
        
        # Compute eigenvalues and eigenvectors
        # L is symmetric
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            # Fallback to regular polygon if eigendecomposition fails
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack((np.cos(angles), np.sin(angles)))
        
        # eigvecs are columns, sorted by increasing eigenvalue
        # eigvecs[:, 0] is the constant vector (eigenvalue 0)
        # eigvecs[:, 1] is the first non-trivial mode
        # eigvecs[:, 2] is the second non-trivial mode
        
        v1 = eigvecs[:, 1]
        v2 = eigvecs[:, 2]
        
        # Form points
        # We might need to rotate or scale, but for now, just use components
        points = np.column_stack((v1, v2))
        
        return points

    def objective(x):
        w = x
        pts = get_points_from_weights(w)
        
        # Check if points are degenerate
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            return 1e6
            
        # Calculate convex hull area
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume
        except Exception:
            # If not convex or degenerate, return large penalty
            return 1e6
            
        if hull_area < 1e-10:
            return 1e6
            
        # Calculate minimum triangle area
        min_area = float('inf')
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = pts[i]
                    p2 = pts[j]
                    p3 = pts[k]
                    area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                    if area < min_area:
                        min_area = area
        
        if min_area == float('inf'):
            return 1e6
            
        score = min_area / hull_area
        return -score

    # Initial guess: regular weights
    w0 = np.ones(N)
    
    # Bounds: weights should be positive
    bounds = [(1e-3, 10.0) for _ in range(N)]
    
    # Optimize
    res = minimize(objective, w0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8})
    
    if res.success:
        w_opt = res.x
    else:
        w_opt = w0
        
    points = get_points_from_weights(w_opt)
    
    # Fallback if optimization produced garbage
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return points
