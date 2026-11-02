import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

def calculate_score(pts):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if hull area is 0.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = len(pts)
    
    # Calculate Convex Hull Area
    if n < 3:
        return 0.0
        
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-12:
            return 0.0
    except Exception:
        return 0.0
        
    # Calculate min triangle area
    min_area = np.inf
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Vectorized calculation of all triangle areas
    # Create indices for all triples i < j < k
    # This might be overkill for N=13, but let's try a simple loop first for safety and speed in Python
    # Loop is 286 iterations. Very fast.
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * np.abs(x[i] * (y[j] - y[k]) + x[j] * (y[k] - y[i]) + x[k] * (y[i] - y[j]))
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        return 0.0
                        
    return min_area / hull_area

def construct():
    rng = np.random.default_rng(123)
    
    # Initialize with regular 13-gon
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack((np.cos(theta), np.sin(theta)))
    
    best_pts = init_pts.copy()
    best_score = calculate_score(best_pts)
    
    # Simulated Annealing
    T = 0.1
    T_min = 1e-5
    alpha = 0.995
    max_iter = 3000
    step_size = 0.05
    
    current_pts = init_pts.copy()
    current_score = best_score
    
    for i in range(max_iter):
        # Generate new candidate
        # Add noise
        new_pts = current_pts + rng.normal(0, step_size, size=(N, 2))
        
        # Normalize: Center and scale to unit radius
        new_pts = new_pts - np.mean(new_pts, axis=0)
        norms = np.linalg.norm(new_pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 1e-10:
            new_pts = new_pts / max_norm
        else:
            # If all points collapsed, restart
            new_pts = init_pts + rng.normal(0, 0.1, size=(N, 2))
            new_pts = new_pts - np.mean(new_pts, axis=0)
            norms = np.linalg.norm(new_pts, axis=1)
            new_pts = new_pts / np.max(norms)
            
        new_score = calculate_score(new_pts)
        
        # Acceptance
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                # Metropolis
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        # Cool down
        T = max(T * alpha, T_min)
        step_size *= 0.999
        
    # Local Refinement
    def objective(flat_pts):
        pts_2d = flat_pts.reshape(-1, 2)
        # Re-normalize to prevent drift during optimization
        pts_2d = pts_2d - np.mean(pts_2d, axis=0)
        norms = np.linalg.norm(pts_2d, axis=1)
        max_norm = np.max(norms)
        if max_norm > 1e-10:
            pts_2d = pts_2d / max_norm
        return -calculate_score(pts_2d)
        
    initial_flat = best_pts.flatten()
    # Use Nelder-Mead to polish
    try:
        res = minimize(objective, initial_flat, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        final_pts = res.x.reshape(-1, 2)
        # Normalize final points
        final_pts = final_pts - np.mean(final_pts, axis=0)
        norms = np.linalg.norm(final_pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 1e-10:
            final_pts = final_pts / max_norm
            
        final_score = calculate_score(final_pts)
        if final_score > best_score:
            best_pts = final_pts
            best_score = final_score
    except Exception:
        pass
        
    return best_pts
