import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
    except Exception:
        # Fallback: use random sampling if hull fails
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for f in facets:
        a = points[f[0]]
        b = points[f[1]]
        c = points[f[2]]
        
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        m = n / norm_n
        if np.dot(m, a) < 0:
            m = -m
        r = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
        if r > max_r:
            max_r = r
    return max_r

def construct():
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s for final polish
    
    alpha = 0.05
    decay = 0.995
    
    while time.time() - start_time < time_limit:
        # Compute Convex Hull
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
        except Exception:
            points += np.random.normal(0, 0.01, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        # Calculate circumradius for each facet
        radii = np.zeros(len(facets))
        centers = np.zeros((len(facets), 3))
        
        for idx, f in enumerate(facets):
            a = points[f[0]]
            b = points[f[1]]
            c = points[f[2]]
            
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                radii[idx] = 0
                continue
            m = n / norm_n
            if np.dot(m, a) < 0:
                m = -m
            radii[idx] = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
            centers[idx] = m
            
        current_radius = np.max(radii)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        threshold = current_radius - 0.05
        worst_indices = np.where(radii > threshold)[0]
        
        if len(worst_indices) == 0:
            worst_indices = np.array([np.argmax(radii)])
            
        updates = np.zeros_like(points)
        
        for idx in worst_indices:
            c = centers[idx]
            f = facets[idx]
            for k in f:
                updates[k] += c - points[k]
                
        if np.any(updates != 0):
            points += alpha * updates
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
        alpha *= decay

    # Final Polish: Use scipy minimize on the points
    if time.time() - start_time < 118.0:
        # Flatten points for optimizer
        x0 = best_points.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            return _covering_radius(p)
            
        # Use Nelder-Mead as it doesn't need gradients and is robust to non-smoothness
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success or res.fun < best_radius:
            final_points = res.x.reshape(N, 3)
            final_points /= np.linalg.norm(final_points, axis=1, keepdims=True)
            final_radius = _covering_radius(final_points)
            if final_radius < best_radius:
                best_points = final_points
                best_radius = final_radius

    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
