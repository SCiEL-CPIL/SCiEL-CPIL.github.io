import numpy as np
from scipy.spatial import ConvexHull
import time

def construct():
    N = 130
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def calc_covering_radius(points):
        """Calculate the covering radius (in radians) using ConvexHull."""
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            max_radius = 0.0
            for k in range(len(facets)):
                idx = facets[k]
                A = points[idx[0]]
                B = points[idx[1]]
                C = points[idx[2]]
                
                # Calculate circumcenter of the spherical triangle
                # O is parallel to (B-A) x (C-B)
                AB = B - A
                BC = C - B
                V = np.cross(AB, BC)
                norm_V = np.linalg.norm(V)
                if norm_V < 1e-10:
                    continue
                O = V / norm_V
                centroid = (A + B + C) / 3.0
                if np.dot(O, centroid) < 0:
                    O = -O
                
                cos_theta = np.dot(O, A)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                radius = np.arccos(cos_theta)
                if radius > max_radius:
                    max_radius = radius
            return max_radius
        except Exception:
            return np.pi # Invalid hull, treat as very bad

    # 2. Simulated Annealing Parameters
    T_start = 0.01 # Initial temperature (radians)
    T_end = 0.0001
    num_iters = 5000
    # We will adjust num_iters based on time if needed, but 5000 hull calcs should be fast.
    
    current_radius = calc_covering_radius(pts)
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    for it in range(num_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Temperature decay
        if it < num_iters - 1:
            T = T_start * (T_end / T_start) ** (it / (num_iters - 1))
        else:
            T = T_end
            
        # Pick a random point
        i = np.random.randint(N)
        
        # Save original point
        orig_pt = pts[i].copy()
        
        # Generate random move
        # Step size can be adaptive or fixed. Let's use a small random step.
        step_size = 0.1 * (1.0 - 0.5 * (it / num_iters)) # Decay step size
        move = np.random.randn(3) * step_size
        
        # Update point
        pts[i] = pts[i] + move
        pts[i] = pts[i] / np.linalg.norm(pts[i])
        
        # Calculate new radius
        new_radius = calc_covering_radius(pts)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_radius = new_radius
        else:
            # Revert
            pts[i] = orig_pt
            
    return pts
