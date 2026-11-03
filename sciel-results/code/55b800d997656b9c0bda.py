import numpy as np
import scipy.spatial
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    points += 0.01 * rng.normal(size=(N, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = np.pi # Initialize to max possible
    
    def calc_radius(pts):
        try:
            hull = scipy.spatial.ConvexHull(pts)
            max_r = 0.0
            worst_simplex = None
            worst_center = None
            for simplex in hull.simplices:
                A, B, C = pts[simplex]
                # Normal to the plane
                N_vec = np.cross(B - A, C - A)
                norm = np.linalg.norm(N_vec)
                if norm < 1e-9:
                    continue
                O = N_vec / norm
                # Ensure O points "outward" (same side as centroid)
                centroid = (A + B + C) / 3.0
                if np.dot(O, centroid) < 0:
                    O = -O
                # Distance is arccos(dot(O, A))
                # Clip to avoid numerical errors
                d = np.dot(O, A)
                d = np.clip(d, -1.0, 1.0)
                r = np.arccos(d)
                if r > max_r:
                    max_r = r
                    worst_simplex = simplex
                    worst_center = O
            return max_r, worst_simplex, worst_center
        except Exception:
            return np.pi, None, None

    # Initial radius
    best_radius, _, _ = calc_radius(points)
    
    alpha = 0.05
    min_alpha = 0.001
    
    while time.time() < deadline:
        current_radius, worst_simplex, worst_center = calc_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        if worst_simplex is None:
            break
            
        # Move the vertices of the worst facet towards the circumcenter
        A_idx, B_idx, C_idx = worst_simplex
        A, B, C = points[A_idx], points[B_idx], points[C_idx]
        
        # Update points
        for idx in [A_idx, B_idx, C_idx]:
            P = points[idx]
            P_new = P + alpha * (worst_center - P)
            norm = np.linalg.norm(P_new)
            if norm > 1e-9:
                points[idx] = P_new / norm
            else:
                # If it collapsed, perturb
                points[idx] = P + 0.01 * rng.normal(size=3)
                points[idx] /= np.linalg.norm(points[idx])
        
        # Check for coincident points and perturb if necessary
        # Simple check: min distance
        # For efficiency, we can skip this every few iterations or use a cheap check
        # Let's do a cheap check: if any point moved too close to another
        
        # Cooling
        if alpha > min_alpha:
            alpha *= 0.995
            
    return best_points
