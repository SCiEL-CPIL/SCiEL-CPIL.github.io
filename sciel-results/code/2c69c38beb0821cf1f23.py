import numpy as np
from scipy.spatial import ConvexHull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        try:
            hull = ConvexHull(pts)
            a_hull = hull.volume
        except Exception:
            return 0.0
        if a_hull < 1e-10:
            return 0.0
        
        min_area = float('inf')
        # Vectorized calculation of triangle areas is possible but loop is fine for n=13 (286 triangles)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
        if min_area < 1e-10:
            return 0.0
        return min_area / a_hull

    best_score = 0.0
    best_points = None
    
    # Try multiple initial configurations
    rng = np.random.default_rng(seed=42)
    
    for restart in range(5):
        if restart == 0:
            # Regular polygon
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            # Random points in unit circle, then projected to hull? No, just random.
            # Or regular polygon with random jitter
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.1, 0.1, n)
            radii = 1.0 + rng.uniform(-0.1, 0.1, n)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        current_score = get_score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Local search
        step_size = 0.05
        no_improve = 0
        max_iter = 500
        
        for _ in range(max_iter):
            if no_improve > 50:
                break
            idx = rng.integers(n)
            orig = points[idx].copy()
            points[idx] += rng.normal(0, step_size, 2)
            
            new_score = get_score(points)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                no_improve = 0
            else:
                points[idx] = orig
                no_improve += 1
                
    return best_points
