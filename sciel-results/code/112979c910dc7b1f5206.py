import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of points on the unit sphere.
    points: (N, 3) array of unit vectors.
    """
    # Ensure points are normalized
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull computation fails (e.g., points are collinear or in a hemisphere in a degenerate way)
        # Return a large penalty
        return np.pi

    # For each facet, compute the circumradius
    max_radius = 0.0
    
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex]
        
        # Compute side lengths (angular distances)
        # cos(a) = v2 . v3, etc.
        cos_a = np.clip(np.dot(v2, v3), -1.0, 1.0)
        cos_b = np.clip(np.dot(v1, v3), -1.0, 1.0)
        cos_c = np.clip(np.dot(v1, v2), -1.0, 1.0)
        
        a = np.arccos(cos_a)
        b = np.arccos(cos_b)
        c = np.arccos(cos_c)
        
        # Determine if the triangle is obtuse
        # A triangle is obtuse if one angle is > 90 degrees.
        # In a spherical triangle, the sum of angles is > 180.
        # We can check if the circumcenter is inside.
        # A simpler check: If the longest side squared > sum of squares of other two? (Flat approximation)
        # Better: Compute the circumcenter and check if it's inside the triangle.
        
        # Circumcenter calculation:
        # The circumcenter K is the point on the sphere equidistant from v1, v2, v3.
        # K = (v1 + v2 + v3) / |v1 + v2 + v3| if the triangle is acute.
        # If the triangle is obtuse, the "circumcenter" of the smallest enclosing cap is the midpoint of the longest side.
        
        # Let's find the longest side
        sides = [a, b, c]
        max_side_idx = np.argmax(sides)
        max_side = sides[max_side_idx]
        
        # Check for obtuse:
        # If max_side > np.pi/2, it's definitely not acute in the sense of circumcenter inside?
        # Actually, a spherical triangle is acute if all angles < 90.
        # If any angle >= 90, the circumcenter is outside or on the boundary.
        # For covering radius, if the triangle is obtuse, the covering radius is half the longest side.
        
        # How to check if obtuse?
        # Compute the angles using the spherical law of cosines?
        # Or check the position of the circumcenter.
        
        # Let's try to compute the circumcenter K.
        # K is in the plane of v1, v2, v3.
        # Normal n = cross(v1, v2).
        # K = v1 + alpha * (v2 - v1) + beta * (n normalized? no, in-plane vector)
        
        # Easier method:
        # The circumcenter is the intersection of the perpendicular bisectors.
        # In 3D, we can solve the linear system:
        # (v1 - v2) . K = (|v1|^2 - |v2|^2) / 2 = 0
        # (v1 - v3) . K = 0
        # K . K = 1
        # This is non-linear.
        
        # Alternative:
        # If the triangle is acute, K = normalize(v1 + v2 + v3).
        # Let's test this.
        K_candidate = v1 + v2 + v3
        norm_K = np.linalg.norm(K_candidate)
        if norm_K < 1e-10:
            # Degenerate
            radius = max_side / 2.0
        else:
            K = K_candidate / norm_K
            
            # Check if K is inside the triangle.
            # K is inside if the barycentric coordinates are all positive.
            # Or, check if the angle between (K - v1) and (v2 - v1) is < 90?
            # A robust check: The circumcenter is inside if the triangle is acute.
            # We can check if the dot product of the normal to the triangle with (K - v_i) is consistent?
            
            # Let's use the angle check.
            # Compute angles A, B, C.
            # cos A = (cos a - cos b cos c) / (sin b sin c)
            sin_b = np.sqrt(1 - cos_b**2)
            sin_c = np.sqrt(1 - cos_c**2)
            sin_a = np.sqrt(1 - cos_a**2)
            
            if sin_b < 1e-10 or sin_c < 1e-10:
                 # Degenerate
                 radius = max_side / 2.0
            else:
                cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
                cos_A = np.clip(cos_A, -1.0, 1.0)
                A = np.arccos(cos_A)
                
                if A >= np.pi / 2:
                    # Obtuse at v1? No, A is angle at v1?
                    # Standard notation: side a is opposite angle A (at v1).
                    # So if A >= 90, the triangle is obtuse at v1.
                    # The covering radius is half the longest side.
                    radius = max_side / 2.0
                else:
                    # Check other angles? If one is obtuse, the circumcenter is outside.
                    # We only need to know if ANY angle is >= 90.
                    # If all are < 90, use circumradius R.
                    # R = arccos(K . v1)
                    R = np.arccos(np.clip(np.dot(K, v1), -1.0, 1.0))
                    radius = R
                    
                    # But wait, if another angle is obtuse, this R is wrong for covering.
                    # So we must check all angles.
                    # Let's just check if the triangle is acute.
                    # If not acute, radius = max_side / 2.
                    
                    # Compute all angles
                    cos_B = (cos_b - cos_a * cos_c) / (sin_a * sin_c)
                    cos_B = np.clip(cos_B, -1.0, 1.0)
                    B = np.arccos(cos_B)
                    
                    cos_C = (cos_c - cos_a * cos_b) / (sin_a * sin_b)
                    cos_C = np.clip(cos_C, -1.0, 1.0)
                    C = np.arccos(cos_C)
                    
                    if A < np.pi/2 and B < np.pi/2 and C < np.pi/2:
                        # Acute
                        radius = R
                    else:
                        # Obtuse
                        radius = max_side / 2.0
                        
        if radius > max_radius:
            max_radius = radius
            
    return max_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib_points)
    
    # 2. Random points
    for _ in range(5):
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1)[:, np.newaxis]
        inits.append(rand_points)
        
    # Evaluate initial configurations
    for p in inits:
        r = _covering_radius(p)
        if r < best_radius:
            best_radius = r
            best_points = p.copy()
            
    # Local Optimization
    # We will use Nelder-Mead on the flattened array of points.
    # We need to project back to sphere after each step.
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1)
        # Avoid division by zero
        norms[norms < 1e-10] = 1.0
        pts = pts / norms[:, np.newaxis]
        return _covering_radius(pts)
        
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        # Run Nelder-Mead
        # maxiter is a safety net, but we rely on time
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if time.time() > deadline:
            break
            
        if res.success or res.fun < best_radius:
            pts = res.x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms[:, np.newaxis]
            
            r = _covering_radius(pts)
            if r < best_radius:
                best_radius = r
                best_points = pts.copy()
                
    # Final check
    if best_points is None:
        return fib_points
        
    return best_points

construct()
