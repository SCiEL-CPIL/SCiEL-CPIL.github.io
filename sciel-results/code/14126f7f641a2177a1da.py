import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of points on the unit sphere.
    points: (N, 3) array of vectors.
    Returns the covering radius in radians.
    """
    # Normalize points to ensure they are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Handle zero norms (should not happen if initialized properly, but for safety)
    norms[norms == 0] = 1.0
    p = points / norms
    
    # Check for coincident points (distance < 1e-8)
    # This might be slow for large N, but N=50 is small.
    # We can skip explicit check if we trust the optimizer, but for validity check it's good.
    # For speed in the objective function, we might skip the full O(N^2) check and rely on the hull.
    # If points are coincident, the hull will be degenerate or have 0 area facets.
    
    try:
        hull = scipy.spatial.ConvexHull(p)
    except Exception:
        # If hull fails (e.g., points in a hemisphere or degenerate), return a large penalty
        # The problem states configurations inside a closed hemisphere are invalid.
        # A convex hull of points in a hemisphere will have facets that don't "wrap" the sphere properly?
        # Actually, if points are in a hemisphere, the origin is not in the convex hull.
        # The "covering radius" concept implies the points cover the whole sphere.
        # If they don't, the max distance is > 90 degrees.
        # We should penalize this heavily.
        # A simple check: is the origin inside the convex hull?
        # Or just return a large value.
        return np.pi / 2 + 0.1 # ~100 degrees, penalty
        
    max_rad = 0.0
    for simplex in hull.simplices:
        # simplex contains indices of 3 vertices for a triangle
        i, j, k = simplex
        v1 = p[i]
        v2 = p[j]
        v3 = p[k]
        
        # Compute circumcenter of the planar triangle v1, v2, v3
        # The circumcenter C satisfies |C-v1|^2 = |C-v2|^2 = |C-v3|^2
        # 2*C.(v2-v1) = |v2|^2 - |v1|^2 + 2*v1.(v2-v1) ? No.
        # |C|^2 - 2*C.v1 + |v1|^2 = |C|^2 - 2*C.v2 + |v2|^2
        # 2*C.(v2-v1) = |v2|^2 - |v1|^2
        # Since |v1|=|v2|=|v3|=1, RHS is 0.
        # So C.(v2-v1) = 0 and C.(v3-v1) = 0.
        # This means C is parallel to the cross product (v2-v1) x (v3-v1).
        # Wait, this is only true if the triangle is not degenerate and we are looking for the center of the circle passing through them.
        # The circumcenter of a triangle inscribed in the unit sphere lies on the normal to the plane.
        # Let n = (v2-v1) x (v3-v1).
        # The circumcenter is C = alpha * n.
        # We need to find alpha.
        # |C - v1|^2 = R_planar^2.
        # Also, the distance from C to the plane is 0? No, C is IN the plane.
        # So C = v1 + u*(v2-v1) + v*(v3-v1).
        # And C is equidistant from v1, v2, v3.
        
        # Let's solve for C in the plane.
        # Let a = v2 - v1, b = v3 - v1.
        # C = v1 + x*a + y*b.
        # |C-v1|^2 = |C-v2|^2 => |x*a + y*b|^2 = |(x-1)*a + y*b|^2
        # |x*a + y*b|^2 = (x-1)^2|a|^2 + y^2|b|^2 + 2y(x-1)a.b
        # x^2|a|^2 + y^2|b|^2 + 2xy a.b = x^2|a|^2 - 2x|a|^2 + |a|^2 + y^2|b|^2 + 2y(x-1)a.b
        # 0 = -2x|a|^2 + |a|^2 - 2y a.b
        # 2x|a|^2 + 2y a.b = |a|^2
        # Similarly for v3:
        # 2x a.b + 2y|b|^2 = |b|^2
        
        # System:
        # |a|^2 x + a.b y = 0.5 |a|^2
        # a.b x + |b|^2 y = 0.5 |b|^2
        
        a = v2 - v1
        b = v3 - v1
        aa = np.dot(a, a)
        bb = np.dot(b, b)
        ab = np.dot(a, b)
        
        det = aa * bb - ab * ab
        if np.abs(det) < 1e-10:
            # Degenerate triangle, skip or penalty
            continue
            
        x = (0.5 * aa * bb - 0.5 * ab * ab) / det # 0.5 * (aa*bb - ab^2) / det = 0.5 * det / det = 0.5?
        # Wait.
        # x = (0.5*aa*bb - 0.5*ab*ab) / det = 0.5 * (aa*bb - ab^2) / det = 0.5
        # y = (0.5*bb*aa - 0.5*ab*ab) / det = 0.5
        # This implies C = v1 + 0.5*(v2-v1) + 0.5*(v3-v1) = 0.5*(v2+v3).
        # This is the midpoint of the hypotenuse?
        # This logic holds only if the triangle is right-angled?
        # Let's re-derive.
        # 2*C.(v2-v1) = |v2|^2 - |v1|^2 = 0.
        # This equation $C \cdot (v_2 - v_1) = 0$ is correct for the circumcenter of a triangle inscribed in a sphere centered at the origin?
        # No. The circumcenter $C$ of the *planar* triangle is not necessarily related to the origin this way.
        # The condition $|C-v_1|^2 = |C-v_2|^2$ implies $C \cdot (v_2 - v_1) = \frac{1}{2}(|v_2|^2 - |v_1|^2)$.
        # Since $|v_1|=|v_2|=1$, RHS is 0.
        # So $C$ lies on the plane perpendicular to $v_2-v_1$ passing through the origin?
        # No, $C \cdot (v_2 - v_1) = 0$ means the vector $C$ is orthogonal to $v_2-v_1$.
        # This means $C$ lies on the plane passing through the origin with normal $v_2-v_1$.
        # Similarly $C$ lies on the plane passing through the origin with normal $v_3-v_1$.
        # The intersection of these two planes is the line passing through the origin and the circumcenter of the triangle?
        # Yes! The circumcenter of a triangle inscribed in the unit sphere lies on the line passing through the origin and perpendicular to the plane of the triangle.
        # So $C = \alpha n$, where $n = (v_2-v_1) \times (v_3-v_1)$.
        # We need to find $\alpha$.
        # $C$ must be in the plane of the triangle.
        # The plane equation is $n \cdot x = d$.
        # $d = n \cdot v_1$.
        # So $n \cdot C = d \implies n \cdot (\alpha n) = d \implies \alpha |n|^2 = d$.
        # $\alpha = d / |n|^2$.
        # So $C = \frac{n \cdot v_1}{|n|^2} n$.
        
        n = np.cross(a, b)
        n_sq = np.dot(n, n)
        if n_sq < 1e-10:
            continue
        d = np.dot(n, v1)
        alpha = d / n_sq
        C = alpha * n
        
        # C is the circumcenter in 3D space.
        # The circumcenter on the sphere is u = C / |C|.
        C_norm = np.linalg.norm(C)
        if C_norm < 1e-10:
            continue
        u = C / C_norm
        
        # The angular distance is arccos(u . v1)
        # Clamp dot product to [-1, 1]
        cos_theta = np.dot(u, v1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta > max_rad:
            max_rad = theta
            
    return max_rad

def _fibonacci_points(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(N, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(N, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _icosahedron_based_points(N):
    # Start with icosahedron and add random points, then optimize?
    # Or just use random.
    return _random_points(N, seed=42)

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    best_points = _fibonacci_points(N)
    best_radius = _covering_radius(best_points)
    
    # Generate diverse initializations
    inits = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _random_points(N, seed=4),
        _random_points(N, seed=5),
    ]
    
    # Add a "spiral" variant with different golden ratio?
    # Just stick to random + fibonacci for now.
    
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        # Optimize
        # Flatten points to 1D array for optimizer
        x0 = init.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            return _covering_radius(p)
            
        # Nelder-Mead settings
        # maxiter should be sufficient to converge
        # fatol, xatol for precision
        try:
            res = scipy.optimize.minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if res.success or res.fun < best_radius:
                best_points = res.x.reshape(N, 3)
                best_radius = res.fun
        except Exception:
            pass
            
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
