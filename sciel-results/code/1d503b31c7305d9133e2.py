"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if points are collinear or degenerate
        return 0.0

def _score(points):
    """Score is min triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _objective(theta):
    """
    Objective function for optimization.
    theta: array of 13 angles in [0, 2*pi)
    We place points on the unit circle: (cos(theta_i), sin(theta_i))
    The convex hull of points on a circle is the polygon formed by the points
    in angular order. Its area is fixed for the unit circle? No, it depends on the points.
    Actually, for points on a unit circle, the convex hull area is the area of the
    cyclic polygon. But the score is min_triangle_area / hull_area.
    
    To maximize min_triangle_area / hull_area, since both scale with radius^2,
    the ratio is scale-invariant. So we can fix the radius to 1.
    
    However, the hull area for points on a circle is not constant; it depends on the
    angular distribution. But for a given set of angles, the hull is the convex hull
    of those points. Since they are on a circle, the convex hull is the polygon with
    vertices in angular order (assuming no three are collinear, which is true for
    distinct angles on a circle).
    
    The area of the convex hull of points on the unit circle with angles theta_0 <= ... <= theta_{n-1}
    (sorted) is 0.5 * sum_{i} sin(theta_{i+1} - theta_i) where indices are modulo n,
    but only if the points are in convex position, which they are.
    
    Actually, the area of the polygon with vertices (cos(theta_i), sin(theta_i)) in order
    is 0.5 * |sum_{i} (x_i y_{i+1} - x_{i+1} y_i)| = 0.5 * |sum_{i} sin(theta_{i+1} - theta_i)|.
    Since the points are on the unit circle and in counterclockwise order, this sum is positive.
    
    So hull_area = 0.5 * sum_{i=0}^{n-1} sin(d_i) where d_i = theta_{i+1} - theta_i (with theta_n = theta_0 + 2*pi).
    And sum d_i = 2*pi.
    
    The minimum triangle area: for three points on the unit circle at angles a, b, c (with a < b < c),
    the area is 0.5 * |sin(b-a) sin(c-b) sin(c-a)|? No, that's not right.
    
    Let's use the formula: area = 0.5 * |sin(b-a) + sin(c-b) + sin(a-c)|? 
    Actually, the area of a triangle inscribed in a unit circle with central angles between consecutive vertices
    being alpha, beta, gamma (alpha+beta+gamma=2*pi) is 0.5*(sin(alpha) + sin(beta) + sin(gamma)).
    Wait, let me verify. The area is 0.5 * |cross(p2-p1, p3-p1)|.
    If p1=(cos a, sin a), p2=(cos b, sin b), p3=(cos c, sin c),
    then area = 0.5 * |cos a(sin b - sin c) - sin a(cos b - cos c)|
    = 0.5 * |cos a sin b - cos a sin c - sin a cos b + sin a cos c|
    = 0.5 * |sin(b-a) - sin(c-a)|
    = 0.5 * |2 cos((b+c)/2 - a) sin((b-c)/2)|... this is getting messy.
    
    A simpler formula: the area of a triangle with vertices on the unit circle at angles a, b, c is
    0.5 * |sin(b-a) sin(c-b) sin(c-a)|? Let's test with a=0, b=2pi/3, c=4pi/3.
    sin(2pi/3)=sqrt(3)/2, sin(2pi/3)=sqrt(3)/2, sin(4pi/3)=-sqrt(3)/2.
    Product = (sqrt(3)/2)^2 * (-sqrt(3)/2) = -3/8. Absolute value 3/8, times 0.5 is 3/16.
    But the area of an equilateral triangle inscribed in a unit circle is (3*sqrt(3))/4 * (1/sqrt(3))^2?
    Side length s = sqrt(3), area = (sqrt(3)/4) * s^2 = (sqrt(3)/4)*3 = 3sqrt(3)/4 ≈ 1.299.
    3/16 = 0.1875. That's not matching.
    
    Let me recompute. For a=0, b=2pi/3, c=4pi/3:
    p1=(1,0), p2=(-0.5, sqrt(3)/2), p3=(-0.5, -sqrt(3)/2).
    v1 = p2-p1 = (-1.5, sqrt(3)/2), v2 = p3-p1 = (-1.5, -sqrt(3)/2).
    Cross = (-1.5)(-sqrt(3)/2) - (sqrt(3)/2)(-1.5) = 1.5*sqrt(3)/2 + 1.5*sqrt(3)/2 = 1.5*sqrt(3) = (3sqrt(3))/2.
    Area = 0.5 * |cross| = (3sqrt(3))/4 ≈ 1.299. Correct.
    
    So the formula 0.5*|sin(b-a) sin(c-b) sin(c-a)| gave 3/16, which is wrong.
    
    Correct formula: area = 0.5 * |sin(b-a) + sin(c-b) + sin(a-c)|? 
    sin(2pi/3)=sqrt(3)/2, sin(2pi/3)=sqrt(3)/2, sin(-4pi/3)=-sin(4pi/3)=sqrt(3)/2.
    Sum = 3sqrt(3)/2. Times 0.5 = 3sqrt(3)/4. Yes! That matches.
    
    So area = 0.5 * (sin(b-a) + sin(c-b) + sin(a-c)) for a < b < c with the angles taken such that
    the differences are the arcs. But sin(a-c) is negative if c-a > pi. In our case c-a = 4pi/3 > pi,
    so sin(a-c) = sin(-4pi/3) = sin(2pi/3) = sqrt(3)/2. It works out.
    
    Actually, the general formula for the area of a triangle with vertices on the unit circle at angles
    a, b, c (in counterclockwise order) is 0.5 * (sin(b-a) + sin(c-b) + sin(2pi - (c-a)))
    = 0.5 * (sin(b-a) + sin(c-b) - sin(c-a)) if c-a < pi, but if c-a > pi, it's different.
    
    To avoid confusion, I'll just compute the area directly from coordinates in the objective.
    """
    # Convert angles to points on unit circle
    pts = np.column_stack((np.cos(theta), np.sin(theta)))
    
    # Compute hull area
    # Since points are on a circle, sort by angle
    order = np.argsort(theta)
    sorted_theta = theta[order]
    sorted_pts = pts[order]
    
    # Hull area = 0.5 * sum sin(d_i) where d_i are angular gaps
    d = np.diff(np.concatenate([sorted_theta, [sorted_theta[0] + 2*np.pi]]))
    hull_area = 0.5 * np.sum(np.sin(d))
    
    if hull_area < 1e-12:
        return 1.0  # Minimize negative score, so return large value
    
    # Compute min triangle area
    n = len(pts)
    min_tri = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_tri:
                    min_tri = area
    
    score = min_tri / hull_area
    return -score  # We minimize negative score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_score = -1.0
    best_theta = None
    
    # Initial guess: equally spaced
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    # Try differential evolution for global search
    try:
        result = differential_evolution(
            _objective,
            bounds=[(0, 2*np.pi)] * N,
            seed=42,
            maxiter=50,
            tol=1e-6,
            polish=True
        )
        if time.time() > deadline:
            raise TimeoutError()
        
        if result.fun < 0:  # Valid result
            score = -result.fun
            if score > best_score:
                best_score = score
                best_theta = result.x
    except Exception:
        pass
    
    # Local refinement using L-BFGS-B
    if best_theta is not None:
        try:
            result2 = minimize(
                _objective,
                best_theta,
                method='L-BFGS-B',
                bounds=[(0, 2*np.pi)] * N,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if time.time() > deadline:
                raise TimeoutError()
            
            if result2.fun < 0:
                score = -result2.fun
                if score > best_score:
                    best_score = score
                    best_theta = result2.x
        except Exception:
            pass
    
    # If no good result, use equally spaced
    if best_theta is None:
        best_theta = theta0
    
    # Convert to points
    pts = np.column_stack((np.cos(best_theta), np.sin(best_theta)))
    
    return pts
# EVOLVE-BLOCK-END
