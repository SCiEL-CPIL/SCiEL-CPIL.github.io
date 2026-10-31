"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# Vertices of the equilateral triangle with area 1
# Side length s such that (sqrt(3)/4) * s^2 = 1 => s = 2 / 3^(1/4)
S = 2.0 / (3.0 ** 0.25)
H = S * np.sqrt(3) / 2.0  # This should be 1.0 since area = (1/2)*S*H = 1 => H = 2/S = 3^(1/4)
# Actually: area = (1/2) * base * height = (1/2) * S * H = 1 => H = 2/S
# And H = S * sqrt(3)/2, so S * sqrt(3)/2 = 2/S => S^2 * sqrt(3) = 4 => S^2 = 4/sqrt(3) => S = 2/3^(1/4)
# H = 2/S = 3^(1/4)
# Let's verify: area = (1/2) * (2/3^(1/4)) * (3^(1/4)) = (1/2) * 2 = 1. Correct.

V0 = np.array([0.0, 0.0])
V1 = np.array([S, 0.0])
V2 = np.array([S / 2.0, H])

# For the evaluator, the triangle is (0,0), (1,0), (1/2, sqrt(3)/2) which has area sqrt(3)/4
# But the task says "unit-area triangle" and "divides by its area"
# So we work in the unit-area triangle and the evaluator will handle normalization
# Actually, re-reading: "Place 11 points on or inside a unit-area triangle"
# The interface says: "n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
# That triangle has area sqrt(3)/4, not 1.
# "The notebook uses an equilateral triangle and divides by its area."
# So I think we should place points in the triangle (0,0), (1,0), (0.5, sqrt(3)/2)
# and the evaluator computes min area / (sqrt(3)/4)
# OR we place points in a unit-area triangle and the evaluator just takes the raw min area.
# Let me re-read: "maximize the smallest area of a triangle formed by three points"
# "The evaluator measures the raw value (higher is better)"
# "construct() -> array of shape (n, 2): n points inside the equilateral triangle (0,0), (1,0), (1/2, sqrt(3)/2)"
# So the points must be in that specific triangle. The raw value is the min area.
# The triangle (0,0), (1,0), (0.5, sqrt(3)/2) has area sqrt(3)/4 ≈ 0.433.

# Let me work in the triangle T = conv{(0,0), (1,0), (0.5, sqrt(3)/2)}
T0 = np.array([0.0, 0.0])
T1 = np.array([1.0, 0.0])
T2 = np.array([0.5, np.sqrt(3) / 2.0])

def point_in_triangle(p, v0, v1, v2, eps=1e-9):
    """Check if point p is inside or on the boundary of triangle v0, v1, v2."""
    # Barycentric coordinates
    x, y = p
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-15:
        return False
    
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    
    return a >= -eps and b >= -eps and c >= -eps

def clip_to_triangle(p, v0, v1, v2):
    """Clip point p to the triangle v0, v1, v2."""
    # Use barycentric coordinates and clip
    x, y = p
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-15:
        return p
    
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    
    # Clip barycentric coords to [0, 1]
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    c = np.clip(c, 0, 1)
    
    # Renormalize
    s = a + b + c
    if s < 1e-15:
        return p
    a /= s
    b /= s
    c /= s
    
    return a * v0 + b * v1 + c * v2

def triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the given points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area(params):
    """Objective function for optimization: minimize -min_area."""
    points = np.array(params).reshape(N, 2)
    # Clip to triangle
    points = np.array([clip_to_triangle(p, T0, T1, T2) for p in points])
    # Check if all points are in triangle
    # (clip should handle this, but just in case)
    return -min_triangle_area(points)

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    
    best_points = None
    best_area = 0.0
    
    # Initial diverse starting points
    rng = np.random.default_rng(42)
    
    # Try several random initializations with local optimization
    for trial in range(50):
        if time.time() > deadline - 20.0:
            break
        
        # Random initialization inside triangle
        # Use barycentric coordinates for uniform sampling
        u1 = rng.random()
        u2 = rng.random()
        if u1 + u2 > 1:
            u1, u2 = 1 - u1, 1 - u2
        # Barycentric: a, b, c with a+b+c=1
        a = u1
        b = u2
        c = 1 - a - b
        
        # Add some structure: mix of random and quasi-random
        points = np.zeros((N, 2))
        for i in range(N):
            if i == 0:
                # Corner-ish
                points[i] = T0 + 0.1 * (T1 - T0)
            elif i == 1:
                points[i] = T0 + 0.1 * (T2 - T0)
            elif i == 2:
                points[i] = T1 + 0.1 * (T0 - T1)
            elif i == 3:
                points[i] = T2 + 0.1 * (T0 - T2)
            elif i == 4:
                points[i] = T2 + 0.1 * (T1 - T2)
            else:
                # Random interior
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1:
                    r1, r2 = 1 - r1, 1 - r2
                pa = r1
                pb = r2
                pc = 1 - pa - pb
                points[i] = pa * T0 + pb * T1 + pc * T2
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                negative_min_area,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_points = np.array(result.x).reshape(N, 2)
            opt_points = np.array([clip_to_triangle(p, T0, T1, T2) for p in opt_points])
            area = min_triangle_area(opt_points)
            
            if area > best_area:
                best_area = area
                best_points = opt_points.copy()
        except Exception:
            pass
    
    # Try differential evolution for global search
    if time.time() < deadline - 5.0 and best_points is not None:
        try:
            # Define bounds for each coordinate
            # Triangle bounds: x in [0, 1], y in [0, sqrt(3)/2]
            # But we need to respect triangle constraint, so use looser bounds
            bounds = [(0.0, 1.0)] * N + [(0.0, np.sqrt(3) / 2.0)] * N
            
            # Reorder: all x coords then all y coords
            def de_objective(params):
                points = np.zeros((N, 2))
                for i in range(N):
                    points[i] = [params[i], params[N + i]]
                points = np.array([clip_to_triangle(p, T0, T1, T2) for p in points])
                return -min_triangle_area(points)
            
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=100,
                popsize=15,
                seed=42,
                tol=1e-8,
                polish=False
            )
            
            de_points = np.zeros((N, 2))
            for i in range(N):
                de_points[i] = [de_result.x[i], de_result.x[N + i]]
            de_points = np.array([clip_to_triangle(p, T0, T1, T2) for p in de_points])
            de_area = min_triangle_area(de_points)
            
            if de_area > best_area:
                best_area = de_area
                best_points = de_points.copy()
                
                # Polish with Nelder-Mead
                if time.time() < deadline - 2.0:
                    try:
                        result = minimize(
                            negative_min_area,
                            de_points.flatten(),
                            method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                        )
                        polished = np.array(result.x).reshape(N, 2)
                        polished = np.array([clip_to_triangle(p, T0, T1, T2) for p in polished])
                        pol_area = min_triangle_area(polished)
                        if pol_area > best_area:
                            best_area = pol_area
                            best_points = polished.copy()
                    except Exception:
                        pass
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: simple random
        points = np.zeros((N, 2))
        for i in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            points[i] = r1 * T0 + r2 * T1 + (1 - r1 - r2) * T2
        return points
    
    return best_points
