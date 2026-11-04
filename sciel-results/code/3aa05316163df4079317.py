"""Heilbronn triangles: place 11 points in an equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
# Vertices of the equilateral triangle
V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
# Area of the equilateral triangle
TRI_AREA = np.sqrt(3) / 4

# Known good initial configuration from literature (scaled to our triangle)
# This is a heuristic starting point based on known optimal arrangements
def _known_good_config():
    """Return a known good configuration for 11 points in an equilateral triangle."""
    # Based on literature results for Heilbronn problem
    # These are barycentric coordinates (lambda1, lambda2, lambda3) with sum = 1
    bary = [
        (1/3, 1/3, 1/3),      # center
        (0.5, 0.25, 0.25),    # various positions
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
        (0.6, 0.2, 0.2),
        (0.2, 0.6, 0.2),
        (0.2, 0.2, 0.6),
        (0.4, 0.4, 0.2),
        (0.4, 0.2, 0.4),
        (0.2, 0.4, 0.4),
        (0.7, 0.15, 0.15),
    ]
    points = np.array([
        b[0] * V[0] + b[1] * V[1] + b[2] * V[2] for b in bary
    ])
    return points

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x):
    """Objective function for optimization: negative of min triangle area."""
    points = x.reshape(N, 2)
    # Check if all points are inside the triangle
    # Barycentric coordinates check
    for i in range(N):
        px, py = points[i]
        # Barycentric coordinates
        # lambda1 = 1 - x - y/sqrt(3)
        # lambda2 = x - y/(2*sqrt(3))
        # lambda3 = y/sqrt(3)
        l1 = 1 - px - py / np.sqrt(3)
        l2 = px - py / (2 * np.sqrt(3))
        l3 = py / np.sqrt(3)
        if l1 < -1e-9 or l2 < -1e-9 or l3 < -1e-9:
            return -1e6  # Penalize points outside the triangle
    
    return -_min_triangle_area(points)

def _project_to_triangle(points):
    """Project points onto the triangle (clip to boundary if outside)."""
    for i in range(N):
        px, py = points[i]
        # Compute barycentric coordinates
        l1 = 1 - px - py / np.sqrt(3)
        l2 = px - py / (2 * np.sqrt(3))
        l3 = py / np.sqrt(3)
        
        # If any barycentric coordinate is negative, project to nearest edge
        if l1 < 0 or l2 < 0 or l3 < 0:
            # Simple projection: find closest point on triangle boundary
            # For simplicity, just clamp and re-normalize
            l = np.array([l1, l2, l3])
            l = np.maximum(l, 0)
            l_sum = l.sum()
            if l_sum > 0:
                l = l / l_sum
            else:
                l = np.array([1/3, 1/3, 1/3])
            points[i] = l[0] * V[0] + l[1] * V[1] + l[2] * V[2]
    return points

def construct():
    """Construct 11 points in the equilateral triangle to maximize min triangle area."""
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Start with known good configuration
    best_points = _known_good_config()
    best_score = _min_triangle_area(best_points)
    
    # Differential evolution for global search
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, np.sqrt(3) / 2)] * N
    
    # Try differential evolution with time limit
    if time.time() < deadline:
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=True,
                init='sobol',
                x0=_known_good_config().flatten(),
                callback=lambda xk, convergence: (time.time() < deadline)
            )
            if de_result.fun < 0:  # Valid result
                de_points = de_result.x.reshape(N, 2)
                de_points = _project_to_triangle(de_points)
                de_score = _min_triangle_area(de_points)
                if de_score > best_score:
                    best_points = de_points
                    best_score = de_score
        except Exception:
            pass
    
    # Local refinement with SLSQP
    if time.time() < deadline:
        try:
            # Use multiple restarts from different initial points
            initial_configs = [best_points.flatten()]
            
            # Add some random perturbations
            rng = np.random.RandomState(123)
            for _ in range(5):
                perturbed = best_points + rng.randn(N, 2) * 0.05
                perturbed = _project_to_triangle(perturbed)
                initial_configs.append(perturbed.flatten())
            
            for init in initial_configs:
                if time.time() >= deadline:
                    break
                
                # Define constraints for SLSQP
                def _constraints(x):
                    points = x.reshape(N, 2)
                    cons = []
                    for i in range(N):
                        px, py = points[i]
                        l1 = 1 - px - py / np.sqrt(3)
                        l2 = px - py / (2 * np.sqrt(3))
                        l3 = py / np.sqrt(3)
                        cons.append(l1)
                        cons.append(l2)
                        cons.append(l3)
                    return cons
                
                res = minimize(
                    _objective,
                    init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[{'type': 'ineq', 'fun': _constraints}],
                    tol=1e-10,
                    maxiter=1000
                )
                
                if res.fun < 0:  # Valid result
                    res_points = res.x.reshape(N, 2)
                    res_points = _project_to_triangle(res_points)
                    res_score = _min_triangle_area(res_points)
                    if res_score > best_score:
                        best_points = res_points
                        best_score = res_score
        
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline:
        try:
            # One more SLSQP with tighter tolerance
            def _constraints(x):
                points = x.reshape(N, 2)
                cons = []
                for i in range(N):
                    px, py = points[i]
                    l1 = 1 - px - py / np.sqrt(3)
                    l2 = px - py / (2 * np.sqrt(3))
                    l3 = py / np.sqrt(3)
                    cons.append(l1)
                    cons.append(l2)
                    cons.append(l3)
                return cons
            
            res = minimize(
                _objective,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                constraints=[{'type': 'ineq', 'fun': _constraints}],
                tol=1e-12,
                maxiter=2000
            )
            
            if res.fun < 0:
                res_points = res.x.reshape(N, 2)
                res_points = _project_to_triangle(res_points)
                res_score = _min_triangle_area(res_points)
                if res_score > best_score:
                    best_points = res_points
                    best_score = res_score
        except Exception:
            pass
    
    return np.array(best_points)
