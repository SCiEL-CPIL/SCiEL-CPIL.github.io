import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _calc_covering_radius(pts):
    """Calculate the covering radius from the convex hull facets."""
    try:
        hull = ConvexHull(pts)
        simplices = hull.simplices
        max_dist = 0.0
        for s in simplices:
            v1, v2, v3 = pts[s[0]], pts[s[1]], pts[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if np.dot(n, v1) < 0:
                n = -n
            dot_val = np.dot(n, v1)
            if abs(dot_val) < 1e-10:
                continue
            k = 1.0 / (2.0 * dot_val)
            c = k * n
            c = c / np.linalg.norm(c)
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if dist > max_dist:
                max_dist = dist
        return max_dist
    except Exception:
        return np.inf

def _compute_facet_data(points):
    """Compute circumcenters and distances for all facets."""
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        circumcenters = []
        distances = []
        facet_indices = []
        for s in simplices:
            v1, v2, v3 = points[s[0]], points[s[1]], points[s[2]]
            n = np.cross(v2 - v1, v3 - v1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            if np.dot(n, v1) < 0:
                n = -n
            dot_val = np.dot(n, v1)
            if abs(dot_val) < 1e-10:
                continue
            k = 1.0 / (2.0 * dot_val)
            c = k * n
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            dist = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            circumcenters.append(c)
            distances.append(dist)
            facet_indices.append(s)
        return np.array(circumcenters), np.array(distances), facet_indices
    except Exception:
        return np.array([]), np.array([]), []

def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _compute_targets(points):
    """Compute target positions (average circumcenters) for each point."""
    circumcenters, distances, facet_indices = _compute_facet_data(points)
    if len(circumcenters) == 0:
        return points.copy()
    
    targets = np.zeros_like(points)
    counts = np.zeros(N)
    
    for i, s in enumerate(facet_indices):
        c = circumcenters[i]
        targets[s[0]] += c
        targets[s[1]] += c
        targets[s[2]] += c
        counts[s[0]] += 1
        counts[s[1]] += 1
        counts[s[2]] += 1
    
    for j in range(N):
        if counts[j] > 0:
            targets[j] /= counts[j]
        else:
            targets[j] = points[j]
    
    return targets

def _optimize(points, start_time, time_limit, rng):
    """Run local optimization from initial points with adaptive strategy."""
    points = points.copy()
    best_points = points.copy()
    best_radius = _calc_covering_radius(points)
    
    momentum = np.zeros_like(points)
    lr = 0.1
    momentum_factor = 0.6
    iterations = 0
    max_iters = 20000
    no_improve_count = 0
    patience = 300
    
    while time.time() - start_time < time_limit and iterations < max_iters:
        iterations += 1
        
        try:
            circumcenters, distances, facet_indices = _compute_facet_data(points)
            if len(circumcenters) == 0:
                break
            
            # Compute weighted targets: weight by facet distance (worse facets get more weight)
            # This focuses optimization on the worst facets
            max_d = np.max(distances) if len(distances) > 0 else 1.0
            # Use a smooth weighting: weight = exp(beta * (d - max_d)) with beta controlling sharpness
            beta = 5.0
            weights = np.exp(beta * (distances - max_d))
            weights = weights / (np.sum(weights) + 1e-10)
            
            targets = np.zeros_like(points)
            counts = np.zeros(N)
            
            for i, s in enumerate(facet_indices):
                c = circumcenters[i]
                w = weights[i]
                targets[s[0]] += w * c
                targets[s[1]] += w * c
                targets[s[2]] += w * c
                counts[s[0]] += w
                counts[s[1]] += w
                counts[s[2]] += w
            
            for j in range(N):
                if counts[j] > 1e-10:
                    targets[j] /= counts[j]
                else:
                    targets[j] = points[j]
            
            diff = targets - points
            momentum = momentum_factor * momentum + lr * diff
            points += momentum
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            
        except Exception:
            break

        current_radius = _calc_covering_radius(points)
        if current_radius < best_radius - 1e-10:
            best_radius = current_radius
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Adaptive learning rate: decrease slowly
        lr *= 0.9998
        
        # Periodic noise to escape local minima
        if iterations % 150 == 0:
            noise_scale = 0.015
            noise = rng.standard_normal(points.shape) * noise_scale
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0
        
        # Aggressive shake when stuck for a long time
        if no_improve_count > patience:
            noise_scale = 0.04
            noise = rng.standard_normal(points.shape) * noise_scale
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0
            lr = 0.05  # Reset learning rate after shake
        
        if no_improve_count > patience * 4:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s safety margin for 120s limit
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() - start_time >= time_limit - 5:
            break
        
        if restart == 0:
            # First restart: Fibonacci spiral
            points = _fibonacci_sphere(N)
        elif restart == 1:
            # Second restart: slightly perturbed Fibonacci
            points = _fibonacci_sphere(N)
            noise = rng.standard_normal(points.shape) * 0.08
            points += noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 2:
            # Third restart: more perturbed Fibonacci
            points = _fibonacci_sphere(N)
            noise = rng.standard_normal(points.shape) * 0.15
            points += noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif restart == 3:
            # Fourth restart: random
            points = _random_sphere(N, rng)
        else:
            # Other restarts: mix of random and Fibonacci with varying weights
            w = 0.3 + 0.4 * rng.random()
            points = _random_sphere(N, rng)
            fib = _fibonacci_sphere(N)
            points = w * points + (1 - w) * fib
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize from this starting point
        opt_points, opt_radius = _optimize(points, start_time, time_limit, rng)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
        
        # If we've found a very good solution, we can be more conservative
        if best_radius < np.radians(18.35):
            break
    
    # Final fine-tuning with very small steps and focused on worst facets
    if best_points is not None and time.time() - start_time < time_limit - 8:
        fine_points = best_points.copy()
        fine_radius = best_radius
        momentum = np.zeros_like(fine_points)
        lr = 0.005
        
        fine_start = time.time()
        for _ in range(5000):
            if time.time() - start_time >= time_limit - 3:
                break
            if time.time() - fine_start > 20:
                break
                
            try:
                circumcenters, distances, facet_indices = _compute_facet_data(fine_points)
                if len(circumcenters) == 0:
                    break
                
                # Use higher beta for fine-tuning to focus more on worst facets
                max_d = np.max(distances) if len(distances) > 0 else 1.0
                beta = 10.0
                weights = np.exp(beta * (distances - max_d))
                weights = weights / (np.sum(weights) + 1e-10)
                
                targets = np.zeros_like(fine_points)
                counts = np.zeros(N)
                
                for i, s in enumerate(facet_indices):
                    c = circumcenters[i]
                    w = weights[i]
                    targets[s[0]] += w * c
                    targets[s[1]] += w * c
                    targets[s[2]] += w * c
                    counts[s[0]] += w
                    counts[s[1]] += w
                    counts[s[2]] += w
                
                for j in range(N):
                    if counts[j] > 1e-10:
                        targets[j] /= counts[j]
                    else:
                        targets[j] = fine_points[j]
                
                diff = targets - fine_points
                momentum = 0.7 * momentum + lr * diff
                fine_points += momentum
                norms = np.linalg.norm(fine_points, axis=1, keepdims=True)
                fine_points = fine_points / norms
            except Exception:
                break
                
            cur_r = _calc_covering_radius(fine_points)
            if cur_r < fine_radius - 1e-12:
                fine_radius = cur_r
                best_points = fine_points.copy()
                best_radius = fine_radius
    
    return best_points
