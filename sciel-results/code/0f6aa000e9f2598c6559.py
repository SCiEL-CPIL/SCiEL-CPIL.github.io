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

def _optimize(points, start_time, time_limit, rng):
    """Run local optimization from initial points."""
    points = points.copy()
    best_points = points.copy()
    best_radius = _calc_covering_radius(points)
    
    momentum = np.zeros_like(points)
    lr = 0.08
    lr_decay = 0.9995
    momentum_factor = 0.6
    iterations = 0
    max_iters = 15000
    no_improve_count = 0
    patience = 500
    
    while time.time() - start_time < time_limit and iterations < max_iters:
        iterations += 1
        
        try:
            hull = ConvexHull(points)
            simplices = hull.simplices
            
            targets = np.zeros_like(points)
            counts = np.zeros(N)
            
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
                c = c / np.linalg.norm(c)
                
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
        
        # Adaptive learning rate
        lr *= lr_decay
        
        # Add noise to escape local minima
        if iterations % 200 == 0 or no_improve_count > patience:
            noise_scale = 0.02 if no_improve_count < patience else 0.05
            noise = rng.standard_normal(points.shape) * noise_scale
            points += noise
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
            no_improve_count = 0
        
        if no_improve_count > patience * 3:
            break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    time_limit = 105  # Leave 15s safety margin for 120s limit
    rng = np.random.default_rng(42)
    
    best_points = None
    best_radius = np.inf
    
    # Multiple restarts with different initializations
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() - start_time >= time_limit - 5:
            break
        
        if restart == 0:
            # First restart: Fibonacci spiral
            points = _fibonacci_sphere(N)
        elif restart == 1:
            # Second restart: slightly perturbed Fibonacci
            points = _fibonacci_sphere(N)
            noise = rng.standard_normal(points.shape) * 0.05
            points += noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # Other restarts: random with some structure
            points = _random_sphere(N, rng)
            # Mix in some Fibonacci structure
            fib = _fibonacci_sphere(N)
            points = 0.5 * points + 0.5 * fib
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize from this starting point
        opt_points, opt_radius = _optimize(points, start_time, time_limit, rng)
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
        
        # If we've found a very good solution, we can be more conservative
        if best_radius < np.radians(18.5):
            # Try a few more but with less aggressive noise
            pass
    
    # Final fine-tuning with very small steps
    if best_points is not None and time.time() - start_time < time_limit - 10:
        fine_points = best_points.copy()
        fine_radius = best_radius
        momentum = np.zeros_like(fine_points)
        lr = 0.01
        for _ in range(2000):
            if time.time() - start_time >= time_limit - 5:
                break
            try:
                hull = ConvexHull(fine_points)
                simplices = hull.simplices
                targets = np.zeros_like(fine_points)
                counts = np.zeros(N)
                for s in simplices:
                    v1, v2, v3 = fine_points[s[0]], fine_points[s[1]], fine_points[s[2]]
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
