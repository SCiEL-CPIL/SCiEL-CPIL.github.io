# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective(x):
    """
    Objective function for minimization.
    We want to maximize (d_min / d_max)^2.
    This is equivalent to minimizing -log(d_min / d_max) = log(d_max) - log(d_min).
    Using log makes the objective smoother and scale-invariant.
    If d_min or d_max is 0 or invalid, return a large penalty.
    """
    points = x.reshape(14, 3)
    
    # Basic validity checks
    if not np.all(np.isfinite(points)):
        return 1e6
        
    dists = pdist(points)
    if len(dists) == 0:
        return 1e6
        
    d_min = np.min(dists)
    d_max = np.max(dists)
    
    if d_min <= 1e-9 or d_max <= 1e-9:
        return 1e6
        
    # Minimize log(d_max) - log(d_min)
    # This is equivalent to maximizing log(d_min/d_max)
    val = np.log(d_max) - np.log(d_min)
    
    return val

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin
    
    n = 14
    
    # Differential Evolution bounds
    # We can constrain the search to a reasonable box, e.g., [-1, 1]^3
    # The metric is scale-invariant, so the absolute scale doesn't matter,
    # but DE needs finite bounds.
    bounds = [(-1.0, 1.0)] * (n * 3)
    
    # Strategy: Run Differential Evolution with a time budget.
    # DE is a global search method that can escape local minima better than gradient descent.
    # It doesn't rely on gradients, which is good because the "min distance" part of the objective
    # is non-smooth (kinked) at points where the identity of the minimum pair changes.
    
    # We will use a wrapper to handle the time limit.
    # scipy's differential_evolution has a maxiter parameter, but we want time-based stopping.
    # We can pass a callback that raises an exception or sets a flag.
    
    # However, a simpler approach for robustness within the time limit:
    # 1. Run DE with a moderate number of iterations/evaluations.
    # 2. If time remains, run a second pass with tighter settings or different seed.
    
    # Let's estimate evaluations per second.
    # For n=14, pdist is very fast. ~10,000-50,000 evals/sec might be possible.
    # Let's aim for a few thousand evaluations to start.
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Fibonacci sphere (flattened)
    def fibonacci_sphere(n):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)
    
    fib_points = fibonacci_sphere(n)
    best_points = fib_points.copy()
    best_score = _score(best_points)
    
    # Callback to stop DE if time limit is approached
    def callback(xk, convergence):
        if time.time() > deadline - 10:
            raise TimeoutError("Time limit reached")
        # Update best if this candidate is better
        nonlocal best_score, best_points
        s = _score(xk.reshape(14, 3))
        if s > best_score:
            best_score = s
            best_points = xk.reshape(14, 3).copy()
        return False # Continue
    
    # Run Differential Evolution
    # popsize=15 is standard, maxiter=200 is a good start.
    # tol=1e-8 for precision.
    # seed=0 for reproducibility in the search process (though randomness is inherent)
    
    try:
        # Use a relatively small popsize and maxiter to allow for multiple restarts if needed
        # or just one long run.
        # Let's try one run with reasonable parameters first.
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            callback=callback,
            init='sobol', # Better initial sampling
            polish=True   # Use L-BFGS-B on the result to polish
        )
        
        if result.success or result.x is not None:
            candidate = result.x.reshape(14, 3)
            s = _score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()
                
    except TimeoutError:
        pass
    except Exception as e:
        # If DE fails, we still have the initial guess
        pass

    # If we have time left, try a second run with a different seed or strategy
    if time.time() < deadline - 20:
        try:
            # Second run with different seed and maybe different popsize
            result2 = differential_evolution(
                _objective,
                bounds,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                seed=123,
                callback=callback,
                init='sobol',
                polish=True
            )
            
            if result2.x is not None:
                candidate = result2.x.reshape(14, 3)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
                    
        except TimeoutError:
            pass
        except Exception:
            pass

    # Final polish using L-BFGS-B on the best found point
    # This can squeeze out a bit more performance if DE stopped near a local optimum
    if best_points is not None and time.time() < deadline - 5:
        try:
            from scipy.optimize import minimize
            x0 = best_points.flatten()
            
            # Use L-BFGS-B, which handles bounds
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            if res.x is not None:
                candidate = res.x.reshape(14, 3)
                s = _score(candidate)
                if s > best_score:
                    best_score = s
                    best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = fib_points
        
    return best_points

# EVOLVE-BLOCK-END
