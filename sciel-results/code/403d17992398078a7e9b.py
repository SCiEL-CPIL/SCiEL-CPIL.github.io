# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_points_from_params(params):
    """
    Generate 14 points from 3 parameters (a, b, c).
    Configuration:
    8 points: (±a, ±b, ±c)
    4 points: (±b, ±c, ±a) with specific sign patterns to maintain symmetry?
    Actually, let's use the exact structure from the inspiration that worked:
    P1-8: (±a, ±b, ±c)
    P9-12: (±b, ±c, ±a) -> Wait, the inspiration had:
    [b, c, a], [b, -c, a], [b, c, -a], [b, -c, -a]
    This is x=b, y=±c, z=±a.
    P13-14: [c, a, b], [c, -a, b]
    This is x=c, y=±a, z=b.
    
    Let's verify the count: 8 + 4 + 2 = 14.
    """
    a, b, c = params
    # Ensure positive parameters for the base shape, symmetry handles signs
    a = abs(a)
    b = abs(b)
    c = abs(c)
    
    # 8 points: all combinations of signs for (a, b, c)
    pts_8 = np.array([
        [ a,  b,  c], [ a,  b, -c], [ a, -b,  c], [ a, -b, -c],
        [-a,  b,  c], [-a,  b, -c], [-a, -b,  c], [-a, -b, -c]
    ])
    
    # 4 points: x=b, y=±c, z=±a
    pts_4 = np.array([
        [ b,  c,  a], [ b, -c,  a],
        [ b,  c, -a], [ b, -c, -a]
    ])
    
    # 2 points: x=c, y=±a, z=b
    pts_2 = np.array([
        [ c,  a,  b],
        [ c, -a,  b]
    ])
    
    points = np.vstack([pts_8, pts_4, pts_2])
    
    # Normalize to unit sphere to remove scale invariance (ratio is scale invariant)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero if params are 0
    norms[norms == 0] = 1
    points = points / norms
    
    return points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Uses a symmetry-constrained geometric search strategy:
    1. Define a 14-point configuration with D2 symmetry parameterized by 3 variables (a, b, c).
    2. Use multi-start Nelder-Mead optimization to maximize (d_min/d_max)^2.
    3. This reduces the search space from 42 dimensions to 3, allowing for a much more thorough global search.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -np.inf
    best_params = None
    
    # Objective function for minimization (we maximize score, so minimize negative score)
    def objective(params):
        points = _generate_points_from_params(params)
        s = _score(points)
        return -s
    
    # Initial guesses for (a, b, c)
    # We want to explore a range of aspect ratios
    initial_guesses = []
    
    # 1. Near equal parameters (cube-like)
    initial_guesses.append([1.0, 1.0, 1.0])
    
    # 2. Elongated
    initial_guesses.append([1.0, 0.5, 0.5])
    initial_guesses.append([0.5, 1.0, 0.5])
    initial_guesses.append([0.5, 0.5, 1.0])
    
    # 3. Flattened
    initial_guesses.append([1.0, 1.0, 0.5])
    initial_guesses.append([1.0, 0.5, 1.0])
    initial_guesses.append([0.5, 1.0, 1.0])
    
    # 4. Random diverse starts
    np.random.seed(42)
    for _ in range(20):
        a = np.random.uniform(0.1, 2.0)
        b = np.random.uniform(0.1, 2.0)
        c = np.random.uniform(0.1, 2.0)
        initial_guesses.append([a, b, c])
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead with tight tolerances
            res = minimize(
                objective, 
                guess, 
                method='Nelder-Mead', 
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000, 'maxfev': 5000}
            )
            
            if res.success or res.fun < 0: # Check if we got a valid score
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
                    
        except Exception:
            continue

    # If Nelder-Mead failed or didn't find anything good, try a simple grid search as fallback
    if best_params is None or best_score < 0:
        best_score = -np.inf
        for a in np.linspace(0.1, 2.0, 20):
            for b in np.linspace(0.1, 2.0, 20):
                for c in np.linspace(0.1, 2.0, 20):
                    if time.time() > deadline:
                        break
                    params = [a, b, c]
                    points = _generate_points_from_params(params)
                    s = _score(points)
                    if s > best_score:
                        best_score = s
                        best_params = params

    # Final polish with a very fine local search around the best found params
    if best_params is not None and time.time() < deadline - 10:
        # Do a few more Nelder-Mead runs starting from the best found point with slightly perturbed starts
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbation = np.random.normal(0, 0.01, 3)
            guess = best_params + perturbation
            try:
                res = minimize(
                    objective, 
                    guess, 
                    method='Nelder-Mead', 
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 5000}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                continue

    # Generate final points from best parameters
    if best_params is None:
        # Fallback to a known good configuration (e.g., icosahedron + 2 points) if optimization failed completely
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        points = np.vstack([ico, extra])
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        return points

    points = _generate_points_from_params(best_params)
    return points

# EVOLVE-BLOCK-END
