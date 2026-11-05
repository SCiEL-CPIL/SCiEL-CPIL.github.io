import numpy as np
from scipy.optimize import minimize
import time
from math import log

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin
    
    n = 8
    num_pairs = n * (n - 1) // 2
    
    def get_distances(points):
        # points shape (n, 2)
        # return array of pairwise distances shape (num_pairs,)
        # and indices if needed, but just values is enough for score
        d = np.zeros(num_pairs)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                d[idx] = np.linalg.norm(diff)
                idx += 1
        return d

    def objective(x):
        points = x.reshape(n, 2)
        d = get_distances(points)
        # Avoid log(0)
        if np.any(d < 1e-10):
            return 1e100
        D = np.max(d)
        if D < 1e-10:
            return 1e100
        # Maximize sum(log(d)) - 28 * log(D)
        # Minimize negative of that
        val = np.sum(np.log(d)) - 28.0 * np.log(D)
        return -val

    # Regular octagon initialization
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_oct = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_score = -np.inf
    best_points = init_oct
    
    # List of initial guesses
    inits = [init_oct.flatten()]
    
    # Add some random inits
    for _ in range(5):
        inits.append(np.random.uniform(-1, 1, n * 2))
        
    # Add perturbed octagon
    for _ in range(5):
        perturb = np.random.normal(0, 0.1, n * 2)
        inits.append(init_oct.flatten() + perturb)

    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < -best_score:
                # Calculate true score for this result
                pts = res.x.reshape(n, 2)
                d = get_distances(pts)
                D = np.max(d)
                if D > 1e-10 and np.all(d > 1e-10):
                    true_score_val = np.exp(np.sum(np.log(d)) - 28.0 * np.log(D))
                    # The score is Delta / n^n where Delta is product of distances after scaling to diameter 2.
                    # My objective was sum(log d) - 28 log(D).
                    # exp(obj) = prod(d) / D^28.
                    # This is exactly the score.
                    if true_score_val > best_score:
                        best_score = true_score_val
                        best_points = pts
        except Exception:
            pass

    # Final check: ensure distinct points
    d = get_distances(best_points)
    if np.min(d) < 1e-6:
        # Fallback to regular octagon if points collapsed
        best_points = init_oct
        best_score = 1.0 # Approximate

    # Convert to list of lists
    return best_points.tolist()
