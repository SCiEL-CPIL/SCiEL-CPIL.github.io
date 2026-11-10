# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates.
    """
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    n = 16
    d = 2

    def compute_R(points):
        """Compute the R metric directly."""
        dists = pdist(points, 'euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(params):
        """Objective for minimization: negative R."""
        points = params.reshape(n, d)
        return -compute_R(points)

    # Generate initial candidates
    def make_grid():
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        X, Y = np.meshgrid(x, y)
        return np.column_stack((X.ravel(), Y.ravel()))

    def make_random(seed):
        rng = np.random.RandomState(seed)
        return rng.uniform(-2, 2, size=(n, d))

    # Simulated Annealing
    def simulated_annealing(initial_points, max_iter=50000, t0=1.0, t_end=0.001, cooling=0.9998):
        """
        Simulated Annealing to explore the global landscape.
        """
        best_points = initial_points.copy()
        best_R = compute_R(best_points)
        current_points = initial_points.copy()
        current_R = best_R
        t = t0
        
        for i in range(max_iter):
            # Check deadline
            if time.time() > start_time + deadline - safety_margin:
                break
                
            # Perturb one point
            idx = np.random.randint(n)
            delta = np.random.normal(0, 0.1, size=d)
            new_points = current_points.copy()
            new_points[idx] += delta
            new_R = compute_R(new_points)
            
            # Acceptance criterion
            if new_R > current_R or np.random.random() < np.exp((new_R - current_R) / t):
                current_points = new_points
                current_R = new_R
                if current_R > best_R:
                    best_R = current_R
                    best_points = current_points.copy()
            
            t = max(t_end, t * cooling)
            
        return best_points, best_R

    # Powell refinement
    def powell_refine(points):
        """Refine using Powell's method."""
        params = points.flatten()
        try:
            result = minimize(
                objective,
                params,
                method='Powell',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            return result.x.reshape(n, d)
        except Exception:
            return points

    # Nelder-Mead refinement
    def nelder_mead_refine(points):
        """Refine using Nelder-Mead method."""
        params = points.flatten()
        try:
            result = minimize(
                objective,
                params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            return result.x.reshape(n, d)
        except Exception:
            return points

    # Multi-start strategy
    initial_candidates = [
        make_grid(),
        make_random(42),
        make_random(123),
        make_random(456),
        make_random(789),
    ]
    
    best_points = None
    best_R = 0.0
    
    for i, init in enumerate(initial_candidates):
        if time.time() > start_time + deadline - safety_margin:
            break
            
        # Simulated Annealing
        sa_points, sa_R = simulated_annealing(init, max_iter=20000)
        if sa_R > best_R:
            best_R = sa_R
            best_points = sa_points.copy()
        
        # Powell refinement
        pow_points = powell_refine(sa_points)
        pow_R = compute_R(pow_points)
        if pow_R > best_R:
            best_R = pow_R
            best_points = pow_points.copy()
        
        # Nelder-Mead refinement on the best so far
        nm_points = nelder_mead_refine(best_points)
        nm_R = compute_R(nm_points)
        if nm_R > best_R:
            best_R = nm_R
            best_points = nm_points.copy()
    
    # Final check
    if best_points is None:
        best_points = make_grid()
    
    if best_points.shape != (n, d):
        best_points = make_grid()
    
    return best_points

# EVOLVE-BLOCK-END
