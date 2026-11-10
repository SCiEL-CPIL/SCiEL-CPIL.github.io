# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s kill

    def exact_score(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (n, d):
            pts = pts.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        d_min = np.min(dists)
        return (d_min / d_max) ** 2

    def soft_min(dists, t=0.05):
        """Differentiable approximation of the minimum distance."""
        # Use a numerically stable log-sum-exp
        shifted = dists - np.max(dists)
        log_sum = np.log(np.sum(np.exp(shifted / t))) + np.max(dists) / t
        return t * (np.log(len(dists)) - log_sum)

    def smooth_objective(x):
        pts = x.reshape(n, d)
        dists = pdist(pts)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        # Normalize by d_max to fix scale invariance
        dists_norm = dists / d_max
        d_min_norm = soft_min(dists_norm)
        return -d_min_norm

    def hex_lattice_init(seed=0):
        """Generate a hexagonal lattice-like initial configuration for 16 points."""
        rng = np.random.RandomState(seed)
        
        # Hexagonal lattice basis
        a = 1.0
        # Generate points in a hexagonal pattern
        # Use a grid with hexagonal offset
        points = []
        
        # Try different hexagonal arrangements
        # 4x4 grid with hexagonal offset
        for i in range(4):
            for j in range(4):
                x = j * a + (i % 2) * a * 0.5
                y = i * a * np.sqrt(3) / 2
                points.append([x, y])
        
        pts = np.array(points)
        
        # Center and scale
        pts -= np.mean(pts, axis=0)
        dmax = np.max(pdist(pts))
        if dmax > 1e-12:
            pts = pts / dmax
        
        # Add small random perturbation
        pts += rng.normal(0, 0.01, pts.shape)
        
        return pts.flatten()

    def random_init(seed=0):
        """Generate a random initial configuration."""
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-1, 1, (n, d))
        pts -= np.mean(pts, axis=0)
        dmax = np.max(pdist(pts))
        if dmax > 1e-12:
            pts = pts / dmax
        return pts.flatten()

    best_x = None
    best_score = 0.0

    # Try multiple initialization strategies
    init_strategies = []
    
    # Hexagonal lattice with different seeds
    for seed in range(10):
        init_strategies.append(hex_lattice_init(seed))
    
    # Random initializations
    for seed in range(20):
        init_strategies.append(random_init(seed))
    
    # Sort by initial score (descending) to try best first
    init_scores = [(exact_score(x.reshape(n, d)), x) for x in init_strategies]
    init_scores.sort(key=lambda t: -t[0])
    
    # Try each initialization with L-BFGS-B refinement
    for init_score, x_start in init_scores:
        if time.time() > deadline - 5:
            break
        
        # Skip if initial score is already worse than best
        if init_score < best_score and best_score > 0.3:
            continue
        
        try:
            # Use L-BFGS-B with the smooth objective
            res = minimize(
                smooth_objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            local_score = exact_score(res.x)
            if local_score > best_score:
                best_score = local_score
                best_x = res.x.copy()
                
                # If we've achieved a very good score, we can stop early
                if best_score > 0.45:
                    break
                    
        except Exception:
            continue

    # If we still don't have a good solution, try a few more random restarts
    if best_score < 0.3:
        np.random.seed(123)
        for i in range(30):
            if time.time() > deadline - 5:
                break
            x_start = np.random.uniform(-1, 1, (n * d))
            # Normalize
            pts = x_start.reshape(n, d)
            pts -= np.mean(pts, axis=0)
            dmax = np.max(pdist(pts))
            if dmax > 1e-12:
                x_start = (pts / dmax).flatten()
            
            try:
                res = minimize(
                    smooth_objective,
                    x_start,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12}
                )
                s = exact_score(res.x)
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
            except Exception:
                continue

    # Fallback: simple grid if everything failed
    if best_x is None:
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        best_x = np.column_stack((xx.ravel(), yy.ravel())).flatten()
        dmax_init = np.max(pdist(best_x.reshape(n, d)))
        if dmax_init > 1e-12:
            best_x = (best_x.reshape(n, d) / dmax_init).flatten()

    return best_x.reshape(n, d)

# EVOLVE-BLOCK-END
