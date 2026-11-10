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
    deadline = start_time + 580  # Safety margin before 630s kill

    def get_score(pts):
        """Compute the actual score R = (d_min/d_max)^2"""
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min(dists, t=1.0):
        """Differentiable approximation of the minimum distance using log-sum-exp."""
        a = -dists / t
        a_max = np.max(a)
        return -t * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def soft_max(dists, t=1.0):
        """Differentiable approximation of the maximum distance using log-sum-exp."""
        a = dists / t
        a_max = np.max(a)
        return t * (a_max + np.log(np.sum(np.exp(a - a_max)) / len(dists)))

    def surrogate_objective(x, t=1.0):
        """
        Smooth surrogate for the objective. We want to maximize (d_min/d_max)^2.
        Minimizing the negative of the log surrogate is equivalent.
        log(R) = 2 * (log(d_min) - log(d_max))
        We approximate log(d_min) and log(d_max) using soft_min/soft_max on distances.
        Actually, it's better to work with the ratio directly or its log.
        
        Let's define the objective to minimize as:
        f(x) = -log(soft_min(dists, t) / soft_max(dists, t))
             = log(soft_max(dists, t)) - log(soft_min(dists, t))
        
        This is smooth and differentiable.
        """
        pts = x.reshape(n, d)
        dists = pdist(pts)
        
        # Avoid log(0) issues
        sm = soft_min(dists, t)
        smx = soft_max(dists, t)
        
        if sm <= 0 or smx <= 0:
            return 1e10
        
        return np.log(smx) - np.log(sm)

    def surrogate_gradient(x, t=1.0):
        """
        Numerical gradient of the surrogate objective.
        """
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = surrogate_objective(x, t)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = surrogate_objective(x_plus, t)
            grad[i] = (f_plus - f0) / eps
        return grad

    # --- Phase 1: Multi-start with smooth surrogate optimization ---
    
    best_points = None
    best_score = -1.0

    # Generate diverse initial points
    np.random.seed(42)
    n_starts = 50  # Number of random starts
    
    # Also add some structured starts
    structured_starts = []
    
    # 1. Random uniform in unit square
    for i in range(n_starts):
        pts = np.random.uniform(-1, 1, (n, d))
        structured_starts.append(pts)
    
    # 2. Hexagonal lattice-like patterns
    # Try a few different hex lattice spacings
    for spacing in [0.5, 0.7, 1.0, 1.5]:
        pts = np.zeros((n, d))
        # Place points in a hexagonal pattern
        idx = 0
        for row in range(5):
            for col in range(4):
                if idx >= n:
                    break
                x = col * spacing * 1.5
                y = row * spacing * np.sqrt(3)
                if col % 2 == 1:
                    y += spacing * np.sqrt(3) / 2
                pts[idx, 0] = x
                pts[idx, 1] = y
                idx += 1
        structured_starts.append(pts)
    
    # 3. Circle with random perturbation
    for radius in [1.0, 1.5, 2.0]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        structured_starts.append(pts)
    
    # 4. Square grid
    for size in [2.0, 3.0, 4.0]:
        side = int(np.sqrt(n))
        pts = np.zeros((n, d))
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= n:
                    break
                pts[idx, 0] = (i - (side-1)/2) * size / side
                pts[idx, 1] = (j - (side-1)/2) * size / side
                idx += 1
        structured_starts.append(pts)
    
    # Combine all starts
    all_starts = structured_starts
    
    # Evaluate each start with local optimization
    for i, init_pts in enumerate(all_starts):
        if time.time() > deadline - 10:
            break
        
        x0 = init_pts.flatten()
        
        # Try a few different temperature values for the soft-min/max
        for t in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 5:
                break
            
            # Use BFGS with numerical gradient for the smooth surrogate
            try:
                res = minimize(
                    surrogate_objective,
                    x0,
                    args=(t,),
                    method='BFGS',
                    jac=surrogate_gradient,
                    options={'gtol': 1e-6, 'maxiter': 1000}
                )
                
                if res.success or res.fun < 1e9:
                    opt_pts = res.x.reshape(n, d)
                    score = get_score(opt_pts)
                    
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts.copy()
                        
            except Exception:
                continue
    
    # --- Phase 2: Polish with exact objective using Nelder-Mead ---
    if best_points is not None and time.time() < deadline - 2:
        # Use Nelder-Mead on the exact (nonsmooth) objective
        # We minimize -log(score) = -log(d_min^2/d_max^2) = 2*(log(d_max) - log(d_min))
        
        def exact_objective(x):
            pts = x.reshape(n, d)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0:
                return 1e10
            if d_min == 0:
                return 1e10
            return -2 * (np.log(d_min) - np.log(d_max))
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                exact_objective,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'maxfev': 10000}
            )
            
            if res.fun < 1e9:
                opt_pts = res.x.reshape(n, d)
                score = get_score(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
                    
        except Exception:
            pass
    
    # --- Phase 3: Final high-precision polish with tighter tolerances ---
    if best_points is not None and time.time() < deadline - 1:
        def exact_objective_tight(x):
            pts = x.reshape(n, d)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max == 0:
                return 1e10
            if d_min == 0:
                return 1e10
            return -2 * (np.log(d_min) - np.log(d_max))
        
        x0 = best_points.flatten()
        try:
            res = minimize(
                exact_objective_tight,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 10000, 'maxfev': 20000}
            )
            
            if res.fun < 1e9:
                opt_pts = res.x.reshape(n, d)
                score = get_score(opt_pts)
                
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
                    
        except Exception:
            pass
    
    # Ensure we always return a valid array
    if best_points is None:
        # Fallback: return random points
        np.random.seed(42)
        best_points = np.random.randn(n, d)
    
    return best_points


# EVOLVE-BLOCK-END
