# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Uses multi-start Nelder-Mead with a soft-minimum objective to maximize
    the minimum clearance between circles and the boundary.
    """
    
    def soft_min_obj(x, k=20.0):
        """
        Objective function: Minimize the soft-minimum of the clearances.
        Clearances are pairwise distances and distances to boundary.
        Minimizing this maximizes the minimum clearance (radius).
        """
        pts = x.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(d2, np.inf)
        # We only need upper triangle to avoid double counting, 
        # but for soft min sum, order doesn't matter, just count each pair once.
        # Using triu_indices for efficiency
        iu = np.triu_indices(N, 1)
        d_pairs = np.sqrt(d2[iu] + 1e-12)
        
        # Boundary distances
        d_origin = np.linalg.norm(pts, axis=1)
        d_bound = 1.0 - d_origin
        
        # Combine all clearances
        # Clearances are distances between circles (d_pairs) and to wall (d_bound)
        # Note: The radius is min(min_pair_dist, min_bound_dist).
        # The soft min of the set of all these distances approximates the global min.
        all_clearances = np.concatenate([d_pairs, d_bound])
        
        # Soft min: -1/k * log(sum(exp(-k * d)))
        # To avoid overflow/underflow, we can scale or just use logsumexp
        # scipy.special.logsumexp is available
        from scipy.special import logsumexp
        log_sum_exp = logsumexp(-k * all_clearances)
        obj = log_sum_exp / k
        
        return obj

    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair_dist = np.sqrt(np.min(d2))
        min_bound_dist = 1.0 - np.max(d_origin)
        return min(min_pair_dist, min_bound_dist)

    def init_random(seed):
        rng = np.random.default_rng(seed)
        # Uniform distribution in unit disk
        r = rng.uniform(0, 1, N) ** 0.5
        theta = rng.uniform(0, 2 * np.pi, N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y]).flatten()

    def init_grid(seed):
        # A slightly structured initial guess to help escape local minima
        rng = np.random.default_rng(seed)
        pts = []
        # Hexagonal packing approximation
        sp = 0.2
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * sp
                y = j * sp * 0.866
                if i % 2:
                    x += sp * 0.5
                if x*x + y*y < 0.8**2:
                    pts.append([x, y])
        if len(pts) < N:
             # Fill rest randomly
             r = rng.uniform(0, 0.8, N - len(pts)) ** 0.5
             theta = rng.uniform(0, 2 * np.pi, N - len(pts))
             pts.extend([r[i]*np.cos(theta[i]), r[i]*np.sin(theta[i])] for i in range(N-len(pts)))
        return np.array(pts[:N]).flatten()

    best_x = None
    best_r = 0.0
    
    # Multi-start optimization
    # We use several random starts and one grid start
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    
    for seed in seeds:
        if seed % 2 == 0:
            x0 = init_random(seed)
        else:
            x0 = init_grid(seed)
            
        try:
            # Nelder-Mead is derivative-free
            # We set tolerances to be reasonably tight
            res = minimize(soft_min_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 3000,
                                    'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            r = get_radius(res.x)
            if r > best_r:
                best_r = r
                best_x = res.x.copy()
                
        except Exception as e:
            pass

    if best_x is None:
        return init_random(1).reshape(N, 2)
        
    # Refinement: Run a second pass with a sharper soft-min (higher k) 
    # starting from the best found solution to sharpen the minimum.
    # Or just run Nelder-Mead again with tighter tolerances.
    try:
        # Increase k to make the soft min closer to the hard min
        # We define a wrapper to pass k
        def sharp_obj(x):
            return soft_min_obj(x, k=50.0)
            
        res2 = minimize(sharp_obj, best_x, method='Nelder-Mead',
                       options={'maxiter': 1000, 'maxfev': 2000,
                                'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        r2 = get_radius(res2.x)
        if r2 > best_r:
            best_x = res2.x.copy()
            best_r = r2
    except:
        pass

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
