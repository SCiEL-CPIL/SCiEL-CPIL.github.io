"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Random initialization + Nelder-Mead with soft-min objective, annealed sharpness.
    """
    start_time = time.time()
    time_budget = 115.0  # Leave 5s margin

    def get_radius(x):
        """Compute the actual packing radius."""
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2)) / 2.0
        min_boundary = 1.0 - np.max(d_origin)
        return min(min_pair, min_boundary)

    def objective(x, alpha):
        """
        Negative soft-min of all pair half-distances and boundary distances.
        Uses log-sum-exp for numerical stability.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Pair distances (half)
        pair_d = np.sqrt(d2) / 2.0
        # Boundary distances
        boundary_d = 1.0 - d_origin
        
        # Combine all distances
        all_d = np.concatenate([pair_d.flatten(), boundary_d])
        all_d = np.maximum(all_d, 1e-12)
        
        # Soft min: -log(sum(exp(-alpha*d)))/alpha
        scaled = -alpha * all_d
        max_s = np.max(scaled)
        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        soft_min = -log_sum / alpha
        
        return -soft_min

    def random_init(seed):
        """Uniform random points in unit disk."""
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(0, 1.0, N)) * 0.95
        pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        return pts.flatten()

    def run_nelder_mead(x0, alpha, maxiter, maxfev):
        """Run Nelder-Mead with time check."""
        res = minimize(lambda x: objective(x, alpha), x0,
                       method='Nelder-Mead',
                       options={'maxiter': maxiter, 'maxfev': maxfev,
                                'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
        return res.x

    best_x = None
    best_r = -1.0

    # Multiple random restarts with annealed sharpness
    n_restarts = 8
    seeds = [1, 42, 123, 456, 789, 1000, 2024, 5050]
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget - 10:
            break
        
        seed = seeds[restart % len(seeds)]
        x0 = random_init(seed)
        current_x = x0.copy()
        current_r = get_radius(current_x)
        
        # Anneal sharpness: start low, increase over phases
        alphas = [10.0, 30.0, 80.0, 200.0, 500.0]
        iters_per_phase = [800, 1000, 1200, 1000, 800]
        fev_per_phase = [1000, 1500, 2000, 1500, 1000]
        
        for phase, (alpha, maxiter, maxfev) in enumerate(zip(alphas, iters_per_phase, fev_per_phase)):
            if time.time() - start_time > time_budget - 5:
                break
            
            try:
                res_x = run_nelder_mead(current_x, alpha, maxiter, maxfev)
                res_r = get_radius(res_x)
                
                if res_r > current_r:
                    current_x = res_x
                    current_r = res_r
                
                if current_r > best_r:
                    best_r = current_r
                    best_x = current_x.copy()
                    
            except Exception:
                pass
        
        # Final polish with very high sharpness
        if time.time() - start_time < time_budget - 3:
            try:
                res_x = run_nelder_mead(current_x, 1000.0, 500, 800)
                res_r = get_radius(res_x)
                if res_r > current_r:
                    current_x = res_x
                    current_r = res_r
                if current_r > best_r:
                    best_r = current_r
                    best_x = current_x.copy()
            except Exception:
                pass

    # If no valid result, return a simple ring pattern
    if best_x is None:
        rng = np.random.default_rng(0)
        angles = 2 * np.pi * np.arange(N) / N
        r = np.sqrt(rng.uniform(0, 1, N)) * 0.9
        best_x = np.column_stack([r * np.cos(angles), r * np.sin(angles)]).flatten()

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
