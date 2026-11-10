# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 630s kill

    def score(points):
        """Calculate R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 0.0
        ratio = d_min / d_max
        return ratio ** 2

    def soft_score(params, beta=20.0):
        """
        Smooth surrogate using soft-min and soft-max via log-sum-exp.
        Returns -soft_R where soft_R = (soft_min/soft_max)^2
        """
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        
        # Soft min: -beta * log(sum(exp(-dists/beta)))
        # Soft max: beta * log(sum(exp(dists/beta)))
        scaled_min = -dists / beta
        max_smin = np.max(scaled_min)
        soft_min = -beta * (max_smin + np.log(np.sum(np.exp(scaled_min - max_smin))))
        
        scaled_max = dists / beta
        max_smax = np.max(scaled_max)
        soft_max = beta * (max_smax + np.log(np.sum(np.exp(scaled_max - max_smax))))
        
        if soft_max < 1e-10:
            return 0.0
        
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def soft_score_tight(params, beta=5.0):
        """Tighter smooth surrogate for final polish"""
        points = params.reshape((n, 2))
        dists = pdist(points, 'euclidean')
        
        scaled_min = -dists / beta
        max_smin = np.max(scaled_min)
        soft_min = -beta * (max_smin + np.log(np.sum(np.exp(scaled_min - max_smin))))
        
        scaled_max = dists / beta
        max_smax = np.max(scaled_max)
        soft_max = beta * (max_smax + np.log(np.sum(np.exp(scaled_max - max_smax))))
        
        if soft_max < 1e-10:
            return 0.0
        
        ratio = soft_min / soft_max
        return -(ratio ** 2)

    def make_hex_init(scale=1.0, aspect=1.0):
        """Hexagonal lattice arrangement with adjustable aspect ratio"""
        points = []
        # Center
        points.append([0.0, 0.0])
        # First ring (6 points)
        for i in range(6):
            angle = i * np.pi / 3
            points.append([scale * np.cos(angle), scale * aspect * np.sin(angle)])
        # Second ring (9 points) - hexagonal pattern
        for i in range(9):
            if i < 6:
                angle = i * np.pi / 3 + np.pi / 6
                r = 2.0 * scale
                ry = 2.0 * scale * aspect
            else:
                angle = (i - 6) * 2 * np.pi / 3
                r = 2.0 * scale
                ry = 2.0 * scale * aspect
            points.append([r * np.cos(angle), ry * np.sin(angle)])
        return np.array(points[:n])

    def make_hex_init_v2(scale=1.0):
        """Alternative hexagonal arrangement with different point selection"""
        points = []
        # Use a 4x4 grid rotated 45 degrees, then select 16 points
        # This creates a more symmetric configuration
        grid = []
        for i in range(4):
            for j in range(4):
                x = (i - 1.5) * scale
                y = (j - 1.5) * scale
                grid.append([x, y])
        grid = np.array(grid)
        
        # Rotate by 45 degrees
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        grid_rot = grid @ R.T
        
        # Center and scale
        center = np.mean(grid_rot, axis=0)
        grid_rot = grid_rot - center
        max_dist = np.max(np.linalg.norm(grid_rot, axis=1))
        if max_dist > 0:
            grid_rot = grid_rot / max_dist * 1.5
        
        return grid_rot

    def make_concentric_init():
        """Concentric circles with optimal spacing"""
        points = []
        # 4 points on inner circle
        for i in range(4):
            angle = i * np.pi / 2 + np.pi / 4
            points.append([0.5 * np.cos(angle), 0.5 * np.sin(angle)])
        # 12 points on outer circle
        for i in range(12):
            angle = i * 2 * np.pi / 12
            points.append([1.0 * np.cos(angle), 1.0 * np.sin(angle)])
        return np.array(points[:n])

    def make_random_init(scale=1.0):
        return np.random.randn(n, 2) * scale

    np.random.seed(42)
    inits = [
        make_hex_init(1.0, 1.0),
        make_hex_init(1.0, 0.866),  # True hexagonal aspect
        make_hex_init(1.0, 1.1),
        make_hex_init_v2(1.0),
        make_concentric_init(),
        make_random_init(1.0),
        make_random_init(0.7),
        make_random_init(1.3),
    ]

    best_points = inits[0].copy()
    best_score = score(inits[0])

    # Phase 1: Differential evolution for global exploration
    if time.time() < deadline - 250:
        def de_objective(params):
            points = params.reshape((n, 2))
            return -score(points)
        
        bounds = [(-2.5, 2.5)] * (n * 2)
        try:
            de_result = differential_evolution(
                de_objective, bounds, 
                maxiter=150, popsize=20, 
                tol=1e-10, seed=42,
                polish=False,
                init='sobol'
            )
            candidate = de_result.x.reshape((n, 2))
            trial_score = score(candidate)
            if trial_score > best_score:
                best_score = trial_score
                best_points = candidate.copy()
        except Exception:
            pass

    # Phase 2: Simulated annealing with adaptive step size and soft objective
    if time.time() < deadline - 150:
        def sa_optimize(init_points, n_steps=4000, use_soft=False):
            current = init_points.copy()
            current_score = score(current)
            best_sa = current.copy()
            best_sa_score = current_score
            
            # Adaptive step size
            base_step = 0.15
            step_scale = 1.0
            T = 0.05
            cooling = 0.9995
            
            for step in range(n_steps):
                if time.time() > deadline - 80:
                    break
                    
                # Perturb
                delta = np.random.randn(n, 2) * base_step * step_scale
                proposed = current + delta
                
                if use_soft:
                    proposed_score = -soft_score(proposed.flatten())
                    current_score_eval = -soft_score(current.flatten())
                else:
                    proposed_score = score(proposed)
                    current_score_eval = current_score
                
                if proposed_score > current_score_eval:
                    current = proposed
                    current_score = score(current)
                    if current_score > best_sa_score:
                        best_sa = current.copy()
                        best_sa_score = current_score
                    step_scale *= 0.93
                else:
                    if T > 1e-10:
                        prob = np.exp((proposed_score - current_score_eval) / T)
                        if np.random.random() < prob:
                            current = proposed
                            current_score = score(current)
                            step_scale *= 1.03
                    step_scale = min(step_scale * 1.01, 2.5)
                
                T *= cooling
            
            return best_sa, best_sa_score
        
        # Run SA from multiple starting points, alternating soft/exact
        sa_starts = [best_points]
        for init in inits:
            if time.time() > deadline - 100:
                break
            sa_starts.append(init.copy())
        
        for idx, sa_init in enumerate(sa_starts):
            if time.time() > deadline - 80:
                break
            use_soft = (idx % 2 == 1)
            sa_result, sa_score = sa_optimize(sa_init, n_steps=3500, use_soft=use_soft)
            if sa_score > best_score:
                best_score = sa_score
                best_points = sa_result.copy()

    # Phase 3: L-BFGS-B with smooth surrogate for gradient-based refinement
    if time.time() < deadline - 80:
        def lbfgs_soft_objective(params):
            return soft_score(params, beta=20.0)
        
        # Multiple L-BFGS-B runs with different beta values
        for beta in [20.0, 10.0, 5.0]:
            if time.time() > deadline - 40:
                break
            
            def lbfgs_objective(params, b=beta):
                return soft_score(params, beta=b)
            
            try:
                lbfgs_result = minimize(
                    lbfgs_objective, best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                candidate = lbfgs_result.x.reshape((n, 2))
                trial_score = score(candidate)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Phase 4: Nelder-Mead local refinement on exact objective
    if time.time() < deadline - 30:
        def nm_objective(params):
            points = params.reshape((n, 2))
            return -score(points)
        
        nm_runs = 6
        for r in range(nm_runs):
            if time.time() > deadline - 15:
                break
            
            if r == 0:
                init = best_points.copy()
            elif r == 1:
                init = best_points + np.random.randn(n, 2) * 0.02
            elif r == 2:
                init = best_points + np.random.randn(n, 2) * 0.05
            else:
                init = best_points + np.random.randn(n, 2) * 0.03
            
            try:
                nm_result = minimize(
                    nm_objective, init.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 
                             'adaptive': True}
                )
                candidate = nm_result.x.reshape((n, 2))
                trial_score = score(candidate)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = candidate.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish with tight soft objective
    if time.time() < deadline - 10:
        def final_soft_objective(params):
            return soft_score_tight(params, beta=3.0)
        
        try:
            final_result = minimize(
                final_soft_objective, best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            candidate = final_result.x.reshape((n, 2))
            trial_score = score(candidate)
            if trial_score > best_score:
                best_points = candidate.copy()
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-END
