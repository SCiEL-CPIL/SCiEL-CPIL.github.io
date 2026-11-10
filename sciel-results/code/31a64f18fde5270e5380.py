# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin of 40 seconds

    n = 16

    def compute_score(points):
        if points.shape != (n, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_and_grad(x, beta):
        """
        Smooth surrogate: maximize soft_min / soft_max using log-sum-exp.
        Returns (objective_value, gradient) where we minimize negative of the ratio.
        """
        points = x.reshape(n, 2)
        d = pdist(points)
        
        # Avoid zero distances
        d_safe = np.maximum(d, 1e-12)
        
        # Soft min via log-sum-exp: soft_min = -log(sum(exp(-beta*d)))/beta
        # For numerical stability
        d_min_shift = d_safe - np.min(d_safe)
        log_sum_exp_min = np.log(np.sum(np.exp(-beta * d_min_shift))) + np.min(d_safe)
        soft_min = log_sum_exp_min / beta
        
        # Soft max via log-sum-exp: soft_max = log(sum(exp(beta*d)))/beta
        d_max_shift = d_safe - np.max(d_safe)
        log_sum_exp_max = np.log(np.sum(np.exp(beta * d_max_shift))) + np.max(d_safe)
        soft_max = log_sum_exp_max / beta
        
        if soft_max < 1e-12:
            return 0.0, np.zeros_like(x)
        
        ratio = soft_min / soft_max
        score = ratio * ratio
        
        # Weights for soft min
        exp_min = np.exp(-beta * d_min_shift)
        w_min = exp_min / np.sum(exp_min)
        
        # Weights for soft max
        exp_max = np.exp(beta * d_max_shift)
        w_max = exp_max / np.sum(exp_max)
        
        # Build gradients
        grad_soft_min = np.zeros_like(x)
        grad_soft_max = np.zeros_like(x)
        idx = np.triu_indices(n, k=1)
        
        for k in range(len(idx[0])):
            i, j = idx[0][k], idx[1][k]
            diff = points[i] - points[j]
            dist = d_safe[k]
            if dist < 1e-12:
                continue
            
            # Gradient of d_k w.r.t point i: diff/dist, w.r.t point j: -diff/dist
            grad_i_min = (w_min[k] / beta) * (diff / dist)
            grad_j_min = -(w_min[k] / beta) * (diff / dist)
            
            grad_i_max = (w_max[k] / beta) * (diff / dist)
            grad_j_max = -(w_max[k] / beta) * (diff / dist)
            
            grad_soft_min[2*i:2*i+2] += grad_i_min
            grad_soft_min[2*j:2*j+2] += grad_j_min
            
            grad_soft_max[2*i:2*i+2] += grad_i_min
            grad_soft_max[2*j:2*j+2] += grad_j_min
        
        # d(score)/dx = 2*ratio * (d(soft_min)/dx * soft_max - soft_min * d(soft_max)/dx) / soft_max^2
        # = 2*ratio/soft_max * (grad_soft_min - ratio * grad_soft_max)
        
        grad_score = 2 * ratio / soft_max * (grad_soft_min - ratio * grad_soft_max)
        
        # We minimize -score
        return -score, -grad_score

    def optimize_from(points_init, beta_schedule=None, max_iters=150):
        x0 = points_init.flatten()
        
        if beta_schedule is None:
            beta_schedule = [5.0, 20.0, 100.0, 500.0, 2000.0]
        
        best_x = x0.copy()
        best_score = compute_score(points_init)
        
        for beta in beta_schedule:
            if time.time() > deadline:
                break
            
            try:
                res = minimize(
                    lambda x, b=beta: objective_and_grad(x, b)[0],
                    x0,
                    jac=lambda x, b=beta: objective_and_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': max_iters, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                x0 = res.x
                points = x0.reshape(n, 2)
                score = compute_score(points)
                
                if score > best_score:
                    best_score = score
                    best_x = x0.copy()
            except Exception:
                continue
        
        return best_x.reshape(n, 2), best_score

    def differential_evolution(points_init, pop_size=20, max_iter=50, mutation=0.5, crossover=0.9):
        """
        Global optimizer to escape local minima.
        """
        if time.time() > deadline - 30:
            return points_init, compute_score(points_init)
        
        # Normalize to unit sphere for DE
        pts = points_init.copy()
        centroid = np.mean(pts, axis=0)
        pts = pts - centroid
        scale = np.max(np.max(np.abs(pts), axis=0))
        if scale > 0:
            pts = pts / scale
        
        x0 = pts.flatten()
        
        # Initialize population
        pop = [x0.copy()]
        for _ in range(pop_size - 1):
            # Random perturbation
            perturbation = np.random.randn(n, 2) * 0.5
            p = (pts + perturbation).flatten()
            pop.append(p)
        
        best_x = x0.copy()
        best_score = compute_score(x0.reshape(n, 2))
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            new_pop = []
            for i in range(pop_size):
                # Select three distinct individuals
                a, b, c = np.random.choice(pop_size, 3, replace=False)
                
                # Mutation
                v = pop[a] + mutation * (pop[b] - pop[c])
                
                # Crossover
                trial = pop[i].copy()
                for j in range(len(v)):
                    if np.random.rand() < crossover:
                        trial[j] = v[j]
                
                # Evaluate
                trial_pts = trial.reshape(n, 2)
                score = compute_score(trial_pts)
                
                # Selection
                current_score = compute_score(pop[i].reshape(n, 2))
                if score > current_score:
                    pop[i] = trial
                    if score > best_score:
                        best_score = score
                        best_x = trial.copy()
            
            # Elite preservation
            pop.append(best_x.copy())
            if len(pop) > pop_size:
                pop.pop(0)
        
        return best_x.reshape(n, 2), best_score

    def regular_polygon():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def concentric_rings(r_inn, n_outer, n_inner):
        """Create concentric ring configuration."""
        angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
        pts_outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        pts_inner = r_inn * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
        return np.vstack([pts_outer, pts_inner])

    def two_circles(r1, r2, n1, n2, phase=0):
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase
        pts1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
        pts2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
        return np.vstack([pts1, pts2])

    def random_points(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2)

    def grid_4x4():
        grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
        grid = (grid - 1.5) / 1.5
        return grid

    def triangular_lattice():
        s = 1.0
        h = s * np.sqrt(3) / 2
        lattice = []
        for row in range(-2, 3):
            for col in range(-2, 3):
                x = col * s + (row % 2) * s / 2
                y = row * h
                lattice.append([x, y])
        lattice = np.array(lattice)
        centroid = np.mean(lattice, axis=0)
        dists = np.sum((lattice - centroid) ** 2, axis=1)
        idx = np.argsort(dists)[:16]
        config = lattice[idx]
        config = (config - np.mean(config, axis=0)) / np.max(np.max(np.abs(config), axis=0))
        return config

    def square_config():
        inner_r = 0.5
        outer_r = 1.0
        inner_pts = np.array([[inner_r, inner_r], [-inner_r, inner_r], 
                              [-inner_r, -inner_r], [inner_r, -inner_r]])
        outer_pts = []
        outer_pts.append([outer_r, outer_r])
        outer_pts.append([-outer_r, outer_r])
        outer_pts.append([-outer_r, -outer_r])
        outer_pts.append([outer_r, -outer_r])
        outer_pts.append([0, outer_r])
        outer_pts.append([outer_r, 0])
        outer_pts.append([0, -outer_r])
        outer_pts.append([-outer_r, 0])
        outer_pts = np.array(outer_pts)
        return np.vstack([inner_pts, outer_pts])

    def ring_config():
        center = np.array([[0.0, 0.0]])
        ring6_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ring6_r = 0.6
        ring6 = np.column_stack([ring6_r * np.cos(ring6_angles), ring6_r * np.sin(ring6_angles)])
        ring9_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
        ring9_r = 1.0
        ring9 = np.column_stack([ring9_r * np.cos(ring9_angles), ring9_r * np.sin(ring9_angles)])
        return np.vstack([center, ring6, ring9])

    # Generate diverse initializations
    initializations = []
    
    # Structured configurations
    initializations.append(regular_polygon())
    initializations.append(grid_4x4())
    initializations.append(triangular_lattice())
    initializations.append(square_config())
    initializations.append(ring_config())
    
    # Concentric rings with various inner radii
    for r_inn in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        initializations.append(concentric_rings(r_inn, 11, 5))
    
    # Two circles with various radii and phases
    for r1, r2, n1, n2, phase in [
        (1.0, 0.5, 8, 8, 0),
        (1.0, 0.5, 8, 8, np.pi/8),
        (1.0, 0.6, 10, 6, 0),
        (1.0, 0.6, 10, 6, np.pi/10),
        (1.0, 0.4, 12, 4, 0),
        (1.0, 0.7, 6, 10, np.pi/6),
    ]:
        initializations.append(two_circles(r1, r2, n1, n2, phase))
    
    # Random initializations
    for seed in range(10):
        initializations.append(random_points(seed))
    
    # Grid-based
    for seed in range(5):
        np.random.seed(seed)
        side = int(np.ceil(np.sqrt(n)))
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) < n:
                    pts.append([i, j])
        pts = np.array(pts[:n], dtype=float)
        pts += np.random.randn(n, 2) * 0.3
        initializations.append(pts)
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Local optimization from diverse initializations
    for i, init in enumerate(initializations):
        if time.time() > deadline - 60:
            break
        
        points, score = optimize_from(init)
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        return regular_polygon()
    
    # Phase 2: Global search with Differential Evolution from best local solution
    if time.time() < deadline - 40:
        de_points, de_score = differential_evolution(best_points)
        if de_score > best_score:
            best_score = de_score
            best_points = de_points
    
    # Phase 3: Final polish with higher precision
    if time.time() < deadline - 15:
        points, score = optimize_from(best_points, beta_schedule=[1000.0, 5000.0, 10000.0], max_iters=200)
        if score > best_score:
            best_points = points
    
    return best_points


# EVOLVE-BLOCK-END
