# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # 50s safety margin

    def get_distances(points):
        return scipy.spatial.distance.pdist(points)

    def compute_score(points):
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_points(points):
        """Center and normalize points so max distance = 1."""
        points = points - np.mean(points, axis=0)
        dists = scipy.spatial.distance.pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max
        return points

    def sa_cost(points):
        """
        Cost for simulated annealing: we want to MINIMIZE this.
        Use a smooth approximation of -log(d_min/d_max) = log(d_max) - log(d_min)
        Add penalty for very small distances.
        """
        dists = scipy.spatial.distance.pdist(points)
        if np.any(dists < 1e-10):
            return 1e6
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e6
        # Cost = -log(d_min/d_max) = log(d_max) - log(d_min)
        # We want to minimize this (maximize d_min/d_max)
        cost = np.log(d_max) - np.log(d_min)
        # Additional penalty: sum of log of all distances below a threshold
        # to encourage spreading out
        small_dists = dists[dists < d_min * 2]
        if len(small_dists) > 0:
            cost += 0.1 * np.sum(np.log(d_min / small_dists + 1e-10))
        return cost

    def simulated_annealing(initial_points, num_iterations=50000, t_start=1.0, t_end=0.001, 
                            step_scale=0.1, cooling_rate=None, seed=None):
        """
        Simulated annealing to maximize min/max distance ratio.
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        
        if cooling_rate is None:
            cooling_rate = (t_end / t_start) ** (1.0 / num_iterations)
        
        current_points = initial_points.copy()
        current_cost = sa_cost(current_points)
        best_points = current_points.copy()
        best_cost = current_cost
        
        for i in range(num_iterations):
            if time.time() > deadline - 10:
                break
            
            temperature = t_start * (cooling_rate ** i)
            
            # Generate perturbation
            # Randomly perturb 1-3 points
            num_perturbed = rng.randint(1, 4)
            indices = rng.choice(n, size=num_perturbed, replace=False)
            step_size = step_scale * (temperature / t_start) ** 0.5 + 0.001
            
            new_points = current_points.copy()
            for idx in indices:
                new_points[idx] += rng.randn(d) * step_size
            
            # Normalize to keep scale consistent
            new_points = normalize_points(new_points)
            
            new_cost = sa_cost(new_points)
            
            # Accept if better or with probability exp(-delta/T)
            if new_cost < current_cost:
                current_points = new_points
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_points = current_points.copy()
            else:
                delta = new_cost - current_cost
                if delta < 0:
                    current_points = new_points
                    current_cost = new_cost
                else:
                    if temperature > 1e-10:
                        accept_prob = np.exp(-delta / temperature)
                        if rng.rand() < accept_prob:
                            current_points = new_points
                            current_cost = new_cost
        
        return best_points, best_cost

    def gradient_refine(points, max_iter=200):
        """
        Refine points using gradient descent on the soft objective.
        """
        def soft_objective(points_flat, tau=0.03):
            points = points_flat.reshape(n, d)
            dists = scipy.spatial.distance.pdist(points)
            if np.any(dists < 1e-10):
                return 1e6
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return 1e6
            # Minimize log(d_max) - log(d_min)
            return np.log(d_max) - np.log(d_min)
        
        def gradient(points_flat, tau=0.03):
            points = points_flat.reshape(n, d)
            dists = scipy.spatial.distance.pdist(points)
            if np.any(dists < 1e-10):
                return np.zeros_like(points_flat)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return np.zeros_like(points_flat)
            
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros_like(points_flat)
            for j in range(len(points_flat)):
                old_val = points_flat[j]
                points_flat[j] = old_val + eps
                f_plus = soft_objective(points_flat)
                points_flat[j] = old_val - eps
                f_minus = soft_objective(points_flat)
                points_flat[j] = old_val
                grad[j] = (f_plus - f_minus) / (2 * eps)
            return grad
        
        # Use scipy minimize with BFGS
        try:
            result = scipy.optimize.minimize(
                lambda x: soft_objective(x),
                points.flatten(),
                method='BFGS',
                options={'maxiter': max_iter, 'gtol': 1e-8}
            )
            refined = result.x.reshape(n, d)
            refined = normalize_points(refined)
            return refined
        except Exception:
            return points

    # Generate diverse initializations
    def circular_initial(n_points):
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        return points

    def two_ring_initial(n_points):
        """8 points on inner ring, 8 on outer ring"""
        points = np.zeros((n_points, 2))
        for i in range(8):
            angle = i * 2 * np.pi / 8
            points[i] = [0.5 * np.cos(angle), 0.5 * np.sin(angle)]
        for i in range(8):
            angle = (i + 0.5) * 2 * np.pi / 8
            points[8 + i] = [np.cos(angle), np.sin(angle)]
        return normalize_points(points)

    def fibonacci_spiral_initial(n_points):
        points = np.zeros((n_points, 2))
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            r = np.sqrt(i + 1) / np.sqrt(n_points)
            theta = i * golden_angle
            points[i] = [r * np.cos(theta), r * np.sin(theta)]
        return normalize_points(points)

    def random_initial(n_points, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
        points = rng.randn(n_points, d)
        return normalize_points(points)

    # Phase 1: Run Simulated Annealing from multiple starts
    initializations = [
        circular_initial(n),
        two_ring_initial(n),
        fibonacci_spiral_initial(n),
    ]
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        initializations.append(random_initial(n, seed=seed))

    best_points = None
    best_score = -1.0

    # Run SA from each initialization with different iteration counts
    sa_configs = [
        (50000, 1.0, 0.001, 0.1),
        (30000, 0.5, 0.0005, 0.05),
        (20000, 0.3, 0.0002, 0.03),
    ]

    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline - 60:
            break

        # Run SA with the most aggressive config first
        sa_iters, t_start, t_end, step_scale = sa_configs[0]
        sa_points, sa_cost = simulated_annealing(
            init_points, 
            num_iterations=sa_iters,
            t_start=t_start,
            t_end=t_end,
            step_scale=step_scale,
            seed=init_idx * 1000
        )
        
        sa_score = compute_score(sa_points)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()

        # If time permits, run a second SA pass with different params
        if time.time() < deadline - 30:
            sa_iters2, t_start2, t_end2, step_scale2 = sa_configs[1]
            sa_points2, sa_cost2 = simulated_annealing(
                sa_points,  # Start from best SA result
                num_iterations=sa_iters2,
                t_start=t_start2,
                t_end=t_end2,
                step_scale=step_scale2,
                seed=init_idx * 1000 + 500
            )
            sa_score2 = compute_score(sa_points2)
            if sa_score2 > best_score:
                best_score = sa_score2
                best_points = sa_points2.copy()

    # Phase 2: Gradient refinement on the best point
    if time.time() < deadline - 20 and best_points is not None:
        refined_points = gradient_refine(best_points, max_iter=100)
        refined_score = compute_score(refined_points)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points.copy()

    # Phase 3: Final SA polish with very low temperature
    if time.time() < deadline - 10 and best_points is not None:
        sa_points_final, _ = simulated_annealing(
            best_points,
            num_iterations=10000,
            t_start=0.01,
            t_end=0.0001,
            step_scale=0.01,
            seed=999
        )
        final_score = compute_score(sa_points_final)
        if final_score > best_score:
            best_score = final_score
            best_points = sa_points_final.copy()

    # Phase 4: One more gradient refinement
    if time.time() < deadline - 5 and best_points is not None:
        refined_final = gradient_refine(best_points, max_iter=50)
        refined_final_score = compute_score(refined_final)
        if refined_final_score > best_score:
            best_score = refined_final_score
            best_points = refined_final.copy()

    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, d)

    return best_points


# EVOLVE-BLOCK-END
