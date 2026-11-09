# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    """
    n = 14
    d = 3
    start_time = time.time()
    time_limit = 600  # safety margin before 630s kill

    def get_score(pts):
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts, metric='euclidean')
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def fast_score(pts):
        """Fast score using vectorized numpy (no scipy overhead)."""
        # Pairwise distances: (n, n)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_max):
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_surrogate(pts, alpha=50.0):
        """
        Smooth surrogate using log-sum-exp for soft-min and soft-max.
        Maximizing log(soft_min) - log(soft_max) approximates maximizing log(d_min/d_max).
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        # Soft min: log(sum(exp(-alpha * dists))) / (-alpha) approximates min
        # Soft max: log(sum(exp(alpha * dists))) / alpha approximates max
        # We work in squared distances for stability, then take sqrt at end conceptually
        # Actually let's work with distances directly for clarity but it's expensive.
        # Let's use squared distances for the surrogate to avoid sqrt in the loop?
        # No, the metric is on Euclidean distances.
        # dists = sqrt(sq_dists)
        # This is expensive. Let's stick to fast_score for the main loop but use 
        # a cheaper heuristic or just rely on fast_score being fast enough.
        
        # For 14 points, 91 pairs. Numpy overhead is the main cost.
        # Let's just use fast_score but optimize the inner loop of SA.
        return 0.0 # Placeholder, we will use fast_score

    def sa_optimize(init_pts, time_budget):
        """
        Run Simulated Annealing to optimize the points.
        """
        current_pts = init_pts.copy()
        current_score = fast_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Initial temperature
        T = 0.5 
        T_min = 1e-6
        cooling_rate = 0.9998 # Slower cooling
        
        # Move magnitude
        move_sigma = 0.2
        
        no_improve_steps = 0
        max_no_improve = 5000
        
        deadline = time.time() + time_budget
        
        # Pre-allocate
        new_pts = current_pts.copy()
        
        while time.time() < deadline:
            if no_improve_steps > max_no_improve:
                # Restart from best point with higher T
                current_pts = best_pts.copy()
                current_score = best_score
                T = 0.5
                no_improve_steps = 0
                move_sigma = 0.2
                continue

            # Generate a move: perturb one random point
            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts[idx] = current_pts[idx] + move
            
            new_score = fast_score(new_pts)
            
            # Acceptance criterion
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # Accept with probability exp((new - current) / T)
                delta = new_score - current_score
                # If delta is very negative, exp will be 0
                if delta > -10 * T: # Optimization: avoid underflow
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                no_improve_steps += 1
            
            # Cooling
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # If T is at min, we are effectively doing local search (hill climbing)
                # Reduce move size to polish
                move_sigma *= 0.95
                if move_sigma < 1e-5:
                    # Try to restart with larger moves if stuck
                    if no_improve_steps > 1000:
                         T = 0.1
                         move_sigma = 0.1
                         no_improve_steps = 0

        return best_pts, best_score

    def polish(pts, time_budget):
        """
        High-precision polish pass using very small moves and slow cooling.
        """
        current_pts = pts.copy()
        current_score = fast_score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        T = 0.01
        T_min = 1e-8
        cooling_rate = 0.9999
        move_sigma = 0.01
        
        no_improve_steps = 0
        max_no_improve = 10000
        deadline = time.time() + time_budget
        
        new_pts = current_pts.copy()
        
        while time.time() < deadline:
            if no_improve_steps > max_no_improve:
                current_pts = best_pts.copy()
                current_score = best_score
                T = 0.01
                move_sigma = 0.01
                no_improve_steps = 0
                continue

            idx = np.random.randint(n)
            move = np.random.randn(d) * move_sigma
            new_pts[idx] = current_pts[idx] + move
            
            new_score = fast_score(new_pts)
            
            if new_score >= current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve_steps = 0
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                delta = new_score - current_score
                if delta > -10 * T:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                no_improve_steps += 1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                move_sigma *= 0.98

        return best_pts, best_score

    # Initializations
    initializations = []
    
    # 1. Icosahedron + Poles (normalized)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms_ico = np.linalg.norm(ico_verts, axis=1)
    ico_verts_norm = ico_verts / norms_ico[:, np.newaxis]
    base_points = np.vstack([ico_verts_norm, np.array([[0, 0, 1.0], [0, 0, -1.0]])])
    initializations.append(base_points)
    
    # 2. Random points on sphere
    np.random.seed(42)
    for i in range(20):
        rand_pts = np.random.randn(n, d)
        norms = np.linalg.norm(rand_pts, axis=1)
        rand_pts = rand_pts / norms[:, np.newaxis]
        initializations.append(rand_pts)
        
        # 3. Random points in cube
        if i % 2 == 0:
            rand_pts_cube = np.random.rand(n, d) * 2 - 1
            initializations.append(rand_pts_cube)

    best_points_overall = initializations[0].copy()
    best_score_overall = fast_score(best_points_overall)
    
    # Phase 1: Broad search with multiple initializations
    # Allocate time: 20s per init for first 10 inits (200s), then polish
    # Total time ~600s.
    # Let's do 10 inits * 20s = 200s.
    # Then take the best and polish for the remaining 400s.
    
    num_inits_to_run = 10
    time_per_init = 20.0
    
    for init_idx in range(min(num_inits_to_run, len(initializations))):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 50: # Leave 50s for final polish
            break
            
        init_pts = initializations[init_idx]
        
        # Run SA for this init
        local_best_pts, local_best_score = sa_optimize(init_pts, time_per_init)
        
        if local_best_score > best_score_overall:
            best_score_overall = local_best_score
            best_points_overall = local_best_pts.copy()

    # Phase 2: Final high-precision polish on the best point found
    elapsed = time.time() - start_time
    remaining_time = time_limit - 20 - elapsed # 20s safety margin
    if remaining_time > 10:
        best_points_overall, best_score_overall = polish(best_points_overall, remaining_time)

    # Final verification with scipy
    final_score = get_score(best_points_overall)
    if final_score > best_score_overall:
        best_score_overall = final_score
    else:
        # If fast_score was optimistic, use the scipy score for safety?
        # The task says "Invalid: ... or an exception".
        # Returning the points is what matters. The score is measured by evaluator.
        pass

    return best_points_overall

# EVOLVE-BLOCK-END
