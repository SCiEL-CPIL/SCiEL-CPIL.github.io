# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the ratio of the smallest triangle area
    to the convex hull area using Simulated Annealing.
    """
    n = 13
    start_time = time.time()
    time_limit = 590.0  # Leave 10s margin for safety
    
    # Precompute triangle indices for fast evaluation
    idx = np.arange(n)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_i = i_idx[mask]
    tri_j = j_idx[mask]
    tri_k = k_idx[mask]

    def get_areas(points):
        """Compute areas of all 286 triangles efficiently."""
        p_i = points[tri_i]
        p_j = points[tri_j]
        p_k = points[tri_k]
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def get_hull_area(points):
        """Compute area of the convex hull."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            return 0.0

    def score(points):
        """Compute the Heilbronn score: A_min / A_hull."""
        if not np.all(np.isfinite(points)):
            return 0.0
        areas = get_areas(points)
        if len(areas) == 0:
            return 0.0
        a_min = np.min(areas)
        if a_min < 1e-15:
            return 0.0
        a_hull = get_hull_area(points)
        if a_hull < 1e-15:
            return 0.0
        return a_min / a_hull

    def generate_initialization(seed):
        """Generate diverse initial configurations."""
        rng = np.random.default_rng(seed)
        
        # 1. Regular 13-gon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts1 = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # 2. Random points in a disk
        r = np.sqrt(rng.uniform(0, 1, n))
        theta = rng.uniform(0, 2 * np.pi, n)
        pts2 = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
        # 3. Farthest Point Sampling (Greedy)
        pts3 = np.zeros((n, 2))
        # Start with a random point
        pts3[0] = rng.uniform(-1, 1, 2)
        for i in range(1, n):
            # Compute min distance to existing points for all points on a grid
            # Or simply pick a random point and keep the one with max min-dist
            best_p = None
            best_min_dist = -1
            # Try 100 random candidates
            for _ in range(100):
                cand = rng.uniform(-1, 1, 2)
                dists = np.linalg.norm(pts3[:i] - cand, axis=1)
                min_dist = np.min(dists)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_p = cand
            pts3[i] = best_p
        
        # 4. Jittered Regular Polygon
        pts4 = pts1 + rng.normal(0, 0.2, (n, 2))
        
        # 5. Square Grid subset
        grid_x = np.linspace(-1, 1, 4)
        grid_y = np.linspace(-1, 1, 4)
        gx, gy = np.meshgrid(grid_x, grid_y)
        grid_pts = np.column_stack((gx.ravel(), gy.ravel()))
        # Shuffle and take 13
        rng.shuffle(grid_pts)
        pts5 = grid_pts[:13]

        inits = [pts1, pts2, pts3, pts4, pts5]
        return inits

    best_global_params = None
    best_global_score = 0.0

    # Run SA for a few different initializations
    # We allocate time based on remaining budget
    num_inits = 5
    time_per_init = (time_limit - 30.0) / num_inits # Leave 30s for final polish

    for i in range(num_inits):
        if time.time() - start_time > time_limit - 30.0:
            break
            
        inits = generate_initialization(seed=42 + i)
        init_pts = inits[i % len(inits)]
        
        # Simulated Annealing
        current_pts = init_pts.copy()
        current_score = score(current_pts)
        
        best_pts = current_pts.copy()
        best_score = current_score
        
        T = 0.1  # Initial temperature
        T_min = 1e-6
        alpha = 0.995  # Cooling rate
        max_iter = 20000
        step_size = 0.1
        
        # Dynamic step size adjustment based on score improvement
        # We want to explore widely initially, then refine
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 30.0:
                break
                
            # Generate neighbor
            # Jitter a random subset of points (e.g., 2-4 points) to allow local moves
            num_move = np.random.randint(1, 4)
            move_indices = np.random.choice(n, num_move, replace=False)
            
            new_pts = current_pts.copy()
            # Gaussian jitter
            jitter = np.random.normal(0, step_size, (num_move, 2))
            new_pts[move_indices] += jitter
            
            # Optional: Normalize to prevent drift
            # new_pts = (new_pts - np.mean(new_pts)) / (np.std(new_pts) + 1e-9)
            
            new_score = score(new_pts)
            
            # Metropolis acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                # Accept if better
                current_pts = new_pts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
            else:
                # Accept if worse with probability exp(-delta / T)
                if T > T_min:
                    prob = np.exp(-delta / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
            
            # Cool down
            T *= alpha
            
            # Adaptive step size: if we haven't improved best score in a while, increase step size?
            # Or decrease step size as T decreases to refine.
            # Simple heuristic: step_size scales with T somewhat, but has a floor.
            # Actually, for SA, step size is often fixed or slowly decaying.
            # Let's decay step size slightly to refine local optima.
            step_size *= 0.9995
            if step_size < 0.001:
                step_size = 0.001

        # Update global best
        if best_score > best_global_score:
            best_global_score = best_score
            best_global_params = best_pts.copy()

    # Final Polish: Use L-BFGS-B on the best result found by SA
    # L-BFGS-B can refine the local optimum much faster than SA in the end.
    if time.time() - start_time < time_limit - 10.0 and best_global_params is not None:
        
        def neg_hard_objective(params):
            points = params.reshape((n, 2))
            if not np.all(np.isfinite(points)):
                return 1e10
            s = score(points)
            if s < 1e-15:
                return 1e10
            return -s

        try:
            res = minimize(
                neg_hard_objective,
                best_global_params.flatten(),
                method='Nelder-Mead', # Nelder-Mead is robust for non-smooth functions
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            final_points = res.x.reshape((n, 2))
            final_score = score(final_points)
            
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_params = final_points.copy()
                
        except Exception:
            pass

    # Ensure valid output
    if best_global_params is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_global_params = np.column_stack((np.cos(angles), np.sin(angles))).flatten()

    final_points = best_global_params.reshape((n, 2))
    
    if not np.all(np.isfinite(final_points)):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_points = np.column_stack((np.cos(angles), np.sin(angles)))

    try:
        hull = ConvexHull(final_points)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return final_points

# EVOLVE-BLOCK-END
