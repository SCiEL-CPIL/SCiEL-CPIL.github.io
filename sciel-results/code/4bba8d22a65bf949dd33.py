# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin of 20 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def soft_min_max_score(points, temp=0.05):
        """
        Smooth surrogate for the score.
        Uses soft-min and soft-max of distances to make the objective differentiable.
        """
        d = pdist(points)
        if d.size == 0:
            return 0.0
        
        # Soft min: log-sum-exp trick for min
        # min(d) = -log(sum(exp(-d/temp))) * temp
        d_min_soft = -temp * np.log(np.sum(np.exp(-d / temp)))
        
        # Soft max: log-sum-exp trick for max
        # max(d) = log(sum(exp(d/temp))) * temp
        d_max_soft = temp * np.log(np.sum(np.exp(d / temp)))
        
        if d_max_soft <= 0:
            return 0.0
        
        ratio = d_min_soft / d_max_soft
        return ratio * ratio

    def objective_soft(points_flat, temp=0.05):
        points = points_flat.reshape(16, 2)
        # We want to maximize score, so minimize negative score
        score = soft_min_max_score(points, temp)
        if not np.isfinite(score):
            return 1e10
        return -score

    best_points = None
    best_score = -np.inf

    n = 16
    d = 2

    def update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        points = np.array(points, dtype=float)
        if points.shape != (16, 2) or not np.all(np.isfinite(points)):
            return
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Generate diverse initial candidates based on known topological structures
    candidates = []

    # 1. Regular 4x4 grid (distorted)
    # The optimal 16-point packing in a square is often a distorted grid
    # We generate a few variations of the grid with radial scaling
    base_grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(base_grid)
    
    # Radial distortion of grid: scale each point by r^alpha
    for alpha in [0.5, 0.8, 1.2, 1.5]:
        r = np.linalg.norm(base_grid, axis=1)
        r_safe = np.where(r == 0, 1e-6, r)
        scaled = base_grid * (r_safe ** (alpha - 1))[:, np.newaxis]
        candidates.append(scaled)

    # 2. Two concentric circles (8 inner, 8 outer)
    # This is a very strong candidate for 16 points
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8  # Staggered
    
    # Try different radius ratios
    for r_ratio in [0.4, 0.5, 0.6, 0.7, 0.8]:
        inner_radius = r_ratio
        outer_radius = 1.0
        inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
        outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
        two_circle = np.vstack([inner_points, outer_points])
        candidates.append(two_circle)

    # 3. Three rings: 4 center, 6 mid, 6 outer? Or 2, 6, 8?
    # Let's try 2 center, 6 mid, 8 outer
    # Center
    p0 = np.array([[0.0, 0.0]])
    # Mid ring
    mid_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    mid_radius = 0.6
    mid_points = np.column_stack([mid_radius * np.cos(mid_angles), mid_radius * np.sin(mid_angles)])
    # Outer ring
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_radius = 1.0
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    three_ring = np.vstack([p0, mid_points, outer_points])
    candidates.append(three_ring)

    # 4. Triangular lattice subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 5. Random points (for diversity)
    for seed in [1, 2, 3, 42, 123]:
        np.random.seed(seed)
        random_points = np.random.randn(n, d)
        random_points = random_points / np.std(random_points)
        candidates.append(random_points)

    # Evaluate initial scores
    for cand in candidates:
        update_best(cand)

    # Sort candidates by score to prioritize optimization
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Optimize the top candidates using BFGS with soft objective
    # BFGS is much faster and more robust than Nelder-Mead for smooth functions
    top_k = min(10, len(scored_candidates))
    
    for i in range(top_k):
        if time.time() > deadline:
            break
        
        score, cand = scored_candidates[i]
        x0 = cand.flatten()
        
        # Try a few different temperatures for the soft objective
        temps = [0.1, 0.05, 0.02]
        
        for temp in temps:
            if time.time() > deadline:
                break
            
            try:
                # Use BFGS with numerical gradient
                result = minimize(
                    objective_soft, 
                    x0, 
                    args=(temp,),
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < -score: # If it improved
                    opt_points = result.x.reshape(16, 2)
                    update_best(opt_points)
                    
                    # If BFGS got stuck or didn't improve much, try Nelder-Mead on the result
                    # with a tighter tolerance to polish
                    if time.time() < deadline - 10:
                        result2 = minimize(
                            lambda x: -compute_score(x.reshape(16, 2)), 
                            result.x, 
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'fatol': 1e-10, 'xatol': 1e-10}
                        )
                        opt_points2 = result2.x.reshape(16, 2)
                        update_best(opt_points2)
                        
            except Exception:
                pass

    # If we still have time, do a few random restarts with the same strategy
    if time.time() < deadline - 60:
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 7 + 13)
            random_points = np.random.randn(n, d)
            random_points = random_points / np.std(random_points)
            
            # Quick BFGS optimization
            try:
                result = minimize(
                    objective_soft, 
                    random_points.flatten(), 
                    args=(0.05,),
                    method='BFGS',
                    options={'maxiter': 100, 'gtol': 1e-5}
                )
                opt_points = result.x.reshape(16, 2)
                update_best(opt_points)
            except Exception:
                pass

    # Final check
    if best_points is None:
        # Fallback to the best initial candidate
        best_points = scored_candidates[0][1]

    return best_points

# EVOLVE-BLOCK-END
