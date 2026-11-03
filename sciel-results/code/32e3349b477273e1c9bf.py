import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

def _min_triangle_area_vec(points):
    """
    Vectorized calculation of the minimum area of any triangle formed by N points.
    points: shape (N, 2)
    """
    # points: (N, 2)
    # We want areas for all i < j < k
    # Number of combinations: N*(N-1)*(N-2)/6
    # For N=11, 165 triangles.
    
    # Generate indices for i, j, k
    # Using np.triu_indices is efficient for pairs, but we need triples.
    # Simpler: loop over i, then vectorize j, k? Or full broadcasting?
    # Full broadcasting of (N, N, N) is 1331 entries, which is small.
    # Let's use full broadcasting and mask i<j<k.
    
    p = points
    # x1, y1, x2, y2, x3, y3
    # p[:, np.newaxis, np.newaxis] -> (N, 1, 1)
    # p[np.newaxis, :, np.newaxis] -> (1, N, 1)
    # p[np.newaxis, np.newaxis, :] -> (1, 1, N)
    
    x1, y1 = p[:, 0][:, np.newaxis, np.newaxis], p[:, 1][:, np.newaxis, np.newaxis]
    x2, y2 = p[:, 0][np.newaxis, :, np.newaxis], p[:, 1][np.newaxis, :, np.newaxis]
    x3, y3 = p[:, 0][np.newaxis, np.newaxis, :], p[:, 1][np.newaxis, np.newaxis, :]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |
    # Note: (x3-x1) is x3 - x1, (y3-y1) is y3 - y1
    cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    areas = 0.5 * np.abs(cross)
    
    # Mask: only i < j < k
    # i is axis 0, j is axis 1, k is axis 2
    idx = np.arange(N)
    mask = (idx[:, np.newaxis, np.newaxis] < idx[np.newaxis, :, np.newaxis]) & \
           (idx[np.newaxis, :, np.newaxis] < idx[np.newaxis, np.newaxis, :])
    
    # Extract valid areas
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
        
    return np.min(valid_areas)

def _is_inside_triangle_vec(points):
    """
    Check if all points are inside the triangle.
    Returns a boolean scalar.
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # 1. x in [0, 1]
    # 2. y in [0, H]
    # 3. y <= -SQRT3 * (x - 1)  => y <= SQRT3 * (1 - x)
    
    cond1 = (x >= 0) & (x <= 1)
    cond2 = (y >= 0) & (y <= H + 1e-9)
    cond3 = y <= SQRT3 * (1 - x) + 1e-9
    
    return np.all(cond1 & cond2 & cond3)

def _penalized_objective(points):
    """
    Objective for minimization.
    Returns -min_area + penalty if outside.
    """
    if not _is_inside_triangle_vec(points):
        # Calculate distance outside to create a smooth penalty?
        # For simplicity in SA, a large constant penalty is often sufficient if steps are small.
        # But let's use a penalty proportional to violation to help SA push back.
        x = points[:, 0]
        y = points[:, 1]
        
        # Distance to edges
        # Edge 1: x=0 -> dist = max(0, -x)
        # Edge 2: x=1 -> dist = max(0, x-1)
        # Edge 3: y=0 -> dist = max(0, -y)
        # Edge 4: y=H -> dist = max(0, y-H)
        # Edge 5: y = SQRT3(1-x) -> line: SQRT3 x + y - SQRT3 = 0
        # dist = |SQRT3 x + y - SQRT3| / sqrt(3+1) = |...|/2
        # Inside means SQRT3 x + y - SQRT3 <= 0. So violation is max(0, SQRT3 x + y - SQRT3)/2
        
        d1 = np.maximum(0, -x)
        d2 = np.maximum(0, x - 1)
        d3 = np.maximum(0, -y)
        d4 = np.maximum(0, y - H)
        d5 = np.maximum(0, (SQRT3 * x + y - SQRT3) / 2.0)
        
        penalty = np.sum(d1**2 + d2**2 + d3**2 + d4**2 + d5**2) * 100.0
        
    min_area = _min_triangle_area_vec(points)
    return -min_area + penalty

def construct():
    start_time = time.time()
    deadline = 110.0  # 150s limit, keep 40s margin for safety/overhead
    
    # Initial Heuristic:
    # Start with a configuration that is known to be decent.
    # For N=11, a grid-like distribution or concentric rings might be good.
    # Let's try a distorted grid or a known best approximation for smaller N scaled up.
    # Or just use a random start but with a few restarts in SA.
    
    # We will run Simulated Annealing.
    
    def simulated_annealing(points_init, max_iters, T_start, T_end, alpha):
        """
        points_init: (N, 2) array
        Returns best_points, best_score
        """
        current_points = points_init.copy()
        current_score = _penalized_objective(current_points)
        
        best_points = current_points.copy()
        best_score = current_score
        
        # Precompute step size
        # Typical step size for coordinates in [0,1] is around 0.01-0.05
        step_size = 0.05
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Generate neighbor
            # Perturb all points or random subset? All points is standard for this dim.
            # Random Gaussian noise
            noise = np.random.normal(0, step_size, size=(N, 2))
            next_points = current_points + noise
            
            # Optional: Clamp to bounding box [0,1]x[0,H] to avoid huge penalties?
            # The penalty handles it, but clamping might speed up convergence.
            # Let's rely on penalty for now to avoid discontinuities in gradient.
            
            next_score = _penalized_objective(next_points)
            
            # Metropolis criterion
            if next_score < current_score:
                # Improvement
                current_points = next_points
                current_score = next_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                # Worse
                delta = next_score - current_score
                # P = exp(-delta / T)
                # If T is very small, exp(-large) -> 0
                if np.random.random() < np.exp(-delta / T):
                    current_points = next_points
                    current_score = next_score
            
            # Adapt step size?
            # If we are stuck (no improvement for a while), we could increase step size.
            # But simple SA often works well with fixed or slowly decaying step size.
            # Let's decrease step size slightly over time to allow fine-tuning.
            if i % 1000 == 0:
                step_size *= 0.95
        
        return best_points, best_score

    # Run SA with a few restarts
    best_global_points = None
    best_global_score = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Initial points:
    # 1. Centroid + Ring
    # 2. Random
    # 3. Grid
    
    initial_candidates = []
    
    # Candidate 1: Center + Rings
    cx, cy = 0.5, H/3
    r1 = 0.2
    r2 = 0.35
    pts1 = np.array([[cx, cy]])
    angles1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    pts1 = np.vstack([pts1, [[cx + r1*np.cos(a), cy + r1*np.sin(a)] for a in angles1]])
    angles2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    pts1 = np.vstack([pts1, [[cx + r2*np.cos(a), cy + r2*np.sin(a)] for a in angles2]])
    # Check bounds and adjust if needed, SA will fix it.
    initial_candidates.append(pts1)
    
    # Candidate 2: Random
    pts2 = np.random.uniform(0.1, 0.9, size=(N, 2))
    # Ensure inside triangle roughly
    # Simple rejection sampling for init
    valid_pts = []
    while len(valid_pts) < N:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, H)
        if y <= SQRT3 * (1 - x):
            valid_pts.append([x, y])
    initial_candidates.append(np.array(valid_pts))
    
    # Candidate 3: Distorted Grid
    # 3x4 grid approx
    xs = np.linspace(0.2, 0.8, 3)
    ys = np.linspace(0.2, 0.6, 4)
    pts3 = []
    for x in xs:
        for y in ys:
            if y <= SQRT3 * (1 - x):
                pts3.append([x, y])
    # If we have less than 11, add randoms
    while len(pts3) < N:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.5)
        if y <= SQRT3 * (1 - x):
            pts3.append([x, y])
    initial_candidates.append(np.array(pts3[:N]))
    
    # Run SA for each candidate
    max_iters = 20000
    T_start = 0.1
    T_end = 0.001
    
    for init_pts in initial_candidates:
        if time.time() > deadline - 5:
            break
            
        best_pts, best_score = simulated_annealing(init_pts, max_iters, T_start, T_end, 0.995)
        
        # Un-penalize score to get area
        # Note: best_score is -area + penalty. If penalty is 0, area = -best_score.
        # We should verify it's inside.
        if _is_inside_triangle_vec(best_pts):
            area = _min_triangle_area_vec(best_pts)
            if area > (best_global_score if best_global_points is not None else 0):
                # Wait, best_score is MINIMIZED. So lower is better.
                # Area is MAXIMIZED. So higher is better.
                # best_score = -area.
                # So if -best_score > current_best_area, update.
                if -best_score > (best_global_score if best_global_points is not None else 0):
                     # Correction: best_global_score stores the PENALTY-LESS NEGATIVE AREA? 
                     # Let's just store the points and re-evaluate area at the end for clarity.
                     if best_global_points is None or -best_score > -best_global_score_val:
                         best_global_points = best_pts
                         best_global_score_val = best_score
        else:
            # If SA failed to keep it inside, try to project or discard
            pass

    if best_global_points is None:
        # Fallback: Return the best from the first candidate's raw SA output, projected
        # Or just a simple heuristic
        best_global_points = initial_candidates[0]
        
    # Final Polish with Nelder-Mead
    if time.time() < deadline - 10:
        def nm_obj(p):
            pts = p.reshape((N, 2))
            return _penalized_objective(pts)
        
        try:
            res = minimize(nm_obj, best_global_points.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
            if res.fun < best_global_score_val:
                best_global_points = res.x.reshape((N, 2))
        except Exception:
            pass
            
    # Final check and return
    if not _is_inside_triangle_vec(best_global_points):
        # Project points inside triangle
        # Simple projection: clip x, y and handle slant
        x = best_global_points[:, 0]
        y = best_global_points[:, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, H)
        # Slant: y <= SQRT3(1-x)
        # If violated, move to edge
        # This is a rough projection
        for i in range(N):
            if best_global_points[i, 1] > SQRT3 * (1 - best_global_points[i, 0]):
                # Project onto line y = SQRT3(1-x)
                # Normal vector (SQRT3, 1). 
                # Just clamp y to the line value if x is valid
                best_global_points[i, 1] = SQRT3 * (1 - best_global_points[i, 0])
                # Ensure y >= 0
                if best_global_points[i, 1] < 0:
                    best_global_points[i, 1] = 0
                    best_global_points[i, 0] = 1.0 # Corner?
        
    return best_global_points

# EVOLVE-BLOCK-END
