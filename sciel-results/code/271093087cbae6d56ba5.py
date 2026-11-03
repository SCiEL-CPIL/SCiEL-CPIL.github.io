"""Pack 200 equal circles in a unit square to maximize common radius."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200
HALF = 0.5


def _score(centers):
    """Compute the radius for a given set of centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check inside square
    if np.any(np.abs(centers) > HALF + 1e-12):
        return 0.0
    # Wall distance
    wall_min = np.min(HALF - np.abs(centers))
    # Pairwise distance using KDTree
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0)
    if len(pairs) == 0:
        pair_min = np.inf
    else:
        d = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            d[i] = np.linalg.norm(centers[a] - centers[b])
        pair_min = np.min(d)
    r = min(wall_min, pair_min / 2.0)
    return r


def _bottleneck_info(centers):
    """
    Identify the bottleneck constraint (wall or pair).
    Returns:
        r: current radius
        type: 'wall' or 'pair'
        idx: index of point (for wall) or tuple (i, j) (for pair)
    """
    wall_dists = HALF - np.abs(centers)
    min_wall_idx = np.argmin(wall_dists)
    min_wall = wall_dists.flat[min_wall_idx]
    
    tree = cKDTree(centers)
    pairs = tree.query_pairs(2.0)
    if len(pairs) == 0:
        min_pair = np.inf
        min_pair_idx = None
    else:
        d = np.zeros(len(pairs))
        for i, (a, b) in enumerate(pairs):
            d[i] = np.linalg.norm(centers[a] - centers[b])
        min_pair_idx_local = np.argmin(d)
        min_pair = d[min_pair_idx_local]
        min_pair_idx = pairs[min_pair_idx_local]
    
    if min_wall < min_pair / 2.0:
        point_idx = min_wall_idx // 2
        return min_wall, 'wall', point_idx
    else:
        return min_pair / 2.0, 'pair', min_pair_idx


def _hex_grid_init():
    """Initialize with a hexagonal close-packed pattern."""
    best_points = None
    best_score = 0.0
    
    for s in np.linspace(0.03, 0.15, 50):
        dx = s
        dy = s * np.sqrt(3) / 2
        points = []
        row = 0
        while len(points) < N:
            offset = 0 if row % 2 == 0 else dx / 2
            x = -HALF + dx / 2 + offset
            while x <= HALF - dx / 2 + 1e-9:
                if len(points) < N:
                    points.append([x, -HALF + dy / 2 + row * dy])
                x += dx
            row += 1
            if row * dy > 1.0:
                break
        
        if len(points) >= N:
            pts = np.array(points[:N])
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_points = pts
    
    if best_points is not None:
        return best_points
    
    # Fallback: square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _bottleneck_sa(centers, time_budget=60.0):
    """
    Bottleneck elimination with simulated annealing.
    Targets the most constrained points and applies targeted perturbations.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    np.random.seed(42)
    
    T = 0.005
    T_min = 1e-7
    cooling_rate = 0.999
    
    while T > T_min and time.time() < deadline - 2.0:
        # Identify bottleneck
        current_r, btype, idx = _bottleneck_info(centers)
        
        if current_r > best_score:
            best_score = current_r
            best_centers = centers.copy()
        
        # Generate perturbation based on bottleneck type
        if btype == 'pair':
            i, j = idx
            vec = centers[i] - centers[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                vec = np.random.randn(2)
                dist = np.linalg.norm(vec)
            vec = vec / dist
            
            # Kick the pair apart with some noise
            kick = T * 0.3
            noise = T * 0.7
            old_i = centers[i].copy()
            old_j = centers[j].copy()
            centers[i] += vec * kick + np.random.randn(2) * noise
            centers[j] -= vec * kick + np.random.randn(2) * noise
            changed_indices = [i, j]
        else:  # wall
            k = idx
            x, y = centers[k]
            dx = HALF - abs(x)
            dy = HALF - abs(y)
            
            if dx < dy:
                if x > 0:
                    vec = np.array([-1.0, 0.0])
                else:
                    vec = np.array([1.0, 0.0])
            else:
                if y > 0:
                    vec = np.array([0.0, -1.0])
                else:
                    vec = np.array([0.0, 1.0])
            
            kick = T * 0.3
            noise = T * 0.7
            old_k = centers[k].copy()
            centers[k] += vec * kick + np.random.randn(2) * noise
            changed_indices = [k]
        
        # Clamp to square
        centers = np.clip(centers, -HALF, HALF)
        
        new_score = _score(centers)
        
        # Accept if better or with SA probability
        if new_score >= best_score:
            pass  # Accept
        else:
            delta = new_score - best_score
            if np.random.rand() < math.exp(delta / T):
                pass  # Accept
            else:
                # Reject: revert changes
                for idx_c in changed_indices:
                    if btype == 'pair':
                        if idx_c == i:
                            centers[i] = old_i
                        else:
                            centers[j] = old_j
                    else:
                        centers[k] = old_k
                # Re-clamp after revert
                centers = np.clip(centers, -HALF, HALF)
        
        T *= cooling_rate
    
    return best_centers, best_score


def _refine_bottlenecks(centers, time_budget=40.0):
    """
    Refine by iteratively fixing the worst constraints.
    For each bottleneck, try multiple random perturbations and keep the best.
    """
    centers = centers.copy()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    start_time = time.time()
    deadline = start_time + time_budget
    
    np.random.seed(123)
    
    while time.time() < deadline - 1.0:
        current_r, btype, idx = _bottleneck_info(centers)
        
        if current_r > best_score:
            best_score = current_r
            best_centers = centers.copy()
        
        # Try multiple perturbations for the bottleneck
        best_trial = centers.copy()
        best_trial_score = current_r
        
        n_trials = 50
        for t in range(n_trials):
            trial = centers.copy()
            
            if btype == 'pair':
                i, j = idx
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    vec = np.random.randn(2)
                    dist = np.linalg.norm(vec)
                vec = vec / dist
                
                # Random perturbation magnitude
                mag = np.random.uniform(0.001, 0.01)
                noise = np.random.uniform(0.0005, 0.005)
                trial[i] += vec * mag + np.random.randn(2) * noise
                trial[j] -= vec * mag + np.random.randn(2) * noise
            else:  # wall
                k = idx
                x, y = centers[k]
                dx = HALF - abs(x)
                dy = HALF - abs(y)
                
                if dx < dy:
                    if x > 0:
                        vec = np.array([-1.0, 0.0])
                    else:
                        vec = np.array([1.0, 0.0])
                else:
                    if y > 0:
                        vec = np.array([0.0, -1.0])
                    else:
                        vec = np.array([0.0, 1.0])
                
                mag = np.random.uniform(0.001, 0.01)
                noise = np.random.uniform(0.0005, 0.005)
                trial[k] += vec * mag + np.random.randn(2) * noise
            
            trial = np.clip(trial, -HALF, HALF)
            trial_score = _score(trial)
            
            if trial_score > best_trial_score:
                best_trial_score = trial_score
                best_trial = trial.copy()
        
        if best_trial_score > current_r:
            centers = best_trial
            if best_trial_score > best_score:
                best_score = best_trial_score
                best_centers = centers.copy()
        else:
            # No improvement, break if stuck
            break
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    total_budget = 140.0  # Leave 10s margin before 150s kill
    
    # Step 1: Hex grid initialization
    centers = _hex_grid_init()
    best_centers = centers.copy()
    best_score = _score(centers)
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Step 2: Bottleneck SA for major improvements
    if remaining > 30:
        time_budget_sa = min(remaining - 15, 70)
        best_centers, best_score = _bottleneck_sa(
            best_centers, time_budget=time_budget_sa
        )
    
    elapsed = time.time() - start_time
    remaining = total_budget - elapsed
    
    # Step 3: Refine bottlenecks for final polish
    if remaining > 15:
        time_budget_refine = min(remaining - 5, 40)
        best_centers, best_score = _refine_bottlenecks(
            best_centers, time_budget=time_budget_refine
        )
    
    # Final validation
    final_score = _score(best_centers)
    if final_score <= 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
