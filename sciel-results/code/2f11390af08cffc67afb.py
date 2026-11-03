import time
import numpy as np
from scipy.spatial import ConvexHull
import random

N = 13

# EVOLVE-BLOCK-START

def compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Calculate all triangle areas
    min_area = float('inf')
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

def compute_soft_min_score(points, temperature=0.01):
    """
    Compute a smooth surrogate for the min triangle area using soft-min.
    This provides gradients for optimization.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Collect all triangle areas
    areas = []
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    
    if not areas:
        return 0.0
    
    areas = np.array(areas)
    
    # Soft-min: -log(sum(exp(-areas/T)))/T
    # This is differentiable and approximates min when T is small
    scaled = -areas / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -100, 100)
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = -log_sum * temperature
    
    return soft_min / hull_area

def generate_initialization(method='polygon'):
    """
    Generate various initial configurations.
    """
    if method == 'polygon':
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        radius = 0.5
        center = [0.5, 0.5]
        points = np.array([[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)] for t in theta])
        return points
    elif method == 'halton':
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        points = []
        for i in range(1, N + 1):
            x = halton(i, 2)
            y = halton(i, 3)
            points.append([x, y])
        return np.array(points)
    elif method == 'grid':
        coords = [0.0, 1.0/3, 2.0/3, 1.0]
        grid_points = []
        for x in coords:
            for y in coords:
                grid_points.append([x, y])
        return np.array(grid_points[:13])
    elif method == 'random':
        return np.random.rand(N, 2)
    else:
        return np.random.rand(N, 2)

def local_search(points, max_iters=5000, time_limit=10.0, start_temp=0.1, cool_rate=0.98):
    """
    Simulated Annealing to maximize the score.
    """
    current_points = points.copy()
    current_score = compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    temp = start_temp
    start_time = time.time()
    no_improve_count = 0
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Perturbation: move 1 or 2 points
        n_moves = random.randint(1, 2)
        indices = random.sample(range(N), n_moves)
        
        new_points = current_points.copy()
        for idx in indices:
            noise = np.random.normal(0, 0.05, 2)
            new_points[idx] = current_points[idx] + noise
            
        new_score = compute_score(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / temp):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        temp *= cool_rate
        
        if no_improve_count > 100:
             kick_idx = random.randint(0, N - 1)
             new_points = current_points.copy()
             new_points[kick_idx] = current_points[kick_idx] + np.random.normal(0, 0.2, 2)
             new_score = compute_score(new_points)
             if new_score > current_score:
                 current_points = new_points
                 current_score = new_score
                 if new_score > best_score:
                     best_score = new_score
                     best_points = current_points.copy()
                 no_improve_count = 0
                 temp = 0.1
             else:
                 temp *= 0.9

    return best_points, best_score

def polish_with_soft_min(points, time_limit=5.0, initial_temp=0.01):
    """
    High-precision polish using gradient-free optimization on the smooth surrogate.
    Uses Nelder-Mead style simplex on the flat parameterization.
    """
    start_time = time.time()
    best_points = points.copy()
    best_score = compute_score(best_points)
    
    # Use a smaller temperature for the soft-min to be closer to true min
    soft_temp = initial_temp
    
    # Nelder-Mead style optimization on the soft-min surrogate
    # We'll use a simple coordinate descent + adaptive step size
    
    current_points = points.copy()
    current_soft_score = compute_soft_min_score(current_points, soft_temp)
    
    step_size = 0.01
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        # Try moving each point in each direction
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            for dim in range(2):
                if time.time() - start_time > time_limit:
                    break
                for direction in [-1, 1]:
                    new_points = current_points.copy()
                    new_points[i, dim] += direction * step_size
                    
                    new_soft_score = compute_soft_min_score(new_points, soft_temp)
                    
                    if new_soft_score > current_soft_score:
                        current_points = new_points
                        current_soft_score = new_soft_score
                        improved = True
                        no_improve = 0
                    else:
                        no_improve += 1
                        # Reduce step size if stuck
                        if no_improve > 100:
                            step_size *= 0.7
                            no_improve = 0
                            # Check actual score
                            actual_score = compute_score(current_points)
                            if actual_score > best_score:
                                best_score = actual_score
                                best_points = current_points.copy()
        
        if not improved and step_size < 1e-6:
            break
        
        # Periodically check the true score
        if no_improve % 50 == 0:
            actual_score = compute_score(current_points)
            if actual_score > best_score:
                best_score = actual_score
                best_points = current_points.copy()
    
    # Final check
    final_score = compute_score(current_points)
    if final_score > best_score:
        best_score = final_score
        best_points = current_points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_global_points = None
    best_global_score = 0.0
    
    strategies = [
        'polygon',
        'halton',
        'grid',
        'random',
        'random',
        'random',
        'polygon',
        'halton'
    ]
    
    random.shuffle(strategies)
    
    num_strategies = len(strategies)
    
    for i, strat in enumerate(strategies):
        if time.time() > deadline - 15.0:
            break
            
        remaining = num_strategies - i
        time_per_strat = max(2.0, (deadline - time.time() - 15.0) / remaining)
        
        init_pts = generate_initialization(strat)
        
        # First run standard SA
        pts, score = local_search(init_pts, time_limit=time_per_strat * 0.7)
        
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
            
    # Final polish phase using the smooth surrogate
    if best_global_points is not None and time.time() < deadline - 5.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 1.0:
            polished_pts, polished_score = polish_with_soft_min(
                best_global_points, 
                time_limit=remaining_time,
                initial_temp=0.005
            )
            if polished_score > best_global_score:
                best_global_score = polished_score
                best_global_points = polished_pts
    
    if best_global_points is None:
        best_global_points = generate_initialization('polygon')
        
    return best_global_points
# EVOLVE-BLOCK-END
