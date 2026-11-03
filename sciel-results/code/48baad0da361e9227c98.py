import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
import random

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
    # We can vectorize this for speed
    # Indices for all triples
    min_area = float('inf')
    
    # Vectorized approach
    # i, j, k where i < j < k
    # Number of triples is C(13, 3) = 286.
    # Vectorizing might be overkill and memory heavy if not careful, but 286 is small.
    # Let's stick to a loop but optimized with numpy ops if possible, or just pure python if fast enough.
    # 286 iterations is very fast.
    
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v1 = p2 - p1
            for k in range(j + 1, n):
                p3 = points[k]
                v2 = p3 - p1
                # Area is 0.5 * |cross product|
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
                    if min_area < 1e-10:
                        return 0.0
    
    if min_area < 1e-10:
        return 0.0
        
    return min_area / hull_area

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
        # 4x4 grid minus 3 points
        coords = [0.0, 1.0/3, 2.0/3, 1.0]
        grid_points = []
        for x in coords:
            for y in coords:
                grid_points.append([x, y])
        # Remove 3 random points to get 13
        # Or specific ones to maintain symmetry?
        # Let's just take the first 13 for simplicity in init
        return np.array(grid_points[:13])
    elif method == 'random':
        return np.random.rand(N, 2)
    else:
        return np.random.rand(N, 2)

def local_search(points, max_iters=2000, time_limit=10.0, start_temp=0.1, cool_rate=0.95):
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
            # Gaussian noise
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
            
        # Cool down
        temp *= cool_rate
        
        # If stuck for a long time, kick
        if no_improve_count > 100:
             kick_idx = random.randint(0, N - 1)
             new_points = current_points.copy()
             new_points[kick_idx] = current_points[kick_idx] + np.random.normal(0, 0.2, 2)
             new_score = compute_score(new_points)
             if new_score > current_score:
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
                 no_improve_count = 0
                 temp = 0.1 # Reheat
             else:
                 # Just cool faster
                 temp *= 0.9

    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_global_points = None
    best_global_score = 0.0
    
    # List of strategies
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
    
    # Shuffle to ensure variety
    random.shuffle(strategies)
    
    num_strategies = len(strategies)
    
    for i, strat in enumerate(strategies):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on remaining strategies
        remaining = num_strategies - i
        time_per_strat = max(2.0, (deadline - time.time() - 5.0) / remaining)
        
        # Generate initial points
        init_pts = generate_initialization(strat)
        
        # Run local search
        pts, score = local_search(init_pts, time_limit=time_per_strat)
        
        if score > best_global_score:
            best_global_score = score
            best_global_points = pts.copy()
            
    # If we still have time, do a final refinement on the best point
    if best_global_points is not None and time.time() < deadline - 10.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 1.0:
             final_pts, final_score = local_search(best_global_points, time_limit=remaining_time, start_temp=0.01, cool_rate=0.99)
             if final_score > best_global_score:
                 best_global_score = final_score
                 best_global_points = final_pts

    if best_global_points is None:
        # Fallback
        best_global_points = generate_initialization('polygon')
        
    return best_global_points
# EVOLVE-BLOCK-END
