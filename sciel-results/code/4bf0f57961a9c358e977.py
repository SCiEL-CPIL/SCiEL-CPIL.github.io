import time
import random
from collections import Counter

def _calculate_score(points):
    """
    Calculate the maximum multiplicity of any non-zero doubled area.
    points: list of (x, y) tuples
    """
    n = len(points)
    area_counts = Counter()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx1 = x2 - x1
            dy1 = y2 - y1
            for k in range(j + 1, n):
                x3, y3 = points[k]
                dx2 = x3 - x1
                dy2 = y3 - y1
                area = abs(dx1 * dy2 - dy1 * dx2)
                if area > 0:
                    area_counts[area] += 1
    
    if not area_counts:
        return 0
    return max(area_counts.values())

def _local_search(initial_points, time_limit=55.0):
    """
    Perform hill climbing to maximize the score.
    """
    start_time = time.time()
    best_points = [p for p in initial_points]
    best_score = _calculate_score(best_points)
    
    current_points = [p for p in initial_points]
    current_score = best_score
    
    # Precompute valid coordinates
    coords = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    
    # Helper to get random empty point
    def get_random_empty_point(occupied_set):
        while True:
            p = random.choice(coords)
            if p not in occupied_set:
                return p

    occupied_set = set(current_points)
    
    stuck_count = 0
    max_stuck = 1000
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_p = current_points[idx]
        
        # Get a new random position
        new_p = get_random_empty_point(occupied_set)
        
        # Apply move
        current_points[idx] = new_p
        occupied_set.remove(old_p)
        occupied_set.add(new_p)
        
        # Calculate new score
        new_score = _calculate_score(current_points)
        
        if new_score > current_score:
            current_score = new_score
            stuck_count = 0
            if new_score > best_score:
                best_score = new_score
                best_points = [p for p in current_points]
        else:
            # Revert move
            current_points[idx] = old_p
            occupied_set.remove(new_p)
            occupied_set.add(old_p)
            stuck_count += 1
            
            if stuck_count > max_stuck:
                # Restart from best found so far
                current_points = [p for p in best_points]
                occupied_set = set(current_points)
                current_score = best_score
                stuck_count = 0
                # Optional: randomize the restart slightly
                # For now, simple restart

    return best_points, best_score

def construct():
    # Initial guesses
    # 5x6 grid
    grid_5x6 = [(x, y) for x in range(5) for y in range(6)]
    # 6x5 grid
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    # 3x10 grid
    grid_3x10 = [(x, y) for x in range(3) for y in range(10)]
    # 2x15 grid
    grid_2x15 = [(x, y) for x in range(2) for y in range(15)]
    
    # Random initialization
    random.seed(42)
    random_points = random.sample([(x, y) for x in range(-30, 31) for y in range(-30, 31)], 30)
    
    candidates = [grid_5x6, grid_6x5, grid_3x10, grid_2x15, random_points]
    
    best_overall_points = None
    best_overall_score = -1
    
    # Evaluate candidates
    for c in candidates:
        s = _calculate_score(c)
        if s > best_overall_score:
            best_overall_score = s
            best_overall_points = c
            
    # Run local search from the best candidate
    # We have ~55 seconds for search
    # We can run multiple independent searches or one long one.
    # One long search with restarts is usually sufficient for this size.
    
    # To be safe and explore different basins, we can split time among top candidates.
    # But typically the grid is a strong basin.
    
    final_points, final_score = _local_search(best_overall_points, time_limit=50.0)
    
    if final_score > best_overall_score:
        return final_points
    else:
        return best_overall_points

# EVOLVE-BLOCK-END
