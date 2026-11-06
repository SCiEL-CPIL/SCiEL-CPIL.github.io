"""
Strategy: Multi-Phase Search with Structured Initialization and Targeted SA

Key improvements over the previous version:
1. Multiple diverse initial configurations (grids, random, structured)
2. Targeted simulated annealing that focuses on a specific target area
3. Periodic restarts with different strategies
4. Better cooling schedule
5. Vectorized evaluation for speed
"""
import numpy as np
import time
import random
from collections import Counter

# Pre-compute triple indices for 30 points
def _get_triple_indices(n=30):
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    return i_idx[mask], j_idx[mask], k_idx[mask]

_TRIPLE_I, _TRIPLE_J, _TRIPLE_K = _get_triple_indices(30)

def _vectorized_compute_areas(points):
    """
    Compute absolute doubled areas for all triples of 30 points.
    points: numpy array of shape (30, 2)
    Returns: numpy array of shape (4500,) with absolute determinants.
    """
    pi = points[_TRIPLE_I]
    pj = points[_TRIPLE_J]
    pk = points[_TRIPLE_K]
    
    dx_ji = pj[:, 0] - pi[:, 0]
    dy_ji = pj[:, 1] - pi[:, 1]
    dx_ki = pk[:, 0] - pi[:, 0]
    dy_ki = pk[:, 1] - pi[:, 1]
    
    dets = dx_ji * dy_ki - dy_ji * dx_ki
    return np.abs(dets)

def _get_max_multiplicity(areas):
    """
    Given an array of areas, return (max_count, most_frequent_area).
    Ignores zero areas.
    """
    if len(areas) == 0:
        return 0, 0
    
    non_zero = areas[areas > 0]
    if len(non_zero) == 0:
        return 0, 0
        
    unique_areas, counts = np.unique(non_zero, return_counts=True)
    max_idx = np.argmax(counts)
    return counts[max_idx], unique_areas[max_idx]

def _count_area(areas, target):
    """Count how many times target area appears in areas."""
    return np.sum(areas == target)

def _random_points_in_box(n=30, box=30):
    """Generate n distinct random points in [-box, box]^2."""
    points = set()
    while len(points) < n:
        x = random.randint(-box, box)
        y = random.randint(-box, box)
        points.add((x, y))
    return np.array(list(points), dtype=np.int32)

def _grid_points(rows, cols, offset_x=0, offset_y=0):
    """Generate a grid of points."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            x = i * 2 + offset_x
            y = j * 2 + offset_y
            pts.append((x, y))
    return np.array(pts[:30], dtype=np.int32)

def _structured_points():
    """Generate structured points that might have good area properties."""
    # Try a few structured patterns
    patterns = []
    
    # 6x5 grid
    patterns.append(_grid_points(6, 5, -5, -4))
    
    # 5x6 grid
    patterns.append(_grid_points(5, 6, -4, -5))
    
    # Random
    patterns.append(_random_points_in_box())
    
    # Sparse grid
    pts = []
    for i in range(7):
        for j in range(5):
            if len(pts) < 30:
                pts.append((i * 3 - 9, j * 3 - 6))
    patterns.append(np.array(pts, dtype=np.int32))
    
    return patterns

def _sa_search(points, target_area, deadline, initial_temp=3.0):
    """
    Simulated annealing focused on maximizing count of target_area.
    Returns (best_score, best_points, best_area).
    """
    curr_points = points.copy()
    curr_occupied = set(map(tuple, curr_points.tolist()))
    
    curr_areas = _vectorized_compute_areas(curr_points)
    curr_target_count = _count_area(curr_areas, target_area)
    curr_max_mult, curr_max_area = _get_max_multiplicity(curr_areas)
    
    best_score = max(curr_target_count, curr_max_mult)
    best_points = curr_points.copy()
    best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
    
    temp = initial_temp
    cooling = 0.9998
    min_temp = 0.005
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random point to move
        idx = random.randint(0, 29)
        old_x, old_y = curr_points[idx]
        
        # Generate new position
        if random.random() < 0.4:
            new_x = random.randint(-30, 30)
            new_y = random.randint(-30, 30)
        elif random.random() < 0.7:
            new_x = old_x + random.randint(-4, 4)
            new_y = old_y + random.randint(-4, 4)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        else:
            # Larger jump
            new_x = old_x + random.randint(-10, 10)
            new_y = old_y + random.randint(-10, 10)
            new_x = max(-30, min(30, new_x))
            new_y = max(-30, min(30, new_y))
        
        new_point = (new_x, new_y)
        if new_point == (old_x, old_y):
            continue
        if new_point in curr_occupied:
            continue
            
        # Perform move
        curr_points[idx] = [new_x, new_y]
        curr_occupied.remove((old_x, old_y))
        curr_occupied.add(new_point)
        
        # Evaluate
        new_areas = _vectorized_compute_areas(curr_points)
        new_target_count = _count_area(new_areas, target_area)
        new_max_mult, new_max_area = _get_max_multiplicity(new_areas)
        
        # Score: primarily target count, but also consider max multiplicity
        new_score = max(new_target_count, new_max_mult)
        curr_score = max(curr_target_count, curr_max_mult)
        
        improved = False
        if new_score > curr_score:
            improved = True
            no_improve_count = 0
        elif new_score == curr_score:
            # Tie: prefer higher target count
            if new_target_count > curr_target_count:
                improved = True
                no_improve_count = 0
            elif random.random() < 0.5:
                improved = True
        else:
            delta = curr_score - new_score
            if random.random() < np.exp(-delta / temp):
                improved = True
        
        if improved:
            curr_target_count = new_target_count
            curr_max_mult = new_max_mult
            curr_max_area = new_max_area
            curr_score = new_score
            
            if curr_score > best_score:
                best_score = curr_score
                best_points = curr_points.copy()
                best_area = target_area if curr_target_count >= curr_max_mult else curr_max_area
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Revert
            curr_points[idx] = [old_x, old_y]
            curr_occupied.remove(new_point)
            curr_occupied.add((old_x, old_y))
            no_improve_count += 1
        
        # Cool down
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        
        # If stuck for too long, restart with higher temp
        if no_improve_count > max_no_improve:
            temp = max(temp, 1.0)
            no_improve_count = 0
    
    return best_score, best_points, best_area

def construct():
    start_time = time.time()
    # Set deadline to 50s to ensure we return well before the 60s limit
    deadline = start_time + 50
    
    # Try multiple initial configurations
    initial_configs = _structured_points()
    
    best_overall_score = 0
    best_overall_points = None
    best_overall_area = 0
    
    # Phase 1: Quick evaluation of initial configs
    for config in initial_configs:
        if time.time() > deadline:
            break
        areas = _vectorized_compute_areas(config)
        score, area = _get_max_multiplicity(areas)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = config.copy()
            best_overall_area = area
    
    if best_overall_points is None:
        # Fallback if initial configs failed or were too slow (should not happen)
        best_overall_points = _random_points_in_box()
        areas = _vectorized_compute_areas(best_overall_points)
        best_overall_score, best_overall_area = _get_max_multiplicity(areas)

    # Phase 2: SA search from best initial config
    # Use a significant portion of the time for the main search
    sub_deadline_2 = min(deadline - 10, start_time + 30)
    if time.time() < sub_deadline_2:
        score, points, area = _sa_search(best_overall_points, best_overall_area, sub_deadline_2, initial_temp=3.0)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_points = points
            best_overall_area = area
    
    # Phase 3: Try different target areas
    if time.time() < deadline - 5:
        # Get top few areas from current best
        areas = _vectorized_compute_areas(best_overall_points)
        non_zero = areas[areas > 0]
        if len(non_zero) > 0:
            unique_areas, counts = np.unique(non_zero, return_counts=True)
            # Sort by count descending
            sorted_idx = np.argsort(counts)[::-1]
            top_areas = unique_areas[sorted_idx[:3]] # Limit to 3 to save time
            
            for target in top_areas:
                if time.time() > deadline - 5:
                    break
                # Allocate remaining time minus safety margin
                sub_deadline = deadline - 2
                score, points, area = _sa_search(best_overall_points, int(target), sub_deadline, initial_temp=2.0)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_points = points
                    best_overall_area = area
    
    # Ensure we have valid points to return
    if best_overall_points is None:
        best_overall_points = _random_points_in_box()

    return [(int(x), int(y)) for x, y in best_overall_points.tolist()]
