"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

# EVOLVE-BLOCK-START
import sys

def _doubled_areas(points):
    """Compute absolute doubled areas for all triples of points.
    Returns list of (area, triple_indices) for non-zero areas."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                if det > 0:
                    areas.append((det, (i, j, k)))
    return areas

def _score(points):
    """Compute the maximum multiplicity of a single positive doubled area."""
    areas = _doubled_areas(points)
    if not areas:
        return 0
    counts = Counter(a for a, _ in areas)
    return max(counts.values())

def _generate_grid():
    """Generate the initial 6x5 grid of points."""
    return [(x, y) for x in range(6) for y in range(5)]

def _is_valid(points):
    """Check if points are 30 distinct integer pairs in [-30,30]^2."""
    if len(points) != 30:
        return False
    seen = set()
    for x, y in points:
        if not (isinstance(x, int) and isinstance(y, int)):
            return False
        if x < -30 or x > 30 or y < -30 or y > 30:
            return False
        if (x, y) in seen:
            return False
        seen.add((x, y))
    return True

def _local_search(points, time_limit=55):
    """Local search to maximize the frequency of a specific triangle area."""
    start_time = time.time()
    points = [list(p) for p in points]
    
    # Compute initial areas
    current_areas = _doubled_areas(points)
    current_counts = Counter(a for a, _ in current_areas)
    current_best_area = max(current_counts, key=current_counts.get)
    current_best_count = current_counts[current_best_area]
    current_score = current_best_count
    
    # Precompute all triples
    n = len(points)
    triples = list(combinations(range(n), 3))
    
    # For each point and each direction, precompute which triples are affected
    # When we move point i, all triples containing i change area
    
    best_points = [tuple(p) for p in points]
    best_score = current_score
    
    # Try to improve by moving individual points
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            for dx, dy in directions:
                new_x = points[i][0] + dx
                new_y = points[i][1] + dy
                if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                    continue
                # Check if position is already taken
                pos = (new_x, new_y)
                if any(tuple(p) == pos for j, p in enumerate(points) if j != i):
                    continue
                
                old_x, old_y = points[i]
                points[i] = [new_x, new_y]
                
                # Recompute areas for triples involving point i
                new_areas = []
                for tri in triples:
                    if i in tri:
                        x1, y1 = points[tri[0]]
                        x2, y2 = points[tri[1]]
                        x3, y3 = points[tri[2]]
                        det = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
                        if det > 0:
                            new_areas.append(det)
                    else:
                        # Use cached area for triples not involving i
                        # We need to recompute all, so let's just do full recompute
                        pass
                
                # Full recompute for simplicity (4060 triples is fast)
                all_areas = _doubled_areas(points)
                all_counts = Counter(a for a, _ in all_areas)
                
                if all_counts:
                    best_area_in_config = max(all_counts, key=all_counts.get)
                    best_count_in_config = all_counts[best_area_in_config]
                    
                    if best_count_in_config > current_score:
                        current_score = best_count_in_config
                        current_best_area = best_area_in_config
                        current_best_count = best_count_in_config
                        best_points = [tuple(p) for p in points]
                        improved = True
                        if time.time() - start_time > time_limit:
                            break
                    else:
                        # Revert
                        points[i] = [old_x, old_y]
                else:
                    points[i] = [old_x, old_y]
            
            if time.time() - start_time > time_limit:
                break
        
        if not improved:
            # Try random restarts within the neighborhood
            for _ in range(100):
                if time.time() - start_time > time_limit:
                    break
                i = random.randint(0, n-1)
                dx = random.randint(-3, 3)
                dy = random.randint(-3, 3)
                if dx == 0 and dy == 0:
                    continue
                new_x = points[i][0] + dx
                new_y = points[i][1] + dy
                if new_x < -30 or new_x > 30 or new_y < -30 or new_y > 30:
                    continue
                pos = (new_x, new_y)
                if any(tuple(p) == pos for j, p in enumerate(points) if j != i):
                    continue
                
                old_x, old_y = points[i]
                points[i] = [new_x, new_y]
                all_areas = _doubled_areas(points)
                all_counts = Counter(a for a, _ in all_areas)
                
                if all_counts:
                    best_area_in_config = max(all_counts, key=all_counts.get)
                    best_count_in_config = all_counts[best_area_in_config]
                    
                    if best_count_in_config > current_score:
                        current_score = best_count_in_config
                        current_best_area = best_area_in_config
                        current_best_count = best_count_in_config
                        best_points = [tuple(p) for p in points]
                        improved = True
                    else:
                        points[i] = [old_x, old_y]
                else:
                    points[i] = [old_x, old_y]
            
            if not improved:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = 85  # Leave 5s safety margin
    
    # Start with the grid
    points = _generate_grid()
    best_points = points
    best_score = _score(points)
    
    # Run local search
    search_points, search_score = _local_search(points, time_limit=80)
    if search_score > best_score:
        best_points = search_points
        best_score = search_score
    
    # Additional random restarts if time permits
    while time.time() - start_time < deadline - 5:
        # Perturb the best solution
        perturbed = [list(p) for p in best_points]
        for _ in range(random.randint(1, 5)):
            i = random.randint(0, 29)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            new_x = perturbed[i][0] + dx
            new_y = perturbed[i][1] + dy
            if -30 <= new_x <= 30 and -30 <= new_y <= 30:
                pos = (new_x, new_y)
                if not any(tuple(p) == pos for j, p in enumerate(perturbed) if j != i):
                    perturbed[i] = [new_x, new_y]
        
        if _is_valid(perturbed):
            pert_score = _score(perturbed)
            if pert_score > best_score:
                best_points = [tuple(p) for p in perturbed]
                best_score = pert_score
            
            # Quick local search from perturbed
            if time.time() - start_time < deadline - 10:
                sp, ss = _local_search(perturbed, time_limit=min(30, deadline - 10 - (time.time() - start_time)))
                if ss > best_score:
                    best_points = sp
                    best_score = ss
    
    return best_points
# EVOLVE-BLOCK-END
