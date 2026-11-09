import time
import numpy as np
from itertools import product

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

def _count_violations(colors):
    """Count number of APs with fewer than 3 distinct colors."""
    violations = 0
    for ap in _ALL_APS:
        c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
        if len(c) < 3:
            violations += 1
    return violations

def _count_distinct_colors(colors):
    return len(set(colors))

def _greedy_valid_coloring(k_colors):
    """Try to build a valid coloring with at most k_colors using greedy assignment."""
    n = 128
    colors = [-1] * n
    
    # Order positions by most-constrained first (most APs)
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)
    
    degree = [len(pos_in_aps[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-degree[i], i))
    
    for pos in order:
        # Find a color that doesn't create immediate violations
        best_color = None
        min_violations = float('inf')
        
        for c in range(k_colors):
            colors[pos] = c
            # Count violations involving pos
            v = 0
            for ap in pos_in_aps[pos]:
                # Only count if all 4 positions are assigned
                if all(colors[i] != -1 for i in ap):
                    if len({colors[i] for i in ap}) < 3:
                        v += 1
            if v < min_violations:
                min_violations = v
                best_color = c
            if v == 0:
                break
            colors[pos] = -1
        
        colors[pos] = best_color
    
    # Check if valid
    if _count_violations(colors) == 0:
        return colors
    return None

def _local_search(initial_colors, k_colors, deadline):
    """Simulated annealing to minimize distinct colors while maintaining validity."""
    n = 128
    best_colors = initial_colors[:]
    best_score = _count_distinct_colors(initial_colors)
    current_colors = initial_colors[:]
    current_violations = _count_violations(current_colors)
    current_distinct = _count_distinct_colors(current_colors)
    
    # If already valid, try to reduce colors
    if current_violations == 0:
        # Try to merge colors to reduce distinct count
        while time.time() < deadline:
            # Find two colors that can be merged without creating violations
            color_map = {c: i for i, c in enumerate(sorted(set(current_colors)))}
            used_colors = sorted(set(current_colors))
            merged = False
            
            for i in range(len(used_colors)):
                for j in range(i+1, len(used_colors)):
                    # Try merging color j into color i
                    new_colors = current_colors[:]
                    for idx in range(n):
                        if new_colors[idx] == used_colors[j]:
                            new_colors[idx] = used_colors[i]
                    
                    v = _count_violations(new_colors)
                    if v == 0:
                        current_colors = new_colors
                        current_distinct = _count_distinct_colors(current_colors)
                        if current_distinct < best_score:
                            best_score = current_distinct
                            best_colors = current_colors[:]
                        merged = True
                        break
                if merged:
                    break
            
            if not merged:
                break
    
    # Now use simulated annealing to find better solutions
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() < deadline and T > T_min:
        # Pick a random position to recolor
        pos = np.random.randint(n)
        old_color = current_colors[pos]
        new_color = np.random.randint(k_colors)
        
        if new_color == old_color:
            continue
        
        current_colors[pos] = new_color
        new_violations = _count_violations(current_colors)
        new_distinct = _count_distinct_colors(current_colors)
        
        # Score: prefer valid (0 violations) and fewer colors
        old_score = current_violations * 1000 + current_distinct
        new_score = new_violations * 1000 + new_distinct
        
        if new_score < old_score or (new_score > old_score and np.random.random() < np.exp(-(new_score - old_score) / T)):
            current_violations = new_violations
            current_distinct = new_distinct
            
            if new_violations == 0 and new_distinct < best_score:
                best_score = new_distinct
                best_colors = current_colors[:]
        else:
            current_colors[pos] = old_color
        
        T *= cooling_rate
    
    return best_colors

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    n = 128
    
    # First, try to find a valid coloring with 3 colors using greedy
    best = None
    best_ncolors = n
    
    for k in [3, 4, 5, 6, 8, 10, 12, 16]:
        if time.time() > DEADLINE:
            break
        
        result = _greedy_valid_coloring(k)
        if result is not None:
            best = result
            best_ncolors = _count_distinct_colors(result)
            break
    
    # If no valid coloring found, use local search from a random initialization
    if best is None or best_ncolors > 3:
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > DEADLINE - 5:
                break
            
            # Random initialization with 3 colors
            init = np.random.randint(0, 3, n).tolist()
            
            # Run local search
            result = _local_search(init, 3, DEADLINE - 2)
            
            if result is not None:
                v = _count_violations(result)
                d = _count_distinct_colors(result)
                
                if v == 0 and d < best_ncolors:
                    best = result
                    best_ncolors = d
                
                if best_ncolors == 3:
                    break
    
    # If still no valid solution, fallback to a simple pattern
    if best is None or _count_violations(best) > 0:
        # Use a pattern that's likely to work: i mod 4
        best = [i % 4 for i in range(n)]
        # This may not be valid, but it's a fallback
    
    # Final validation and return
    if best is not None and _count_violations(best) == 0:
        return best
    
    # Emergency fallback: unique colors (always valid but many colors)
    return list(range(n))
