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

def _progression_violated(colors, idx_set, k_colors):
    # Determine if this progression is already IMPOSSIBLE to satisfy
    present = set()
    unassigned = 0
    for i in idx_set:
        c = colors[i]
        if c == -1:
            unassigned += 1
        else:
            present.add(c)
    if len(present) >= 3:
        return False  # already satisfied
    # Need to end with >=3 distinct colors
    if len(present) + unassigned < 3:
        return True  # impossible
    return False

def _find_periodic_pattern(period, k_colors, deadline):
    """
    Exhaustively search for a periodic pattern of given period and k_colors
    that satisfies the 3-color progression constraint for all x <= 128.
    Returns a list of 128 colors if found, else None.
    """
    # Enumerate all possible patterns of length 'period' with k_colors
    # Use symmetry breaking: fix pattern[0]=0, and require pattern to use all k_colors
    
    # For small periods, enumerate all assignments
    if period <= 10:
        # Generate all patterns with pattern[0]=0
        # Use colors 0..k_colors-1
        for rest in product(range(k_colors), repeat=period-1):
            pattern = [0] + list(rest)
            
            # Check if pattern uses at least 3 colors (required for any AP to have 3 colors)
            if len(set(pattern)) < 3:
                continue
            
            # Check if this periodic pattern satisfies all APs for positions 0..127
            valid = True
            for ap in _ALL_APS:
                colors_in_ap = {pattern[i % period] for i in ap}
                if len(colors_in_ap) < 3:
                    valid = False
                    break
            
            if valid:
                # Expand to 128 positions
                expanded = [pattern[i % period] for i in range(128)]
                return expanded
            
            # Time check
            if time.time() > deadline:
                return None
    
    return None

def _validate(coloring):
    """Check if a coloring satisfies the constraint for all 4-term APs."""
    for ap in _ALL_APS:
        if len({coloring[i] for i in ap}) < 3:
            return False
    return True

def _local_search_reduce_colors(coloring, deadline):
    """
    Try to reduce the number of distinct colors in a valid coloring
    by reassigning positions to existing colors.
    """
    coloring = list(coloring)
    n = len(coloring)
    
    # Get current distinct colors
    def get_used_colors(arr):
        return sorted(set(arr))
    
    used = get_used_colors(coloring)
    current_count = len(used)
    
    # Try to reduce colors by reassigning positions
    # Strategy: for each color in use, try to remove it by reassigning all its positions
    # to other colors, checking validity after each change
    
    best = coloring[:]
    best_count = current_count
    
    for _ in range(100):  # limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        used = get_used_colors(best)
        
        # Try removing each color
        for color_to_remove in used:
            if time.time() > deadline:
                break
            
            if len(used) <= 3:
                break  # can't go below 3
            
            trial = best[:]
            # Find positions with this color
            positions = [i for i in range(n) if trial[i] == color_to_remove]
            
            if not positions:
                continue
            
            # Try to reassign each position to another color
            success = True
            for pos in positions:
                original = trial[pos]
                # Try each other color
                found_new = False
                for new_color in used:
                    if new_color == color_to_remove:
                        continue
                    trial[pos] = new_color
                    if _validate(trial):
                        found_new = True
                        break
                    trial[pos] = original
                
                if not found_new:
                    success = False
                    break
            
            if success:
                new_used = get_used_colors(trial)
                if len(new_used) < len(get_used_colors(best)):
                    best = trial
                    best_count = len(new_used)
                    improved = True
                    break
        
        if not improved:
            break
    
    return best, best_count

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    n = 128
    ap_count = len(_ALL_APS)
    
    # Precompute position -> APs mapping
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)
    
    best = None
    best_ncolors = n  # start with worst

    # First, try to find a periodic pattern with 3 colors
    # Try periods from small to large
    for period in [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]:
        if time.time() > DEADLINE - 10:
            break
        
        pattern = _find_periodic_pattern(period, 3, DEADLINE - 10)
        if pattern is not None:
            best = pattern
            best_ncolors = 3
            break
    
    # If no periodic pattern found, try backtracking search with 3 colors
    if best is None:
        if time.time() < DEADLINE - 20:
            colors = [-1] * n
            solution_found = False
            node_count = 0
            
            # Heuristic: order positions by most-constrained first
            degree = [len(pos_in_aps[i]) for i in range(n)]
            order = list(range(n))
            order.sort(key=lambda i: (-degree[i], i))
            
            pos_to_fill = order
            
            def allowed_colors(pos, k_colors):
                al = set(range(k_colors))
                for ap in pos_in_aps[pos]:
                    others = [i for i in ap if i != pos]
                    oc = {colors[i] for i in others if colors[i] != -1}
                    unassigned_others = sum(1 for i in others if colors[i] == -1)
                    # For each candidate color, check if it leads to a violation
                    to_remove = set()
                    for c in al:
                        current = oc | {c}
                        max_possible = len(current) + unassigned_others
                        if max_possible < 3:
                            to_remove.add(c)
                    al -= to_remove
                return al
            
            def backtrack(idx_in_order, k_colors):
                nonlocal solution_found, node_count
                if time.time() > DEADLINE - 5:
                    return False
                if solution_found:
                    return True
                node_count += 1
                if node_count % 1000 == 0 and time.time() > DEADLINE - 5:
                    return False

                if idx_in_order == n:
                    for ap in _ALL_APS:
                        if len({colors[i] for i in ap}) < 3:
                            return False
                    solution_found = True
                    return True

                pos = pos_to_fill[idx_in_order]
                for c in sorted(allowed_colors(pos, k_colors)):
                    colors[pos] = c
                    ok = True
                    for ap in pos_in_aps[pos]:
                        if _progression_violated(colors, ap, k_colors):
                            ok = False
                            break
                    if ok:
                        if backtrack(idx_in_order + 1, k_colors):
                            return True
                    colors[pos] = -1
                return False

            # Try 3 colors first
            if backtrack(0, 3):
                best = colors[:]
                best_ncolors = 3
    
    # If we have a solution, try to reduce colors further via local search
    if best is not None and best_ncolors > 3 and time.time() < DEADLINE - 5:
        improved, improved_count = _local_search_reduce_colors(best, DEADLINE - 5)
        if improved_count < best_ncolors:
            best = improved
            best_ncolors = improved_count
    
    if best is not None:
        # Ensure all are valid non-negative and within 0..127
        return best
    else:
        # Fallback: greedy construction
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1) if x-1-3*d >= 0)
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1) if x-1-2*d >= 0)):
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors
