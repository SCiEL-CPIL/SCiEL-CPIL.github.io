"""Combined program: greedy initialization + local search to minimize distinct colors."""
import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v
    
    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3
    
    def get_distinct_colors(colors):
        return len(set(colors))
    
    # Step 1: Greedy initialization - same as original program
    def greedy_construct():
        colors = []
        for x in range(1, n + 1):
            for color in range(n):  # Try colors 0..n-1
                # Check all 4-term APs ending at position x-1 (0-indexed)
                valid = True
                # APs where x-1 is the last element: (x-1-3d, x-1-2d, x-1-d, x-1)
                for d in range(1, (x - 1) // 3 + 1):
                    if x - 1 - 3 * d < 0:
                        break
                    idxs = [x - 1 - 3 * d, x - 1 - 2 * d, x - 1 - d, x - 1]
                    if colors[x - 1 - 3 * d] == -1 or colors[x - 1 - 2 * d] == -1 or colors[x - 1 - d] == -1:
                        continue  # Not all positions colored yet
                    if len({color, colors[x - 1 - 3 * d], colors[x - 1 - 2 * d], colors[x - 1 - d]}) < 3:
                        valid = False
                        break
                
                # Also check APs where x-1 is not the last element
                # x-1 is first: (x-1, x-1+d, x-1+2d, x-1+3d)
                for d in range(1, (n - 1 - (x - 1)) // 3 + 1):
                    last = x - 1 + 3 * d
                    if last >= n:
                        break
                    idxs = [x - 1, x - 1 + d, x - 1 + 2 * d, x - 1 + 3 * d]
                    # Only check if all other positions are colored
                    if all(colors[i] != -1 for i in idxs[1:]):
                        if len({color, colors[x - 1 + d], colors[x - 1 + 2 * d], colors[x - 1 + 3 * d]}) < 3:
                            valid = False
                            break
                    if not valid:
                        break
                
                # x-1 is second: (x-1-d, x-1, x-1+d, x-1+2d)
                if valid:
                    for d in range(1, min(x - 1, (n - 1 - (x - 1)) // 2 + 1) + 1):
                        first = x - 1 - d
                        last = x - 1 + 2 * d
                        if first < 0 or last >= n:
                            continue
                        idxs = [first, x - 1, x - 1 + d, last]
                        if all(colors[i] != -1 for i in idxs if i != x - 1):
                            if len({color, colors[first], colors[x - 1 + d], colors[last]}) < 3:
                                valid = False
                                break
                    if not valid:
                        break
                
                # x-1 is third: (x-1-2d, x-1-d, x-1, x-1+d)
                if valid:
                    for d in range(1, min((x - 1) // 2, (n - 1 - (x - 1)) + 1) + 1):
                        first = x - 1 - 2 * d
                        last = x - 1 + d
                        if first < 0 or last >= n:
                            continue
                        idxs = [first, x - 1 - d, x - 1, last]
                        if all(colors[i] != -1 for i in idxs if i != x - 1):
                            if len({color, colors[first], colors[x - 1 - d], colors[last]}) < 3:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    colors.append(color)
                    break
            else:
                # Should not happen with n colors available
                colors.append(0)
        return colors
    
    # Get initial greedy coloring
    colors = greedy_construct()
    current_violations = count_violations(colors)
    best_colors = colors[:]
    best_distinct = get_distinct_colors(colors)
    
    if current_violations != 0:
        # Fallback: identity mapping
        return list(range(n))
    
    # Step 2: Local search to reduce number of distinct colors
    # Strategy: try to recolor positions to merge color classes
    # A move is "good" if it reduces the number of distinct colors without creating violations
    
    def local_search_min_colors(time_limit):
        nonlocal colors, best_colors, best_distinct
        
        temp = 1.0
        cooling = 0.9995
        min_temp = 0.001
        
        while time.time() < time_limit:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all possible new colors (0 to n-1)
            best_new_color = old_color
            best_delta_distinct = 0
            best_delta_violations = 0
            
            affected_aps = pos_to_aps[pos]
            
            # Count current violations for affected APs
            currently_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    currently_invalid += 1
            
            # Get current distinct color count
            current_distinct = get_distinct_colors(colors)
            
            # Try each new color
            for new_color in range(n):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Recount violations for affected APs
                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid += 1
                
                new_distinct = get_distinct_colors(colors)
                
                delta_violations = new_invalid - currently_invalid
                delta_distinct = new_distinct - current_distinct
                
                # We want to minimize distinct colors, but cannot increase violations
                # Accept if: fewer distinct colors and no more violations
                # Or: same distinct colors and fewer violations
                # Or: with probability, accept worse distinct if violations decrease enough
                
                if delta_violations < 0 and delta_distinct <= 0:
                    # Better: fewer violations and no more distinct colors
                    score = -delta_distinct * 100 - delta_violations
                elif delta_violations == 0 and delta_distinct < 0:
                    # Same violations, fewer distinct colors - great!
                    score = -delta_distinct * 100
                elif delta_violations == 0 and delta_distinct == 0:
                    # Neutral
                    score = 0
                elif delta_violations > 0:
                    # More violations - only accept with low probability
                    score = delta_violations * 100 + delta_distinct
                else:
                    score = delta_distinct
                
                if score < best_delta_distinct or (score == best_delta_distinct and random.random() < 0.5):
                    best_delta_distinct = score
                    best_new_color = new_color
                    best_delta_violations = delta_violations
            
            colors[pos] = old_color
            
            # Decide whether to accept
            if best_delta_distinct < 0:
                accept = True
            elif best_delta_distinct == 0:
                accept = random.random() < 0.5
            else:
                # Accept worse with probability
                prob = min(1.0, max(0.0, 1.0 - best_delta_distinct / (temp * 100 + 1)))
                accept = random.random() < prob
            
            if accept and best_new_color != old_color:
                colors[pos] = best_new_color
                new_violations = count_violations(colors)
                new_distinct = get_distinct_colors(colors)
                
                if new_violations == 0 and new_distinct < best_distinct:
                    best_distinct = new_distinct
                    best_colors = colors[:]
                
                if new_violations == 0:
                    # Valid solution, continue searching for better
                    pass
                else:
                    # Revert if we created violations
                    colors[pos] = old_color
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors, best_distinct
    
    # Run local search with time budget
    time_left = deadline - 5.0 - time.time()
    if time_left > 1.0:
        best_colors, best_distinct = local_search_min_colors(time.time() + time_left)
    
    # Verify final result
    if count_violations(best_colors) != 0:
        # Fallback to greedy
        best_colors = greedy_construct()
    
    return best_colors
