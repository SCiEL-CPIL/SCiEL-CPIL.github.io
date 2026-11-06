"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    # Precompute all nonconstant 4-term APs (0-indexed positions)
    # AP: (i, i+d, i+2d, i+3d) where d >= 1, all indices < n
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map each position to the APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        """Count number of APs with fewer than 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
            elif c0 == c1 and c2 == c3:
                v += 1
            elif c0 == c2 and c1 == c3:
                v += 1
            elif c0 == c3 and c1 == c2:
                v += 1
        return v
    
    def is_ap_valid(colors, ap):
        c = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
        return len(c) >= 3
    
    def greedy_construct():
        """Deterministic greedy with bitmask conflict checking."""
        colors = [-1] * n
        # For each position, track which colors would cause violations
        # Use bitmasks: bit c is set if color c is forbidden at position x
        
        for x in range(n):
            # Determine forbidden colors for position x
            forbidden = 0  # bitmask of forbidden colors
            
            for ap_idx in pos_to_aps[x]:
                ap = aps[ap_idx]
                # Check if all other positions in this AP are already colored
                other_positions = [p for p in ap if p != x]
                if any(colors[p] == -1 for p in other_positions):
                    continue
                
                # Get colors of other positions
                other_colors = [colors[p] for p in other_positions]
                
                # For this AP, if we assign color c to x, the AP is invalid
                # if the set of colors has < 3 distinct values
                # The other 3 positions have colors other_colors
                # If all three are the same color c0:
                #   - If c == c0: all four same -> invalid
                #   - If c != c0: three of one, one of another -> only 2 distinct -> invalid
                # If two are c0 and one is c1 (c0 != c1):
                #   - If c == c0: three c0, one c1 -> 2 distinct -> invalid
                #   - If c == c1: two c0, two c1 -> 2 distinct -> invalid  
                #   - If c != c0 and c != c1: two c0, one c1, one c -> 3 distinct -> valid
                # If all three distinct (c0, c1, c2):
                #   - Any c works: at least 3 distinct always
                
                if other_colors[0] == other_colors[1] == other_colors[2]:
                    c0 = other_colors[0]
                    # All four same if c == c0, or 2 distinct if c != c0
                    # Actually: if c == c0, all 4 same (1 distinct). If c != c0, 3 of c0 and 1 of c (2 distinct)
                    # Both are invalid! So ALL colors are forbidden for this AP
                    forbidden = (1 << 128) - 1  # All colors forbidden
                    break
                elif other_colors[0] == other_colors[1] or other_colors[0] == other_colors[2] or other_colors[1] == other_colors[2]:
                    # Two same, one different
                    c0 = other_colors[0]
                    c1 = other_colors[1]
                    c2 = other_colors[2]
                    if c0 == c1:
                        # c0, c0, c2
                        # c == c0: three c0, one c2 -> 2 distinct -> invalid
                        # c == c2: two c0, two c2 -> 2 distinct -> invalid
                        # c != c0 and c != c2: valid
                        forbidden |= (1 << c0) | (1 << c2)
                    elif c0 == c2:
                        # c0, c1, c0
                        # c == c0: three c0, one c1 -> 2 distinct -> invalid
                        # c == c1: two c0, two c1 -> 2 distinct -> invalid
                        # c != c0 and c != c1: valid
                        forbidden |= (1 << c0) | (1 << c1)
                    else:
                        # c1, c2, c2
                        # c == c2: three c2, one c1 -> 2 distinct -> invalid
                        # c == c1: two c2, two c1 -> 2 distinct -> invalid
                        # c != c1 and c != c2: valid
                        forbidden |= (1 << c2) | (1 << c1)
                else:
                    # All three distinct: any color works
                    pass
            
            # Find smallest valid color
            for color in range(128):
                if not (forbidden & (1 << color)):
                    colors[x] = color
                    break
            else:
                # Fallback: should not happen with 128 colors
                colors[x] = x
        
        return colors
    
    def try_reduce_colors(colors, max_time):
        """Try to reduce the number of distinct colors used while maintaining validity."""
        colors = colors[:]
        used_colors = sorted(set(colors))
        current_count = len(used_colors)
        
        if current_count <= 3:
            return colors, current_count
        
        best_colors = colors[:]
        best_count = current_count
        
        # Try to eliminate each color by remapping it to another
        for color_to_remove in used_colors:
            if time.time() > max_time:
                break
            
            # Find positions with this color
            positions = [i for i in range(n) if colors[i] == color_to_remove]
            if not positions:
                continue
            
            # Try remapping to each other used color
            for target_color in used_colors:
                if target_color == color_to_remove:
                    continue
                
                # Temporarily remap
                old_vals = {}
                for pos in positions:
                    old_vals[pos] = colors[pos]
                    colors[pos] = target_color
                
                v = count_violations(colors)
                if v == 0:
                    # Successful removal
                    best_colors = colors[:]
                    best_count -= 1
                    used_colors = sorted(set(colors))
                    break
                else:
                    # Revert
                    for pos in positions:
                        colors[pos] = old_vals[pos]
        
        return best_colors, best_count
    
    def local_search_refine(colors, time_limit):
        """Simulated annealing to fix violations and minimize color count."""
        colors = colors[:]
        current_violations = count_violations(colors)
        current_colors_used = len(set(colors))
        color_counts = [0] * 128
        for c in colors:
            color_counts[c] += 1
        
        best_colors = colors[:]
        best_violations = current_violations
        best_colors_used = current_colors_used
        
        temp = 3.0
        cooling = 0.997
        min_temp = 0.01
        
        while time.time() < time_limit and temp > min_temp:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Determine candidate colors: current used colors + a few new ones
            used = set(colors)
            candidates = list(used)
            # Add some random new colors to explore
            for _ in range(2):
                new_c = random.randint(0, 127)
                if new_c not in candidates:
                    candidates.append(new_c)
            
            best_new_color = old_color
            best_delta_v = 0
            best_delta_c = 0
            
            affected_aps = pos_to_aps[pos]
            
            # Count current violations among affected APs
            curr_invalid = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    curr_invalid += 1
            
            for new_color in candidates:
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                new_invalid = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid += 1
                
                delta_v = new_invalid - curr_invalid
                # Efficient color count change
                delta_c = 0
                if color_counts[old_color] == 1:
                    delta_c -= 1
                if color_counts[new_color] == 0:
                    delta_c += 1
                
                # Score: prioritize fixing violations, then reducing colors
                score = delta_v * 100 + delta_c
                best_score = best_delta_v * 100 + best_delta_c
                
                if score < best_score:
                    best_score = score
                    best_delta_v = delta_v
                    best_delta_c = delta_c
                    best_new_color = new_color
                elif score == best_score and random.random() < 0.3:
                    best_new_color = new_color
                    best_delta_v = delta_v
                    best_delta_c = delta_c
            
            colors[pos] = old_color
            
            # Accept based on simulated annealing
            accept = False
            if best_delta_v < 0 or (best_delta_v == 0 and best_delta_c < 0):
                accept = True
            elif best_delta_v == 0 and best_delta_c == 0:
                accept = True
            else:
                # Compute acceptance probability
                energy = best_delta_v * 100 + best_delta_c
                if energy > 0 and temp > 0:
                    prob = math.exp(-energy / temp)
                    accept = random.random() < prob
            
            if accept:
                # Update color counts
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    pass
                color_counts[best_new_color] += 1
                
                colors[pos] = best_new_color
                current_violations += best_delta_v
                current_colors_used = len(set(colors))
                
                if current_violations < best_violations or \
                   (current_violations == best_violations and current_colors_used < best_colors_used):
                    best_violations = current_violations
                    best_colors_used = current_colors_used
                    best_colors = colors[:]
                
                if current_violations == 0 and current_colors_used <= 3:
                    break
            
            temp *= cooling
        
        return best_colors, best_violations, best_colors_used
    
    def final_polish(colors, time_limit):
        """High-precision polish: try all single-position changes to minimize color count."""
        colors = colors[:]
        best_colors = colors[:]
        best_count = len(set(colors))
        
        # Repeatedly try to improve
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for pos in range(n):
                old_color = colors[pos]
                # Try each other used color
                for new_color in set(colors):
                    if new_color == old_color:
                        continue
                    colors[pos] = new_color
                    v = count_violations(colors)
                    c = len(set(colors))
                    if v == 0 and c < best_count:
                        best_colors = colors[:]
                        best_count = c
                        improved = True
                    colors[pos] = old_color
        
        return best_colors, best_count
    
    # Step 1: Greedy construction with bitmask
    colors = greedy_construct()
    initial_violations = count_violations(colors)
    initial_colors = len(set(colors))
    
    # Step 2: Try to reduce colors if already valid
    if initial_violations == 0:
        colors, reduced_count = try_reduce_colors(colors, time.time() + 20.0)
    
    # Step 3: Local search refinement if time remains
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        colors, final_violations, final_colors = local_search_refine(colors, deadline - 5.0)
    else:
        final_violations = initial_violations
        final_colors = initial_colors
    
    # Step 4: Final polish pass
    if time.time() < deadline - 3.0:
        colors, final_colors = final_polish(colors, deadline - 2.0)
    
    # Ensure we return a valid solution
    if final_violations > 0:
        # Fallback: identity mapping is always valid
        return list(range(n))
    
    return colors
# EVOLVE-BLOCK-END
