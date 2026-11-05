import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 128
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) 0-indexed
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # pos_to_aps: for each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    # Count distinct colors in an AP using bitmask
    def count_distinct_mask(c0, c1, c2, c3):
        mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3)
        return bin(mask).count('1')
    
    # Initial greedy coloring
    colors = [0] * n
    for x in range(n):
        best_color = -1
        for color in range(128):
            valid = True
            for ap_idx in pos_to_aps[x]:
                ap = ap_list[ap_idx]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                # Replace the color at position x
                if ap[0] == x:
                    c0 = color
                elif ap[1] == x:
                    c1 = color
                elif ap[2] == x:
                    c2 = color
                elif ap[3] == x:
                    c3 = color
                if count_distinct_mask(c0, c1, c2, c3) < 3:
                    valid = False
                    break
            if valid:
                best_color = color
                break
        if best_color == -1:
            best_color = 0
        colors[x] = best_color
    
    num_colors_used = len(set(colors))
    
    # If we already have 3 colors, we're done (optimal)
    if num_colors_used <= 3:
        return colors
    
    # Local search to reduce colors
    best_colors = colors[:]
    best_num = num_colors_used
    
    # Try to reduce to k colors for k = 3, 4, ...
    for target_k in range(3, best_num):
        if time.time() > deadline:
            break
        
        found = False
        for restart in range(200):
            if time.time() > deadline:
                break
            
            # Random initial coloring with target_k colors
            c = [random.randint(0, target_k - 1) for _ in range(n)]
            
            # Compute initial violations
            violations = [0] * num_aps
            total_violations = 0
            for idx, ap in enumerate(ap_list):
                c0, c1, c2, c3 = c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]]
                if count_distinct_mask(c0, c1, c2, c3) < 3:
                    violations[idx] = 1
                    total_violations += 1
            
            for iteration in range(20000):
                if time.time() > deadline:
                    break
                
                if total_violations == 0:
                    best_colors = c[:]
                    best_num = target_k
                    found = True
                    break
                
                # Pick random position
                pos = random.randint(0, n - 1)
                old_color = c[pos]
                
                best_new = old_color
                best_v = total_violations
                
                for new_color in range(target_k):
                    if new_color == old_color:
                        continue
                    
                    # Calculate change in violations for APs containing pos
                    delta = 0
                    for ap_idx in pos_to_aps[pos]:
                        ap = ap_list[ap_idx]
                        c0, c1, c2, c3 = c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]]
                        
                        # Old distinct count
                        old_distinct = count_distinct_mask(c0, c1, c2, c3)
                        old_viol = 1 if old_distinct < 3 else 0
                        
                        # New distinct count with new_color at pos
                        if ap[0] == pos:
                            new_distinct = count_distinct_mask(new_color, c1, c2, c3)
                        elif ap[1] == pos:
                            new_distinct = count_distinct_mask(c0, new_color, c2, c3)
                        elif ap[2] == pos:
                            new_distinct = count_distinct_mask(c0, c1, new_color, c3)
                        else:
                            new_distinct = count_distinct_mask(c0, c1, c2, new_color)
                        
                        new_viol = 1 if new_distinct < 3 else 0
                        delta += new_viol - old_viol
                    
                    if total_violations + delta < best_v:
                        best_v = total_violations + delta
                        best_new = new_color
                    if best_v == 0:
                        break
                
                if best_new != old_color:
                    # Apply change
                    c[pos] = best_new
                    for ap_idx in pos_to_aps[pos]:
                        ap = ap_list[ap_idx]
                        c0, c1, c2, c3 = c[ap[0]], c[ap[1]], c[ap[2]], c[ap[3]]
                        new_distinct = count_distinct_mask(c0, c1, c2, c3)
                        violations[ap_idx] = 1 if new_distinct < 3 else 0
                    total_violations = sum(violations)
                
                if total_violations == 0:
                    best_colors = c[:]
                    best_num = target_k
                    found = True
                    break
            
            if found:
                break
    
    if best_num <= 3:
        return best_colors
    
    return best_colors
