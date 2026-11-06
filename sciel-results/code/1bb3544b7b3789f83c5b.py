import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute all nonconstant 4-APs (0-indexed)
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    # Convert to numpy arrays for fast vectorized operations
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each position, precompute which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, (i, j, k, l) in enumerate(aps):
        pos_aps[i].append(idx)
        pos_aps[j].append(idx)
        pos_aps[k].append(idx)
        pos_aps[l].append(idx)
    
    # Convert pos_aps to numpy arrays for faster iteration
    pos_aps_np = [np.array(pa, dtype=np.int32) if pa else np.array([], dtype=np.int32) for pa in pos_aps]
    
    def compute_violation_set(colors):
        """Compute set of violating AP indices using vectorized operations."""
        colors_arr = np.array(colors, dtype=np.int32)
        # Extract colors at each AP position
        c0 = colors_arr[ap_array[:, 0]]
        c1 = colors_arr[ap_array[:, 1]]
        c2 = colors_arr[ap_array[:, 2]]
        c3 = colors_arr[ap_array[:, 3]]
        
        # Count distinct colors in each AP
        # An AP violates if it has < 3 distinct colors
        # Equivalent to: all same, or exactly 2 distinct
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        two_distinct = ((c0 == c1) & (c2 == c3) & (c0 != c2)) | \
                       ((c0 == c2) & (c1 == c3) & (c0 != c1)) | \
                       ((c0 == c3) & (c1 == c2) & (c0 != c1))
        
        violation_mask = all_same | two_distinct
        return set(np.where(violation_mask)[0].tolist())
    
    def compute_violation_count(colors):
        return len(compute_violation_set(colors))
    
    def try_merge_colors(colors, max_k, deadline):
        """Try to merge color pairs to reduce distinct color count while maintaining validity."""
        current_colors = colors[:]
        current_distinct = len(set(current_colors))
        
        if current_distinct <= 4:
            return current_colors, current_distinct
        
        # Try merging each pair of colors
        color_list = sorted(set(current_colors))
        improved = True
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(color_list)):
                for j in range(i + 1, len(color_list)):
                    if time.time() > deadline - 5:
                        return current_colors, len(set(current_colors))
                    
                    c1 = color_list[i]
                    c2 = color_list[j]
                    
                    # Try merging c2 into c1
                    test_colors = current_colors[:]
                    for idx in range(n):
                        if test_colors[idx] == c2:
                            test_colors[idx] = c1
                    
                    # Check if still valid
                    if compute_violation_count(test_colors) == 0:
                        current_colors = test_colors
                        color_list = sorted(set(current_colors))
                        improved = True
                        break
                
                if improved:
                    break
        
        return current_colors, len(set(current_colors))
    
    def local_search(colors, max_k, deadline, temperature=0.1):
        """Local search with simulated annealing-inspired acceptance."""
        colors = colors[:]
        viol_set = compute_violation_set(colors)
        cur_viol = len(viol_set)
        
        if cur_viol == 0:
            return colors, 0
        
        best_local = colors[:]
        best_local_viol = cur_viol
        no_improve = 0
        max_steps = 8000
        t = temperature
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            if cur_viol == 0:
                return colors, 0
            
            # Pick position biased toward violation-heavy positions
            if random.random() < 0.7 and viol_set:
                # Pick a random violating AP, then a random position in it
                viol_ap_idx = random.choice(list(viol_set))
                i, j, k, l = aps[viol_ap_idx]
                pos = random.choice([i, j, k, l])
            else:
                pos = random.randint(0, n - 1)
            
            old_color = colors[pos]
            
            # Choose new color: prefer colors that appear in violating APs
            if random.random() < 0.5:
                new_color = random.randint(0, max_k - 1)
            else:
                # Sample from colors present in positions of violating APs
                candidate_colors = set()
                for ap_idx in list(viol_set)[:5]:
                    ii, jj, kk, ll = aps[ap_idx]
                    candidate_colors.add(colors[ii])
                    candidate_colors.add(colors[jj])
                    candidate_colors.add(colors[kk])
                    candidate_colors.add(colors[ll])
                if candidate_colors:
                    new_color = random.choice(list(candidate_colors))
                else:
                    new_color = random.randint(0, max_k - 1)
            
            if new_color == old_color:
                continue
            
            # Compute delta in violation count using vectorized check
            delta = 0
            affected_aps = pos_aps_np[pos]
            
            # Temporarily set new color and check affected APs
            colors[pos] = new_color
            colors_arr = np.array(colors, dtype=np.int32)
            
            for ap_idx in affected_aps:
                was_viol = ap_idx in viol_set
                # Check if this AP is now a violation
                c0 = colors_arr[ap_array[ap_idx, 0]]
                c1 = colors_arr[ap_array[ap_idx, 1]]
                c2 = colors_arr[ap_array[ap_idx, 2]]
                c3 = colors_arr[ap_array[ap_idx, 3]]
                
                is_viol = (c0 == c1 == c2 == c3) or \
                         ((c0 == c1 and c2 == c3 and c0 != c2) or
                          (c0 == c2 and c1 == c3 and c0 != c1) or
                          (c0 == c3 and c1 == c2 and c0 != c1))
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            
            colors[pos] = old_color  # Revert
            
            # Acceptance criterion
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.5
            else:
                if t > 0.01:
                    prob = math.exp(-delta / t)
                    accept = random.random() < prob
            
            if accept:
                colors[pos] = new_color
                # Update violation set
                colors_arr = np.array(colors, dtype=np.int32)
                for ap_idx in affected_aps:
                    was_viol = ap_idx in viol_set
                    c0 = colors_arr[ap_array[ap_idx, 0]]
                    c1 = colors_arr[ap_array[ap_idx, 1]]
                    c2 = colors_arr[ap_array[ap_idx, 2]]
                    c3 = colors_arr[ap_array[ap_idx, 3]]
                    
                    is_viol = (c0 == c1 == c2 == c3) or \
                             ((c0 == c1 and c2 == c3 and c0 != c2) or
                              (c0 == c2 and c1 == c3 and c0 != c1) or
                              (c0 == c3 and c1 == c2 and c0 != c1))
                    
                    if was_viol and not is_viol:
                        viol_set.discard(ap_idx)
                    elif not was_viol and is_viol:
                        viol_set.add(ap_idx)
                
                cur_viol = len(viol_set)
                
                if cur_viol < best_local_viol:
                    best_local_viol = cur_viol
                    best_local = colors[:]
                    no_improve = 0
                else:
                    no_improve += 1
                
                t = max(0.001, t * 0.999)
            else:
                no_improve += 1
            
            if no_improve > 1500:
                break
        
        if best_local_viol < cur_viol:
            return best_local, best_local_viol
        return colors, cur_viol
    
    best_colors = None
    best_distinct = 128
    
    # Try different numbers of colors
    for max_k in range(4, 20):
        if time.time() > deadline - 10:
            break
        
        for trial in range(50):
            if time.time() > deadline - 8:
                break
            
            # Random initial coloring
            colors = [random.randint(0, max_k - 1) for _ in range(n)]
            
            # Run local search
            colors, viol_count = local_search(colors, max_k, deadline)
            
            if viol_count == 0:
                # Try to merge colors
                merged_colors, distinct = try_merge_colors(colors, max_k, deadline)
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_colors = merged_colors[:]
                    if best_distinct <= 4:
                        break
            else:
                pass
        
        if best_distinct <= max_k:
            break
    
    # Final polish: try to further reduce colors from best solution
    if best_colors is not None and time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 3:
                break
            polished, distinct = try_merge_colors(best_colors, best_distinct, deadline)
            if distinct < best_distinct:
                best_distinct = distinct
                best_colors = polished
            else:
                break
    
    if best_colors is None:
        # Fallback: greedy construction
        colors = []
        for x in range(1, 129):
            for color in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 1 - 3 * d
                    if a >= 0:
                        existing = [colors[a], colors[a + d], colors[a + 2 * d]]
                        if len(set(existing + [color])) < 3:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    break
            else:
                colors.append(127)
        return colors
    
    # Remap to 0..distinct-1
    color_map = {}
    result = []
    for c in best_colors:
        if c not in color_map:
            color_map[c] = len(color_map)
        result.append(color_map[c])
    
    return result
