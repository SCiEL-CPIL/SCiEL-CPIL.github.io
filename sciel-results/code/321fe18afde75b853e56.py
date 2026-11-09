import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    n = 128
    # Precompute all 4-term APs
    aps_list = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            aps_list.append((a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d))
    
    aps = np.array(aps_list, dtype=np.int32)  # shape (num_aps, 4)
    num_aps = len(aps)
    
    # Map index to list of (ap_idx, pos_in_ap)
    idx_to_aps_pos = [[] for _ in range(n)]
    for i, ap in enumerate(aps_list):
        for pos, idx in enumerate(ap):
            idx_to_aps_pos[idx].append((i, pos))
    
    # Convert to numpy arrays for faster access
    idx_to_aps_pos_arr = []
    for i in range(n):
        if idx_to_aps_pos[i]:
            ap_idxs = np.array([x[0] for x in idx_to_aps_pos[i]], dtype=np.int32)
            positions = np.array([x[1] for x in idx_to_aps_pos[i]], dtype=np.int32)
            idx_to_aps_pos_arr.append((ap_idxs, positions))
        else:
            idx_to_aps_pos_arr.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))

    def init_matches(colors):
        """Compute matches for all APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        
        matches = ((c0 == c1).astype(np.int8) + 
                   (c0 == c2).astype(np.int8) + 
                   (c0 == c3).astype(np.int8) + 
                   (c1 == c2).astype(np.int8) + 
                   (c1 == c3).astype(np.int8) + 
                   (c2 == c3).astype(np.int8))
        return matches

    def count_distinct_in_ap(colors, ap_idx):
        """Count distinct colors in a single AP."""
        ap = aps[ap_idx]
        c = colors[ap]
        return len(set(c.tolist()))

    def check_all_valid(colors):
        """Check if all APs have at least 3 distinct colors."""
        for i in range(num_aps):
            if count_distinct_in_ap(colors, i) < 3:
                return False
        return True

    def apply_swap(colors, swap_color_a, swap_color_b, ap_matches, idx_to_aps_pos_arr):
        """
        Swap all occurrences of swap_color_a to swap_color_b and vice versa.
        Updates ap_matches in place.
        Returns (delta_bad, success) where success is True if the swap was applied.
        """
        # First, verify the swap doesn't create any bad APs
        # We need to check: for each AP, after swapping, does it still have >= 3 distinct colors?
        
        # Get all positions with swap_color_a or swap_color_b
        positions_a = np.where(colors == swap_color_a)[0]
        positions_b = np.where(colors == swap_color_b)[0]
        
        if len(positions_a) == 0 or len(positions_b) == 0:
            return 0, False
        
        # For each affected AP, check if the swap maintains validity
        # An AP is affected if it contains at least one position with color a or b
        affected_aps = set()
        for pos in positions_a:
            ap_idxs, _ = idx_to_aps_pos_arr[pos]
            affected_aps.update(ap_idxs.tolist())
        for pos in positions_b:
            ap_idxs, _ = idx_to_aps_pos_arr[pos]
            affected_aps.update(ap_idxs.tolist())
        
        # Check each affected AP
        for ap_idx in affected_aps:
            ap = aps[ap_idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            # Apply swap
            if c0 == swap_color_a: c0 = swap_color_b
            elif c0 == swap_color_b: c0 = swap_color_a
            if c1 == swap_color_a: c1 = swap_color_b
            elif c1 == swap_color_b: c1 = swap_color_a
            if c2 == swap_color_a: c2 = swap_color_b
            elif c2 == swap_color_b: c2 = swap_color_a
            if c3 == swap_color_a: c3 = swap_color_b
            elif c3 == swap_color_b: c3 = swap_color_a
            
            # Check distinct count
            distinct = len(set([c0, c1, c2, c3]))
            if distinct < 3:
                return 0, False
        
        # Swap is valid, apply it
        delta_bad = 0
        for pos in positions_a:
            colors[pos] = swap_color_b
        for pos in positions_b:
            colors[pos] = swap_color_a
        
        # Update ap_matches for affected APs
        for ap_idx in affected_aps:
            ap = aps[ap_idx]
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            new_matches = 0
            if c0 == c1: new_matches += 1
            if c0 == c2: new_matches += 1
            if c0 == c3: new_matches += 1
            if c1 == c2: new_matches += 1
            if c1 == c3: new_matches += 1
            if c2 == c3: new_matches += 1
            
            old_matches = ap_matches[ap_idx]
            was_bad = old_matches >= 2
            is_bad = new_matches >= 2
            
            if was_bad and not is_bad:
                delta_bad -= 1
            elif is_bad and not was_bad:
                delta_bad += 1
            
            ap_matches[ap_idx] = new_matches
        
        return delta_bad, True

    def try_eliminate_color(colors, target_color, ap_matches, idx_to_aps_pos_arr, deadline):
        """
        Try to eliminate target_color by swapping it with other colors.
        Returns True if successful (target_color no longer used).
        """
        # Get positions with target_color
        positions_target = np.where(colors == target_color)[0]
        if len(positions_target) == 0:
            return True  # Already eliminated
        
        # Try swapping target_color with each other color
        other_colors = [c for c in range(128) if c != target_color and np.any(colors == c)]
        
        for other_color in other_colors:
            if time.time() > deadline - 2:
                return False
            
            # Save state
            colors_copy = colors.copy()
            ap_matches_copy = ap_matches.copy()
            
            # Try the swap
            delta, success = apply_swap(colors, target_color, other_color, ap_matches, idx_to_aps_pos_arr)
            
            if success:
                # Check if target_color is still used
                if not np.any(colors == target_color):
                    return True
                else:
                    # Swap was applied but target_color still exists
                    # This means some positions had target_color and got swapped to other_color,
                    # but other positions had other_color and got swapped to target_color
                    # Revert
                    colors[:] = colors_copy
                    ap_matches[:] = ap_matches_copy
            else:
                # Swap failed, revert
                colors[:] = colors_copy
                ap_matches[:] = ap_matches_copy
        
        return False

    def greedy_color_elimination(colors, ap_matches, idx_to_aps_pos_arr, deadline):
        """
        Greedily try to eliminate colors one by one.
        Returns the best coloring found.
        """
        best_colors = colors.copy()
        best_num_colors = len(set(colors.tolist()))
        
        # Try to eliminate colors from highest to lowest
        for target_color in sorted(set(colors.tolist()), reverse=True):
            if time.time() > deadline - 2:
                break
            
            if try_eliminate_color(colors, target_color, ap_matches, idx_to_aps_pos_arr, deadline):
                new_num_colors = len(set(colors.tolist()))
                if new_num_colors < best_num_colors:
                    best_colors = colors.copy()
                    best_num_colors = new_num_colors
        
        return best_colors, best_num_colors

    # Start with a valid 4-coloring using 2-adic valuation
    # v2(i) = exponent of 2 in i
    def v2(x):
        if x == 0:
            return 0
        count = 0
        while x % 2 == 0:
            count += 1
            x //= 2
        return count
    
    # 2-adic coloring: color = v2(i) mod 4 for i in 1..128
    # This gives a valid 4-coloring for 4-term APs
    colors_2adic = np.array([v2(i) % 4 for i in range(1, n + 1)], dtype=np.int32)
    
    # Verify validity
    ap_matches = init_matches(colors_2adic)
    bad_count = int(np.sum(ap_matches >= 2))
    
    if bad_count == 0:
        # Valid 4-coloring, try to reduce
        colors = colors_2adic.copy()
        ap_matches = init_matches(colors)
        
        best_colors, best_num_colors = greedy_color_elimination(colors, ap_matches, idx_to_aps_pos_arr, deadline)
        
        # Try more aggressive search if we have time
        if best_num_colors > 3 and time.time() < deadline - 10:
            # Try local search to find a 3-coloring
            for restart in range(50):
                if time.time() > deadline - 3:
                    break
                
                # Start from best found and perturb
                colors = best_colors.copy()
                ap_matches = init_matches(colors)
                
                T = 1.0
                T_min = 0.01
                cooling = 0.9995
                max_iters = 20000
                
                for it in range(max_iters):
                    if time.time() > deadline - 3:
                        break
                    
                    i = np.random.randint(n)
                    old_color = colors[i]
                    new_color = np.random.randint(0, 4)
                    
                    if new_color == old_color:
                        continue
                    
                    # Apply move
                    delta = 0
                    ap_idxs, positions = idx_to_aps_pos_arr[i]
                    
                    for idx_in_list in range(len(ap_idxs)):
                        ap_idx = ap_idxs[idx_in_list]
                        pos = positions[idx_in_list]
                        
                        old_matches = ap_matches[ap_idx]
                        ap = aps[ap_idx]
                        
                        c0 = colors[ap[0]]
                        c1 = colors[ap[1]]
                        c2 = colors[ap[2]]
                        c3 = colors[ap[3]]
                        
                        if pos == 0:
                            nc0, nc1, nc2, nc3 = new_color, c1, c2, c3
                        elif pos == 1:
                            nc0, nc1, nc2, nc3 = c0, new_color, c2, c3
                        elif pos == 2:
                            nc0, nc1, nc2, nc3 = c0, c1, new_color, c3
                        else:
                            nc0, nc1, nc2, nc3 = c0, c1, c2, new_color
                            
                        new_matches = 0
                        if nc0 == nc1: new_matches += 1
                        if nc0 == nc2: new_matches += 1
                        if nc0 == nc3: new_matches += 1
                        if nc1 == nc2: new_matches += 1
                        if nc1 == nc3: new_matches += 1
                        if nc2 == nc3: new_matches += 1
                        
                        ap_matches[ap_idx] = new_matches
                        
                        was_bad = old_matches >= 2
                        is_bad = new_matches >= 2
                        
                        if was_bad and not is_bad:
                            delta -= 1
                        elif is_bad and not was_bad:
                            delta += 1
                    
                    colors[i] = new_color
                    current_bad = int(np.sum(ap_matches >= 2))
                    
                    if delta < 0 or (delta == 0 and random.random() < 0.3) or (delta > 0 and random.random() < np.exp(-delta / T)):
                        pass  # Accept
                    else:
                        colors[i] = old_color
                        # Revert
                        for idx_in_list in range(len(ap_idxs)):
                            ap_idx = ap_idxs[idx_in_list]
                            pos = positions[idx_in_list]
                            ap = aps[ap_idx]
                            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                            new_matches = 0
                            if c0 == c1: new_matches += 1
                            if c0 == c2: new_matches += 1
                            if c0 == c3: new_matches += 1
                            if c1 == c2: new_matches += 1
                            if c1 == c3: new_matches += 1
                            if c2 == c3: new_matches += 1
                            ap_matches[ap_idx] = new_matches
                    
                    T *= cooling
                    if T < T_min:
                        T = T_min
                
                # Check result
                if int(np.sum(ap_matches >= 2)) == 0:
                    num_used = len(set(colors.tolist()))
                    if num_used < best_num_colors:
                        best_colors = colors.copy()
                        best_num_colors = num_used
        
        return [int(c) for c in best_colors]
    
    # Fallback: Greedy construction
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
