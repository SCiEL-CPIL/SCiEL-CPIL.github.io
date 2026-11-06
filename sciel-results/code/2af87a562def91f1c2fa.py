import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 55.0  # Safety margin before 60s kill
    
    N = 128
    
    # Precompute all 4-term APs
    # (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(aps)
    if num_aps == 0:
        return [0] * N
        
    aps_np = np.array(aps, dtype=np.int32)
    
    # For each position, store indices of APs it belongs to
    pos_ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_ap_indices[pos].append(idx)

    def count_violations(colors_np):
        """
        Returns the number of 4-term APs that have fewer than 3 distinct colors.
        """
        # Get colors at all positions of all APs
        ap_colors = colors_np[aps_np]  # shape (num_aps, 4)
        
        # Distinct colors count < 3 means:
        # 1. All 4 same
        # 2. 3 same, 1 different
        # 3. 2 pairs same (e.g., A A B B)
        # 4. 2 same, 2 different? No, that's 3 distinct.
        # Wait, distinct < 3 means distinct == 1 or distinct == 2.
        
        c0, c1, c2, c3 = ap_colors[:, 0], ap_colors[:, 1], ap_colors[:, 2], ap_colors[:, 3]
        
        # Check for 1 distinct color
        all_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # Check for 2 distinct colors
        # This happens if any two are equal and the other two are equal to one of them,
        # or if we can partition into two sets.
        # Easier: distinct == 2 iff not all_same and (c0==c1 or c0==c2 or c0==c3) and 
        #         (c1==c2 or c1==c3 or c2==c3) ... this is getting complicated.
        
        # Alternative: distinct == 2 iff (c0==c1 and (c2==c3 or c2==c0 or c3==c0)) OR ...
        # Let's use a simpler logical check for "at most 2 distinct colors".
        # A set of 4 elements has <= 2 distinct values iff:
        # (c0==c1 and c2==c3) OR (c0==c2 and c1==c3) OR (c0==c3 and c1==c2) OR
        # (c0==c1 and c2==c0) OR (c0==c1 and c3==c0) OR ...
        
        # Actually, distinct < 3 is equivalent to:
        # (c0 == c1 and c2 == c3) OR
        # (c0 == c2 and c1 == c3) OR
        # (c0 == c3 and c1 == c2) OR
        # (c0 == c1 and c2 == c0) OR  # covers 3 same 1 diff where c0 is the majority? No.
        
        # Let's just check if the number of unique values is < 3.
        # We can do this by checking all pairs. If any pair is different, they are not all same.
        # If all pairs are same, distinct=1.
        # If some pairs same, some different, distinct=2 or 3 or 4.
        # Distinct=2 iff the set of values has size 2.
        
        # Vectorized way:
        # Create a mask for each possible pair equality
        # If distinct < 3, then the values must be drawn from a set of size 1 or 2.
        
        # Let's try:
        # v0, v1, v2, v3
        # If v0==v1 and v2==v3: distinct <= 2
        # If v0==v2 and v1==v3: distinct <= 2
        # If v0==v3 and v1==v2: distinct <= 2
        # If v0==v1 and v2==v0: distinct <= 2 (implies v0==v1==v2, v3 anything? No, if v3!=v0, distinct=2)
        # Actually, if ANY of the following is true, distinct <= 2:
        # 1. v0==v1 and v2==v3
        # 2. v0==v2 and v1==v3
        # 3. v0==v3 and v1==v2
        # 4. v0==v1 and v2==v0  (implies v0,v1,v2 same)
        # 5. v0==v1 and v3==v0  (implies v0,v1,v3 same)
        # 6. v0==v2 and v1==v0  (implies v0,v2,v1 same)
        # 7. v0==v2 and v3==v0  (implies v0,v2,v3 same)
        # 8. v0==v3 and v1==v0  (implies v0,v3,v1 same)
        # 9. v0==v3 and v2==v0  (implies v0,v3,v2 same)
        # 10. v1==v2 and v0==v1
        # 11. v1==v2 and v3==v1
        # 12. v1==v3 and v0==v1
        # 13. v1==v3 and v2==v1
        # 14. v2==v3 and v0==v2
        # 15. v2==v3 and v1==v2
        
        # This is tedious. Let's use a simpler property.
        # Distinct < 3 iff (v0==v1 and v2==v3) OR (v0==v2 and v1==v3) OR (v0==v3 and v1==v2) 
        #                  OR (v0==v1 and v1==v2) OR (v0==v1 and v1==v3) OR (v0==v1 and v2==v3) ...
        
        # Actually, the condition "distinct < 3" is equivalent to "the 4 values can be covered by 2 colors".
        # This is true if and only if:
        # (v0==v1 and v2==v3) OR
        # (v0==v2 and v1==v3) OR
        # (v0==v3 and v1==v2) OR
        # (v0==v1 and v2==v0) OR  # v0,v1,v2 same
        # (v0==v1 and v3==v0) OR  # v0,v1,v3 same
        # (v0==v2 and v1==v0) OR  # v0,v2,v1 same
        # (v0==v2 and v3==v0) OR  # v0,v2,v3 same
        # (v0==v3 and v1==v0) OR  # v0,v3,v1 same
        # (v0==v3 and v2==v0) OR  # v0,v3,v2 same
        # (v1==v2 and v0==v1) OR  # v1,v2,v0 same
        # (v1==v2 and v3==v1) OR  # v1,v2,v3 same
        # (v1==v3 and v0==v1) OR  # v1,v3,v0 same
        # (v1==v3 and v2==v1) OR  # v1,v3,v2 same
        # (v2==v3 and v0==v2) OR  # v2,v3,v0 same
        # (v2==v3 and v1==v2)     # v2,v3,v1 same
        
        # We can simplify this.
        # If v0==v1, then we need v2,v3 to be in {v0} or {v0, x}.
        # If v0!=v1, then v2,v3 must be in {v0, v1}.
        
        # Let's just compute the number of distinct elements for each AP.
        # Since N is small (128) and num_aps is ~2000, we can do this efficiently.
        
        # Method: 
        # distinct_count = 1 + (v0!=v1) + (v0!=v2 and v1!=v2) + (v0!=v3 and v1!=v3 and v2!=v3)
        # This is O(1) per AP.
        
        d01 = c0 != c1
        d02 = c0 != c2
        d03 = c0 != c3
        d12 = c1 != c2
        d13 = c1 != c3
        d23 = c2 != c3
        
        # Count distinct
        # Start with 1 (v0)
        # Add 1 if v1 != v0
        # Add 1 if v2 != v0 and v2 != v1
        # Add 1 if v3 != v0 and v3 != v1 and v3 != v2
        
        distinct = 1 + d01 + (d02 & d12) + (d03 & d13 & d23)
        
        violations = distinct < 3
        return int(np.sum(violations)), violations

    # Local search function
    def local_search(colors_np, max_steps=20000):
        """
        Perform local search to reduce violations to 0.
        Returns (success, colors_np)
        """
        num_violations, violation_flags = count_violations(colors_np)
        
        if num_violations == 0:
            return True, colors_np
            
        best_colors = colors_np.copy()
        best_violations = num_violations
        
        for step in range(max_steps):
            if time.time() > DEADLINE:
                break
                
            if num_violations == 0:
                return True, colors_np
                
            if num_violations < best_violations:
                best_violations = num_violations
                best_colors = colors_np.copy()
                
            # Pick a random violated AP
            violated_idx = np.where(violation_flags)[0]
            if len(violated_idx) == 0:
                break
                
            ai = int(random.choice(violated_idx))
            ap = aps_np[ai]
            
            # Pick a random position in the AP to change
            pos_in_ap = random.randint(0, 3)
            idx = int(ap[pos_in_ap])
            
            # Try changing this position to a random color
            # To be efficient, we can try colors that appear in the AP or random ones
            # Let's just try a few random colors
            current_color = colors_np[idx]
            
            # Determine the range of colors to try
            # We don't know the max color yet, but we can use 0..127
            # Or we can track the max color used.
            # For now, let's assume colors are in 0..K-1.
            # We'll pass K to this function or infer it.
            # Let's infer max color from the array.
            max_color = int(np.max(colors_np))
            
            # Try a few random colors
            for _ in range(5):
                new_color = random.randint(0, max_color)
                if new_color == current_color:
                    continue
                    
                # Save old state
                old_flags = []
                affected_aps = pos_ap_indices[idx]
                for aai in affected_aps:
                    old_flags.append((aai, bool(violation_flags[aai])))
                
                # Apply change
                colors_np[idx] = new_color
                
                # Update violations for affected APs
                delta = 0
                for aai in affected_aps:
                    ap2 = aps_np[aai]
                    c0, c1, c2, c3 = colors_np[ap2[0]], colors_np[ap2[1]], colors_np[ap2[2]], colors_np[ap2[3]]
                    
                    # Recalculate distinct for this AP
                    d01 = c0 != c1
                    d02 = c0 != c2
                    d03 = c0 != c3
                    d12 = c1 != c2
                    d13 = c1 != c3
                    d23 = c2 != c3
                    distinct = 1 + d01 + (d02 & d12) + (d03 & d13 & d23)
                    is_violated = distinct < 3
                    
                    was_violated = violation_flags[aai]
                    
                    if was_violated and not is_violated:
                        delta -= 1
                    elif not was_violated and is_violated:
                        delta += 1
                    violation_flags[aai] = is_violated
                
                num_violations += delta
                
                # Accept if better or equal with some probability
                if delta < 0 or (delta == 0 and random.random() < 0.2):
                    # Accept
                    break
                else:
                    # Revert
                    colors_np[idx] = current_color
                    for aai, old_flag in old_flags:
                        violation_flags[aai] = old_flag
                    num_violations -= delta
                    # Try next color for this position
            
        if num_violations < best_violations:
            return best_violations == 0, best_colors
        return num_violations == 0, colors_np

    # Strategy:
    # 1. Start with a random coloring using K colors.
    # 2. Run local search to fix violations.
    # 3. If valid, try to reduce K by merging colors.
    
    best_valid_colors = None
    best_num_colors = 128  # Max possible
    
    # Try different starting K values
    # We want to minimize colors, so we should try small K first?
    # But small K is harder to satisfy.
    # Let's start with K=10, 8, 6, 5, 4, 3.
    
    # Actually, let's start with a larger K to ensure we find *a* solution, then reduce.
    # Or, we can just run local search on a random K=10 assignment.
    
    # Let's try K=5 first. If it fails, try K=6, etc.
    # But "minimize colors" suggests we should find the minimum K.
    # The problem asks to minimize the number of labels actually used.
    
    # Let's try to find a valid solution with K=3.
    # If not, K=4, etc.
    
    # However, local search with fixed K might get stuck.
    # We can allow the number of colors to grow if needed, but we want to minimize.
    
    # Better approach:
    # 1. Generate random coloring with K colors.
    # 2. Run local search.
    # 3. If successful, we have a valid coloring with <= K colors.
    # 4. Check actual distinct colors.
    # 5. Try to reduce distinct colors by remapping.
    
    # Let's iterate K from 3 upwards.
    
    for K in [3, 4, 5, 6, 8, 10]:
        if time.time() > DEADLINE:
            break
            
        found = False
        for attempt in range(20): # 20 restarts for this K
            if time.time() > DEADLINE:
                break
                
            colors_np = np.random.randint(0, K, N, dtype=np.int32)
            
            success, final_colors = local_search(colors_np, max_steps=5000)
            
            if success:
                # Count actual distinct colors
                distinct_colors = len(set(final_colors))
                
                if distinct_colors < best_num_colors:
                    best_num_colors = distinct_colors
                    best_valid_colors = final_colors.copy()
                    
                    # Try to reduce further by remapping
                    # We can try to merge colors in the current solution
                    # But first, let's see if we can find a solution with smaller K directly.
                    # Since we are iterating K upwards, if we found a solution with K=3, we are done (min possible is 3? No, could be 2? No, 4-term AP needs 3 colors. So min is 3).
                    # Wait, is 3 colors enough?
                    # If we find a valid solution with K=3, we can stop.
                    if distinct_colors == 3:
                        return [int(c) for c in best_valid_colors]
                        
                    # If we found a solution with K=4, we might be able to reduce to 3.
                    # Let's try to reduce the current solution.
                    # Remap one color to another and run local search.
                    used_colors = list(set(final_colors))
                    # Try merging each pair
                    for i in range(len(used_colors)):
                        for j in range(len(used_colors)):
                            if i == j:
                                continue
                            
                            c1 = used_colors[i]
                            c2 = used_colors[j]
                            
                            # Map c2 to c1
                            test_colors = final_colors.copy()
                            test_colors[test_colors == c2] = c1
                            
                            # Run local search on this modified solution
                            # We need to handle the fact that colors might be sparse
                            # But local_search uses np.max, which is fine.
                            
                            # We need a version of local_search that takes a specific array
                            # and doesn't change the max color range unnecessarily?
                            # My local_search uses np.max(colors_np) to determine range.
                            # If we remove a color, max might stay same or decrease.
                            
                            success2, reduced_colors = local_search(test_colors, max_steps=2000)
                            
                            if success2:
                                distinct_reduced = len(set(reduced_colors))
                                if distinct_reduced < best_num_colors:
                                    best_num_colors = distinct_reduced
                                    best_valid_colors = reduced_colors.copy()
                                    
                                if distinct_reduced == 3:
                                    return [int(c) for c in best_valid_colors]
                
        # If we found a valid solution with this K, we can try the next smaller K?
        # No, we are iterating K upwards.
        # If K=3 failed, we try K=4.
        # If K=4 succeeded, we try to reduce to 3 via remapping.
        # If that fails, we accept K=4 solution and move to K=5? No, K=4 is better than K=5.
        # So if we find a solution with K, we should try to reduce it.
        # If we can't reduce it, we keep it as the best candidate and continue to see if a smaller K works?
        # But we are going 3, 4, 5...
        # If K=3 fails, K=4 might succeed.
        # If K=4 succeeds, we have a candidate with 4 colors.
        # We should not try K=5 because 4 is better.
        # So, if we find a valid solution for K, we should break the K loop and try to reduce it?
        # Or just keep it and see if a smaller K (which we already tried) works?
        # Since we tried smaller Ks first, if they failed, we accept the first successful K.
        
        if best_valid_colors is not None and best_num_colors <= K:
            # We found a solution with K colors (or less).
            # Since we iterate K from 3, this is the minimal K we've found so far.
            # We can break and try to optimize this specific solution further?
            # Or just return it?
            # Let's try to optimize the current best_valid_colors further with more local search time.
            break

    if best_valid_colors is None:
        # Fallback: Greedy
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                       for d in range(1, (x-1)//3 + 1))
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                       for d in range(1, (x-1)//2 + 1))):
                    colors.append(color)
                    break
        return colors

    return [int(c) for c in best_valid_colors]

# EVOLVE-BLOCK-END
