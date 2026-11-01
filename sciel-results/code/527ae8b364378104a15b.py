import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin before 120s limit

    total = q ** n
    # For q=3, n=7, total=2187. Fits in a Python int (arbitrary precision).
    
    # Precompute balls as bitmasks
    # ball_mask[i] is an int where bit j is 1 if word j is in ball of word i
    ball_masks = [0] * total
    for i in range(total):
        w = [0] * n
        temp = i
        for j in range(n - 1, -1, -1):
            w[j] = temp % q
            temp //= q
            
        mask = 1 << i
        # Add neighbors
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_w = w.copy()
                new_w[coord] = v
                # Convert new_w back to index
                idx = 0
                for k in range(n):
                    idx = idx * q + new_w[k]
                mask |= (1 << idx)
        ball_masks[i] = mask

    # Greedy Set Cover
    def greedy_cover(order):
        covered_mask = 0
        code = []
        remaining_mask = (1 << total) - 1
        
        # Pre-sort or use order
        for c in order:
            if (covered_mask | ball_masks[c]) != covered_mask:
                # Check if it adds new coverage
                if (ball_masks[c] & remaining_mask) != 0:
                    code.append(c)
                    covered_mask |= ball_masks[c]
                    remaining_mask &= ~ball_masks[c]
                    if remaining_mask == 0:
                        break
        return code

    # Local Search Helpers
    def compute_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            # Iterate bits in ball_masks[c]
            mask = ball_masks[c]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                mask ^= lsb
        return counts

    def get_lost_mask(c1, c2, code_set, counts):
        # Returns mask of points that will be uncovered if c1, c2 are removed
        # A point p is lost if it is covered by c1 or c2, and the number of 
        # codewords in code_set covering it is exactly the number of 
        # {c1, c2} covering it.
        
        # Union of balls
        union_mask = ball_masks[c1] | ball_masks[c2]
        # Intersect with code_set balls? No, we use counts.
        
        lost_mask = 0
        mask = union_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            # Count how many of c1, c2 cover idx
            k = 0
            if (ball_masks[c1] >> idx) & 1:
                k += 1
            if (ball_masks[c2] >> idx) & 1:
                k += 1
            
            # If total count equals k, then removing c1, c2 removes all coverage
            if counts[idx] == k:
                lost_mask |= lsb
            
            mask ^= lsb
        return lost_mask

    def is_subset(subset_mask, superset_mask):
        return (subset_mask & ~superset_mask) == 0

    # Main Loop
    best_code = None
    best_size = total + 1
    best_counts = None

    # Multiple greedy restarts
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(total))
        random.shuffle(order)
        code = greedy_cover(order)
        code_set = set(code)
        
        # Delete redundant
        changed = True
        while changed:
            changed = False
            counts = compute_counts(code_set)
            for c in list(code_set):
                # Check if any point in ball(c) has count 1
                # If not, c is redundant
                ball = ball_masks[c]
                mask = ball
                redundant = True
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                if redundant:
                    code_set.remove(c)
                    changed = True
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            best_counts = compute_counts(code_set)

    if best_code is None:
        # Fallback
        best_code = list(range(0, total, max(1, total // 186)))
        best_size = len(best_code)
        best_counts = compute_counts(set(best_code))

    current_code = set(best_code)
    counts = best_counts.copy()
    best_size = len(current_code)
    best_code_list = list(current_code)
    
    no_improve_streak = 0
    max_streak = 10000
    
    while time.time() < deadline - 5.0 and no_improve_streak < max_streak:
        improved = False
        
        # 1. Deletions
        changed = True
        while changed:
            changed = False
            # Recompute counts if needed? 
            # We maintain counts.
            for c in list(current_code):
                ball = ball_masks[c]
                mask = ball
                redundant = True
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                if redundant:
                    current_code.remove(c)
                    # Update counts
                    mask = ball
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        mask ^= lsb
                    changed = True
                    improved = True
                    break # Restart deletion loop
            
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 2. 2-for-1 Swaps
        # We try to find c1, c2 in current_code and c_add not in current_code
        # such that lost_mask(c1, c2) is subset of ball(c_add)
        
        code_list = list(current_code)
        random.shuffle(code_list)
        
        found_swap = False
        # Limit pairs to try to save time
        max_pairs = 1000
        pairs_tried = 0
        
        for i in range(len(code_list)):
            if pairs_tried > max_pairs:
                break
            c1 = code_list[i]
            for j in range(i + 1, len(code_list)):
                if pairs_tried > max_pairs:
                    break
                pairs_tried += 1
                c2 = code_list[j]
                
                lost_mask = get_lost_mask(c1, c2, current_code, counts)
                
                if lost_mask == 0:
                    # We can remove both without adding? 
                    # This implies we can reduce size by 2.
                    # But we must ensure we don't break coverage.
                    # If lost_mask is 0, no point is uniquely covered by c1 or c2.
                    # So we can remove c1 and c2.
                    current_code.remove(c1)
                    current_code.remove(c2)
                    # Update counts
                    for c in [c1, c2]:
                        mask = ball_masks[c]
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            mask ^= lsb
                    improved = True
                    found_swap = True
                    break
                
                # Find c_add
                # We can iterate through all candidates not in current_code
                # To speed up, we can just iterate all 0..total-1 and check
                # if it's in current_code (O(1) set check)
                
                # Optimization: Only check candidates that cover at least one point in lost_mask?
                # Or just scan all. 2187 is small.
                
                for c_add in range(total):
                    if c_add in current_code:
                        continue
                    if is_subset(lost_mask, ball_masks[c_add]):
                        # Valid swap
                        current_code.remove(c1)
                        current_code.remove(c2)
                        current_code.add(c_add)
                        
                        # Update counts
                        for c in [c1, c2]:
                            mask = ball_masks[c]
                            while mask:
                                lsb = mask & -mask
                                idx = lsb.bit_length() - 1
                                counts[idx] -= 1
                                mask ^= lsb
                        
                        mask = ball_masks[c_add]
                        while mask:
                            lsb = mask & -mask
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            mask ^= lsb
                        
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
        
        if improved:
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code_list = list(current_code)
            no_improve_streak = 0
            continue

        # 3. 1-for-1 Swaps (to escape local minima)
        # Find c1 in code, c_add not in code, such that lost_mask(c1, c1) is subset of ball(c_add)
        # Note: get_lost_mask(c1, c1) is just points uniquely covered by c1.
        
        found_swap = False
        max_pairs = 200
        pairs_tried = 0
        code_list = list(current_code)
        random.shuffle(code_list)
        
        for c1 in code_list:
            if pairs_tried > max_pairs:
                break
            pairs_tried += 1
            lost_mask = get_lost_mask(c1, c1, current_code, counts)
            
            if lost_mask == 0:
                # Redundant, should have been caught in deletion
                current_code.remove(c1)
                mask = ball_masks[c1]
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    mask ^= lsb
                improved = True
                found_swap = True
                break
            
            for c_add in range(total):
                if c_add in current_code:
                    continue
                if is_subset(lost_mask, ball_masks[c_add]):
                    current_code.remove(c1)
                    current_code.add(c_add)
                    
                    # Update counts
                    mask = ball_masks[c1]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        mask ^= lsb
                    
                    mask = ball_masks[c_add]
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        mask ^= lsb
                    
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
        
        if improved:
            no_improve_streak = 0
            continue

        no_improve_streak += 1

    # Final cleanup
    if time.time() < deadline - 2.0:
        # One more pass of deletions
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            for c in list(current_code):
                ball = ball_masks[c]
                mask = ball
                redundant = True
                while mask:
                    lsb = mask & -mask
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        redundant = False
                        break
                    mask ^= lsb
                if redundant:
                    current_code.remove(c)
                    mask = ball
                    while mask:
                        lsb = mask & -mask
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        mask ^= lsb
                    changed = True
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code_list = list(current_code)

    # Convert back to list of lists
    result = []
    for idx in best_code_list:
        w = []
        temp_idx = idx
        for i in range(n):
            w.append(temp_idx % q)
            temp_idx //= q
        w.reverse()
        result.append(w)
    
    return result
