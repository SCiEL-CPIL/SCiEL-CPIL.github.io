import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    # 1. Initial Structured Greedy Construction
    # Strategy: Instead of fixed order, pick the next number that maximizes "slack".
    # We maintain counts of sums. A sum s is "available" if counts[s] < 2.
    # When considering adding x, we check how many new sums (x+y for y in chosen, and 2x)
    # are currently available. We prefer x that uses the most available sums (spreading out).
    
    def build_structured_greedy():
        chosen = []
        counts = {}
        # We will iterate through a sorted list, but at each step, we might look ahead
        # or just use a heuristic score. 
        # To keep it fast, we'll do a "look-ahead" greedy.
        # Actually, a simpler structured approach: 
        # Sort candidates. Iterate. But instead of just "can I add", 
        # we can prioritize numbers that don't create "hot spots".
        
        # Let's try a specific heuristic: 
        # Maintain a list of available numbers.
        # At each step, for each candidate x, calculate "cost" = number of sums s=x+y (y in chosen) where counts[s] >= 2.
        # If cost > 0, x is invalid.
        # Among valid x, pick the one that minimizes the "congestion" it creates.
        # Congestion: for each new sum s, if counts[s] becomes 1, it's "half-full". If it becomes 2, it's "full".
        # We want to minimize the number of sums that become "full" (count 2) or "half-full" (count 1) if possible?
        # Actually, we want to maximize the number of sums that stay at 0? No, we are adding to them.
        # We want to minimize the increase in "potential conflicts".
        
        # Let's try: 
        # 1. Start with 1.
        # 2. For each next step, scan remaining candidates.
        # 3. Filter out invalid ones (any sum hits count >= 2).
        # 4. Among valid ones, score them by: 
        #    - Primary: Larger numbers are generally better for range coverage? 
        #    - Secondary: Minimize the number of sums that will reach count 1 (creating new potential conflicts later).
        #    - Tertiary: Minimize the number of sums that will reach count 2 (wasting capacity).
        
        # This O(N^2 * K) where K is set size might be too slow if K is large and N=1000.
        # N=1000, K~90. 1000*90 = 90k ops per step. 90 steps. 8.1M ops. Very fast in Python.
        
        remaining = set(range(1, 1001))
        if not remaining:
            return []
        
        # Force start with 1 to maximize range? Or pick best?
        # Let's pick the best from all.
        # Actually, starting with 1 is usually good for B2 sets to maximize size in [1,N].
        # Let's just evaluate all 1000 candidates for the first pick.
        
        while remaining and time.time() < deadline - 10:
            best_x = None
            best_score = None
            # We need to check validity against current 'chosen'.
            # If chosen is empty, any x is valid.
            
            if not chosen:
                # Pick the smallest? Largest? 
                # For B2[2], starting small allows more room. 
                # Let's pick 1.
                best_x = 1
            else:
                # Evaluate candidates
                # Optimization: Only check a subset? No, check all remaining.
                # But we can early break if we find a "perfect" score?
                
                # Score components:
                # 1. Validity (Must be valid)
                # 2. Minimize # sums that become count 1 (new "half-full" sums)
                # 3. Minimize # sums that become count 2 (new "full" sums)
                # 4. Prefer larger x? (To cover higher range)
                
                min_new_half = float('inf')
                min_new_full = float('inf')
                valid_candidates = []
                
                for x in remaining:
                    # Check validity
                    valid = True
                    new_half = 0
                    new_full = 0
                    
                    # Check x + y for y in chosen
                    for y in chosen:
                        s = x + y
                        c = counts.get(s, 0)
                        if c >= 2:
                            valid = False
                            break
                        if c == 1:
                            new_full += 1
                        else: # c == 0
                            new_half += 1
                    
                    if not valid:
                        continue
                        
                    # Check 2x
                    s2 = 2 * x
                    c2 = counts.get(s2, 0)
                    if c2 >= 2:
                        valid = False
                    else:
                        if c2 == 1:
                            new_full += 1
                        else:
                            new_half += 1
                    
                    if valid:
                        valid_candidates.append((x, new_half, new_full))
                        
                if not valid_candidates:
                    break # No more can be added
                
                # Select best from valid_candidates
                # Criteria: 
                # 1. Minimize new_full (most important, as it blocks future additions)
                # 2. Minimize new_half (less important)
                # 3. Maximize x (prefer larger numbers to extend range)
                
                # Sort: (new_full, new_half, -x)
                valid_candidates.sort(key=lambda t: (t[1], t[2], -t[0]))
                best_x, _, _ = valid_candidates[0]
            
            # Add best_x
            chosen.append(best_x)
            remaining.remove(best_x)
            
            # Update counts
            for y in chosen:
                if y == best_x:
                    continue # Skip self-pair for now, handle 2x separately? 
                    # Wait, the loop "for y in chosen" includes x if x is in chosen.
                    # My validity check above handled 2x separately.
                    # So here, I should only add pairs (x, y) where y != x?
                    # Or just iterate y in chosen, if y == x, skip?
                    # The pair {x,x} is sum 2x.
                    # The pairs {x,y} for y!=x are sums x+y.
                    pass
            
            # Correct update:
            # Pairs are {a,b} with a<=b.
            # New pairs involving x:
            # 1. {x, x} -> sum 2x
            # 2. {x, y} for y in chosen, y < x -> sum x+y
            # 3. {y, x} for y in chosen, y > x -> sum x+y (same sum as above, just different pair)
            # Wait, the count is for the SUM. 
            # If I add x, I add 1 to count[x+y] for EVERY y in chosen.
            # And 1 to count[2x].
            
            for y in chosen:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            s2 = 2 * best_x
            counts[s2] = counts.get(s2, 0) + 1
            
        return sorted(chosen)

    # Initial build
    best_set = build_structured_greedy()
    best_size = len(best_set)
    
    # 2. Local Search (Swap-based)
    # Try to swap an element in the set with one outside to increase size? 
    # No, swap keeps size same. 
    # To increase size, we need to "remove" an element that is "blocking" many others, 
    # and then "add" multiple elements that were previously blocked.
    # This is a "remove-and-rebuild" local search.
    
    def local_search_improve(initial_set):
        current_set = set(initial_set)
        current_counts = {}
        # Rebuild counts
        for x in current_set:
            for y in current_set:
                if x <= y:
                    s = x + y
                    current_counts[s] = current_counts.get(s, 0) + 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing each element and see if we can add more than 1 element back
            candidates_to_remove = list(current_set)
            random.shuffle(candidates_to_remove)
            
            for x_to_remove in candidates_to_remove:
                if time.time() > deadline - 5:
                    break
                
                # Temporarily remove
                current_set.remove(x_to_remove)
                
                # Rebuild counts for the reduced set? 
                # Or just adjust? Adjusting is faster.
                # Remove pairs involving x_to_remove
                for y in list(current_set):
                    s = x_to_remove + y
                    current_counts[s] -= 1
                    if current_counts[s] == 0:
                        del current_counts[s]
                s2 = 2 * x_to_remove
                current_counts[s2] -= 1
                if current_counts[s2] == 0:
                    del current_counts[s2]
                
                # Now try to greedily add as many as possible from the "excluded" set
                # Excluded set: {1..1000} - current_set
                excluded = [x for x in range(1, 1001) if x not in current_set]
                
                # Greedily add from excluded using the same "structured" heuristic
                # But simpler: just try to add valid ones.
                added = []
                # We need a fast validity check against current_counts
                for x in excluded:
                    if time.time() > deadline - 5:
                        break
                    valid = True
                    for y in current_set:
                        s = x + y
                        if current_counts.get(s, 0) >= 2:
                            valid = False
                            break
                    if valid:
                        s2 = 2 * x
                        if current_counts.get(s2, 0) >= 2:
                            valid = False
                    
                    if valid:
                        # Add it
                        current_set.add(x)
                        added.append(x)
                        for y in current_set:
                            if y == x: continue
                            s = x + y
                            current_counts[s] = current_counts.get(s, 0) + 1
                        s2 = 2 * x
                        current_counts[s2] = current_counts.get(s2, 0) + 1
                        
                # If we added more than 1 element, we improved size
                if len(added) > 1:
                    improved = True
                    # Note: x_to_remove is not re-added, so net gain is len(added) - 1
                    # We loop again to try more removals
                else:
                    # Revert
                    current_set.add(x_to_remove)
                    for y in current_set:
                        if y == x_to_remove: continue
                        s = x_to_remove + y
                        current_counts[s] = current_counts.get(s, 0) + 1
                    s2 = 2 * x_to_remove
                    current_counts[s2] = current_counts.get(s2, 0) + 1
                    # Actually, if we revert, we should undo the additions too.
                    # But if len(added) <= 1, we added 0 or 1.
                    # If 0: nothing to undo.
                    # If 1: we added 1, removed 1. Size same. 
                    # But the set composition changed. 
                    # To keep it simple, if no size improvement, revert everything.
                    for a in added:
                        current_set.remove(a)
                        for y in current_set:
                            if y == a: continue
                            s = a + y
                            current_counts[s] -= 1
                            if current_counts[s] == 0: del current_counts[s]
                        s2 = 2 * a
                        current_counts[s2] -= 1
                        if current_counts[s2] == 0: del current_counts[s2]
                        
                    # Re-add x_to_remove
                    current_set.add(x_to_remove)
                    for y in current_set:
                        if y == x_to_remove: continue
                        s = x_to_remove + y
                        current_counts[s] = current_counts.get(s, 0) + 1
                    s2 = 2 * x_to_remove
                    current_counts[s2] = current_counts.get(s2, 0) + 1

        return list(current_set)

    # Run local search
    if time.time() < deadline - 10:
        improved_set = local_search_improve(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(best_set)

    return best_set
