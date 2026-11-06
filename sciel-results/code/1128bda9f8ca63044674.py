"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    best_set = []
    best_size = 0
    
    def greedy_forward(order, max_time=None):
        """Greedy: try elements in given order, add if valid."""
        if max_time is None:
            max_time = deadline
        chosen = []
        # counts[s] = number of representations of s as a+b, a<=b, a,b in chosen
        counts = {}
        
        for x in order:
            if time.time() > max_time:
                break
            # Check if adding x would violate the B2[2] constraint
            # New sums: x + y for y in chosen (x >= y since we process in order? No, order is arbitrary)
            # Actually, sums are x+y where x,y in chosen, x<=y.
            # When we add x, new pairs are (x, x) and (x, y) for y in chosen where y <= x.
            # Wait, the representation is a+b with a<=b. So if we have set S, and we add x:
            # New representations are:
            # 1. x + x (if x not in S, which it isn't since we're adding it)
            # 2. x + y for y in S where y <= x
            # 3. y + x for y in S where y > x? No, that's the same as x+y.
            # So new sums are x+x and x+y for y in S, y <= x.
            
            # But wait, the counts dictionary tracks all sums. When we add x, we need to check
            # that for each new sum s, the new count is <= 2.
            
            # New sums:
            # s1 = 2*x
            # s2 = x + y for y in chosen if y <= x
            
            # Actually, it's simpler: for each y in chosen, the sum x+y is new (if y <= x, it's a new representation; if y > x, the representation y+x was already counted when y was added? No.)
            
            # Let's think carefully. The set is unordered. A representation of s is a pair (a,b) with a<=b, a,b in S.
            # When we add x to S, the new representations are those involving x:
            # - (x, x) -> sum 2x
            # - (x, y) for y in S, y <= x -> sum x+y
            # - (y, x) for y in S, y > x -> this is the same as (x, y) with x < y, so sum x+y
            # So actually, for every y in S, the sum x+y gets a new representation. Plus 2x gets one.
            
            # Wait, no. If y > x, then the pair is (x, y) which is a new representation of x+y.
            # If y <= x, the pair is (y, x) which is a new representation of x+y.
            # So yes, for every y in S, x+y gets +1 count. And 2x gets +1 count.
            
            # So we need to check:
            # For each y in chosen: if counts.get(x+y, 0) + 1 > 2, then invalid.
            # And if counts.get(2*x, 0) + 1 > 2, then invalid.
            
            valid = True
            if counts.get(2*x, 0) + 1 > 2:
                valid = False
            else:
                for y in chosen:
                    if counts.get(x + y, 0) + 1 > 2:
                        valid = False
                        break
            
            if valid:
                # Update counts
                counts[2*x] = counts.get(2*x, 0) + 1
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        
        return chosen
    
    # Strategy 1: Simple forward greedy
    s1 = time.time()
    set1 = greedy_forward(range(1, 1001))
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1
    
    # Strategy 2: Backward greedy
    if time.time() < deadline:
        set2 = greedy_forward(range(1000, 0, -1))
        if len(set2) > best_size:
            best_size = len(set2)
            best_set = set2
    
    # Strategy 3: Randomized greedy with multiple restarts
    random.seed(42)
    while time.time() < deadline - 5:
        # Random permutation
        order = list(range(1, 1001))
        random.shuffle(order)
        set_r = greedy_forward(order)
        if len(set_r) > best_size:
            best_size = len(set_r)
            best_set = set_r
    
    # Strategy 4: Local search / improvement on best_set
    # Try to add elements that were skipped, or swap elements
    if time.time() < deadline - 10 and best_set:
        best_set = list(best_set)
        best_set.sort()
        # Try adding each missing element
        missing = [x for x in range(1, 1001) if x not in best_set]
        # Rebuild counts for best_set
        counts = {}
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                s = best_set[i] + best_set[j]
                counts[s] = counts.get(s, 0) + 1
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in missing:
                if time.time() > deadline - 5:
                    break
                # Check if x can be added
                valid = True
                if counts.get(2*x, 0) + 1 > 2:
                    valid = False
                else:
                    for y in best_set:
                        if counts.get(x + y, 0) + 1 > 2:
                            valid = False
                            break
                if valid:
                    # Add x
                    counts[2*x] = counts.get(2*x, 0) + 1
                    for y in best_set:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    best_set.append(x)
                    missing.remove(x)
                    best_set.sort()
                    improved = True
                    break  # Restart missing list since best_set changed
    
    # Final local search: try to swap elements to allow more additions
    if time.time() < deadline - 5 and best_set:
        best_set = list(best_set)
        best_set.sort()
        # Try removing one element and see if we can add two or more
        # This is expensive, so only do a few iterations
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            current_size = len(best_set)
            # Rebuild counts
            counts = {}
            for i in range(len(best_set)):
                for j in range(i, len(best_set)):
                    s = best_set[i] + best_set[j]
                    counts[s] = counts.get(s, 0) + 1
            
            best_swap = None
            for idx, x in enumerate(best_set):
                if time.time() > deadline - 5:
                    break
                # Try removing x
                trial_set = best_set[:idx] + best_set[idx+1:]
                trial_counts = {}
                for i in range(len(trial_set)):
                    for j in range(i, len(trial_set)):
                        s = trial_set[i] + trial_set[j]
                        trial_counts[s] = trial_counts.get(s, 0) + 1
                
                # Try adding missing elements
                added = 0
                new_set = list(trial_set)
                for y in range(1, 1001):
                    if y in new_set:
                        continue
                    valid = True
                    if trial_counts.get(2*y, 0) + 1 > 2:
                        valid = False
                    else:
                        for z in new_set:
                            if trial_counts.get(y + z, 0) + 1 > 2:
                                valid = False
                                break
                    if valid:
                        trial_counts[2*y] = trial_counts.get(2*y, 0) + 1
                        for z in new_set:
                            s = y + z
                            trial_counts[s] = trial_counts.get(s, 0) + 1
                        new_set.append(y)
                        added += 1
                        if added >= 2:
                            break
                
                if added >= 2:
                    best_swap = (x, new_set)
                    break
            
            if best_swap:
                best_set = best_swap[1]
                best_set.sort()
    
    if time.time() > deadline:
        return best_set
    
    return best_set
# EVOLVE-BLOCK-END
