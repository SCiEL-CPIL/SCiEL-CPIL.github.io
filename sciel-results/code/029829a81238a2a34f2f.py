import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85

    def is_valid_addition(values, x):
        """Check if adding x to values keeps all consecutive block sums distinct."""
        if values and x <= values[-1]:
            return False
        # Compute new sums formed by x
        s = x
        new_sums = [s]
        for a in reversed(values):
            s += a
            new_sums.append(s)
        return new_sums

    def greedy(order, values=None, used=None):
        """Greedy: try values in given order, add if valid."""
        if values is None:
            values = []
        if used is None:
            used = set()
        sums = set()
        # Recompute sums from current values
        for i in range(len(values)):
            s = 0
            for j in range(i, len(values)):
                s += values[j]
                sums.add(s)
        
        for x in order:
            if x in used:
                continue
            if values and x <= values[-1]:
                continue
            # Check if x can be added
            new_sums = is_valid_addition(values, x)
            if new_sums is None:
                continue
            if any(ns in sums for ns in new_sums):
                continue
            values.append(x)
            used.add(x)
            sums.update(new_sums)
        return values, used

    def local_search(values, used, time_limit):
        """Try to improve the sequence by swapping or inserting elements."""
        best = list(values)
        best_used = set(used)
        best_len = len(best)
        end_time = time.time() + time_limit
        
        # Try to insert missing elements at various positions
        missing = [x for x in range(1, 257) if x not in best_used]
        random.shuffle(missing)
        
        while time.time() < end_time:
            improved = False
            for x in missing:
                if time.time() > end_time:
                    break
                # Try inserting x at various positions
                for pos in range(len(best) + 1):
                    if pos > 0 and x <= best[pos-1]:
                        continue
                    if pos < len(best) and x >= best[pos]:
                        continue
                    
                    # Check validity
                    new_values = best[:pos] + [x] + best[pos:]
                    valid = True
                    sums = set()
                    for i in range(len(new_values)):
                        s = 0
                        for j in range(i, len(new_values)):
                            s += new_values[j]
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            break
                    
                    if valid and len(new_values) > best_len:
                        best = new_values
                        best_used = set(best)
                        best_len = len(best)
                        improved = True
                        missing = [m for m in missing if m not in best_used]
                        break
            
            if not improved:
                # Try removing an element and adding a different one
                for i in range(len(best)):
                    if time.time() > end_time:
                        break
                    old_val = best[i]
                    candidate_best = best[:i] + best[i+1:]
                    # Now try to add a better element
                    for x in missing:
                        if x in best_used:
                            continue
                        # Try adding x to candidate_best
                        valid = True
                        sums = set()
                        for pos in range(len(candidate_best) + 1):
                            if pos > 0 and x <= candidate_best[pos-1]:
                                continue
                            if pos < len(candidate_best) and x >= candidate_best[pos]:
                                continue
                            new_values = candidate_best[:pos] + [x] + candidate_best[pos:]
                            valid = True
                            sums = set()
                            for ii in range(len(new_values)):
                                s = 0
                                for jj in range(ii, len(new_values)):
                                    s += new_values[jj]
                                    if s in sums:
                                        valid = False
                                        break
                                    sums.add(s)
                                if not valid:
                                    break
                            if valid and len(new_values) >= best_len:
                                if len(new_values) > best_len:
                                    best = new_values
                                    best_used = set(best)
                                    best_len = len(best)
                                    missing = [m for m in missing if m not in best_used]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best

    best = []
    best_len = 0
    
    # Strategy 1: Greedy with different orderings
    orderings = [
        list(range(1, 257)),           # Increasing
        list(range(256, 0, -1)),       # Decreasing
    ]
    
    # Add some randomized orderings
    for _ in range(5):
        rand_order = list(range(1, 257))
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for order in orderings:
        if time.time() > DEADLINE:
            break
        seq, used = greedy(order)
        if len(seq) > best_len:
            best = seq
            best_len = len(seq)
    
    # Strategy 2: Local search on the best sequence
    if best and time.time() < DEADLINE - 10:
        remaining_time = min(30, DEADLINE - time.time() - 5)
        improved_best = local_search(best, set(best), remaining_time)
        if len(improved_best) > best_len:
            best = improved_best
            best_len = len(best)
    
    return best
