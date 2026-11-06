"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_counts(chosen):
    """Compute sum counts for a set."""
    counts = {}
    n = len(chosen)
    for i in range(n):
        ai = chosen[i]
        for j in range(i, n):
            s = ai + chosen[j]
            counts[s] = counts.get(s, 0) + 1
    return counts

def _try_add(chosen, x, counts):
    """Try to add x to chosen. Return True if valid, and update counts in place if so."""
    n = len(chosen)
    new_sums = []
    for i in range(n):
        s = x + chosen[i]
        new_sums.append(s)
    new_sums.append(2 * x)
    
    # Check if any sum would exceed 2
    for s in new_sums:
        if counts.get(s, 0) + 1 > 2:
            return False
    
    # All good, update counts
    for s in new_sums:
        counts[s] = counts.get(s, 0) + 1
    chosen.append(x)
    return True

def _remove(chosen, idx, counts):
    """Remove element at idx from chosen and update counts."""
    x = chosen.pop(idx)
    n = len(chosen)
    for i in range(n):
        s = x + chosen[i]
        counts[s] -= 1
    counts[2 * x] -= 1

def _greedy_build(start=1, end=1001):
    """Standard greedy: try each integer in order, add if valid."""
    chosen = []
    counts = {}
    for x in range(start, end):
        n = len(chosen)
        valid = True
        for i in range(n):
            s = x + chosen[i]
            if counts.get(s, 0) + 1 > 2:
                valid = False
                break
        if valid:
            s = 2 * x
            if counts.get(s, 0) + 1 > 2:
                valid = False
        if valid:
            for i in range(n):
                s = x + chosen[i]
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen.append(x)
    return chosen

def _greedy_with_backtrack(start=1, end=1001, max_backtrack_depth=5, time_limit=None):
    """Greedy with limited backtracking to escape local optima."""
    best = _greedy_build(start, end)
    best_len = len(best)
    
    t0 = time.time()
    if time_limit is None:
        time_limit = 10
    
    # Try different starting points and small perturbations
    for attempt in range(50):
        if time.time() - t0 > time_limit:
            break
        
        # Randomly skip some early elements to create diversity
        skip_set = set()
        for x in range(start, min(start + 20, end)):
            if random.random() < 0.1:
                skip_set.add(x)
        
        chosen = []
        counts = {}
        for x in range(start, end):
            if x in skip_set:
                continue
            n = len(chosen)
            valid = True
            for i in range(n):
                s = x + chosen[i]
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            if valid:
                for i in range(n):
                    s = x + chosen[i]
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen.append(x)
        
        if len(chosen) > best_len:
            best = chosen
            best_len = len(chosen)
    
    return best

def _local_search(chosen, max_iterations=1000, time_limit=None):
    """Local search: try swapping elements to improve size."""
    t0 = time.time()
    if time_limit is None:
        time_limit = 20
    
    chosen = list(chosen)
    counts = _build_counts(chosen)
    best = list(chosen)
    best_len = len(chosen)
    
    for iteration in range(max_iterations):
        if time.time() - t0 > time_limit:
            break
        
        # Try to add a random element not in the set
        if random.random() < 0.7:
            # Try adding a new element
            candidates = [x for x in range(1, 1001) if x not in set(chosen)]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            n = len(chosen)
            valid = True
            for i in range(n):
                s = x + chosen[i]
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                for i in range(n):
                    s = x + chosen[i]
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen.append(x)
                
                if len(chosen) > best_len:
                    best = list(chosen)
                    best_len = len(chosen)
            else:
                # Try removing one element and adding two
                if len(chosen) > 10:
                    idx = random.randint(0, len(chosen) - 1)
                    x_remove = chosen[idx]
                    
                    # Remove x_remove
                    n = len(chosen)
                    for i in range(n):
                        if i == idx:
                            continue
                        s = x_remove + chosen[i]
                        counts[s] -= 1
                    counts[2 * x_remove] -= 1
                    chosen.pop(idx)
                    
                    # Try adding two new elements
                    added_any = False
                    for _ in range(2):
                        candidates = [x for x in range(1, 1001) if x not in set(chosen)]
                        if not candidates:
                            break
                        x = random.choice(candidates)
                        
                        n2 = len(chosen)
                        valid2 = True
                        for i in range(n2):
                            s = x + chosen[i]
                            if counts.get(s, 0) + 1 > 2:
                                valid2 = False
                                break
                        if valid2:
                            s = 2 * x
                            if counts.get(s, 0) + 1 > 2:
                                valid2 = False
                        
                        if valid2:
                            for i in range(n2):
                                s = x + chosen[i]
                                counts[s] = counts.get(s, 0) + 1
                            counts[2 * x] = counts.get(2 * x, 0) + 1
                            chosen.append(x)
                            added_any = True
                        else:
                            # Revert
                            # This is getting complex; let's just revert the whole thing
                            break
                    
                    if not added_any or len(chosen) <= best_len:
                        # Revert: re-add x_remove
                        n3 = len(chosen)
                        for i in range(n3):
                            s = x_remove + chosen[i]
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x_remove] = counts.get(2 * x_remove, 0) + 1
                        chosen.insert(idx, x_remove)
                    else:
                        if len(chosen) > best_len:
                            best = list(chosen)
                            best_len = len(chosen)
        else:
            # Try swapping
            if len(chosen) >= 2:
                idx1 = random.randint(0, len(chosen) - 1)
                x1 = chosen[idx1]
                
                # Remove x1
                n = len(chosen)
                for i in range(n):
                    if i == idx1:
                        continue
                    s = x1 + chosen[i]
                    counts[s] -= 1
                counts[2 * x1] -= 1
                chosen.pop(idx1)
                
                # Try adding a different element
                candidates = [x for x in range(1, 1001) if x not in set(chosen)]
                if candidates:
                    x2 = random.choice(candidates)
                    n2 = len(chosen)
                    valid = True
                    for i in range(n2):
                        s = x2 + chosen[i]
                        if counts.get(s, 0) + 1 > 2:
                            valid = False
                            break
                    if valid:
                        s = 2 * x2
                        if counts.get(s, 0) + 1 > 2:
                            valid = False
                    
                    if valid:
                        for i in range(n2):
                            s = x2 + chosen[i]
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x2] = counts.get(2 * x2, 0) + 1
                        chosen.append(x2)
                    else:
                        # Revert
                        for i in range(n2):
                            s = x1 + chosen[i]
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x1] = counts.get(2 * x1, 0) + 1
                        chosen.insert(idx1, x1)
    
    return best

def _simulated_annealing(chosen, time_limit=None):
    """Simulated annealing on the set."""
    t0 = time.time()
    if time_limit is None:
        time_limit = 30
    
    chosen = list(chosen)
    counts = _build_counts(chosen)
    best = list(chosen)
    best_len = len(chosen)
    
    T = 10.0
    T_min = 0.1
    cooling = 0.99
    
    while T > T_min:
        if time.time() - t0 > time_limit:
            break
        
        # Random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Add
            candidates = [x for x in range(1, 1001) if x not in set(chosen)]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            n = len(chosen)
            valid = True
            for i in range(n):
                s = x + chosen[i]
                if counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            if valid:
                s = 2 * x
                if counts.get(s, 0) + 1 > 2:
                    valid = False
            
            if valid:
                for i in range(n):
                    s = x + chosen[i]
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen.append(x)
                
                if len(chosen) > best_len:
                    best = list(chosen)
                    best_len = len(chosen)
            else:
                # Accept with probability exp(-1/T)
                if random.random() < np.exp(-1.0 / T):
                    # Force add by removing a conflicting element
                    # Find a sum that would exceed
                    conflict = None
                    for i in range(n):
                        s = x + chosen[i]
                        if counts.get(s, 0) + 1 > 2:
                            conflict = i
                            break
                    if conflict is None:
                        if counts.get(2 * x, 0) + 1 > 2:
                            # Find an element y where 2y = 2x (i.e., y=x, but x not in set)
                            # So conflict is with some other pair
                            for i in range(n):
                                for j in range(i + 1, n):
                                    if chosen[i] + chosen[j] == 2 * x:
                                        conflict = i
                                        break
                                if conflict is not None:
                                    break
                    
                    if conflict is not None:
                        x_remove = chosen[conflict]
                        n2 = len(chosen)
                        for i in range(n2):
                            if i == conflict:
                                continue
                            s = x_remove + chosen[i]
                            counts[s] -= 1
                        counts[2 * x_remove] -= 1
                        chosen.pop(conflict)
                        
                        # Now try adding x again
                        n3 = len(chosen)
                        valid3 = True
                        for i in range(n3):
                            s = x + chosen[i]
                            if counts.get(s, 0) + 1 > 2:
                                valid3 = False
                                break
                        if valid3:
                            s = 2 * x
                            if counts.get(s, 0) + 1 > 2:
                                valid3 = False
                        
                        if valid3:
                            for i in range(n3):
                                s = x + chosen[i]
                                counts[s] = counts.get(s, 0) + 1
                            counts[2 * x] = counts.get(2 * x, 0) + 1
                            chosen.append(x)
                            
                            if len(chosen) > best_len:
                                best = list(chosen)
                                best_len = len(chosen)
                        else:
                            # Revert
                            for i in range(n3):
                                s = x_remove + chosen[i]
                                counts[s] = counts.get(s, 0) + 1
                            counts[2 * x_remove] = counts.get(2 * x_remove, 0) + 1
                            chosen.insert(conflict, x_remove)
                # If not accepted, do nothing (x was not added)
        
        elif move_type < 0.7:
            # Remove
            if len(chosen) > 5:
                idx = random.randint(0, len(chosen) - 1)
                x = chosen[idx]
                n = len(chosen)
                for i in range(n):
                    if i == idx:
                        continue
                    s = x + chosen[i]
                    counts[s] -= 1
                counts[2 * x] -= 1
                chosen.pop(idx)
        
        else:
            # Swap
            if len(chosen) >= 2:
                idx = random.randint(0, len(chosen) - 1)
                x1 = chosen[idx]
                n = len(chosen)
                for i in range(n):
                    if i == idx:
                        continue
                    s = x1 + chosen[i]
                    counts[s] -= 1
                counts[2 * x1] -= 1
                chosen.pop(idx)
                
                candidates = [x for x in range(1, 1001) if x not in set(chosen)]
                if candidates:
                    x2 = random.choice(candidates)
                    n2 = len(chosen)
                    valid = True
                    for i in range(n2):
                        s = x2 + chosen[i]
                        if counts.get(s, 0) + 1 > 2:
                            valid = False
                            break
                    if valid:
                        s = 2 * x2
                        if counts.get(s, 0) + 1 > 2:
                            valid = False
                    
                    if valid:
                        for i in range(n2):
                            s = x2 + chosen[i]
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x2] = counts.get(2 * x2, 0) + 1
                        chosen.append(x2)
                        
                        if len(chosen) > best_len:
                            best = list(chosen)
                            best_len = len(chosen)
                    else:
                        # Revert
                        for i in range(n2):
                            s = x1 + chosen[i]
                            counts[s] = counts.get(s, 0) + 1
                        counts[2 * x1] = counts.get(2 * x1, 0) + 1
                        chosen.insert(idx, x1)
        
        T *= cooling
    
    return best

def construct():
    t0 = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Phase 1: Basic greedy
    result = _greedy_build(1, 1001)
    best = list(result)
    best_len = len(result)
    
    if time.time() - t0 > deadline:
        return best
    
    # Phase 2: Greedy with random perturbations
    result2 = _greedy_with_backtrack(1, 1001, max_backtrack_depth=5, time_limit=20)
    if len(result2) > best_len:
        best = result2
        best_len = len(result2)
    
    if time.time() - t0 > deadline:
        return best
    
    # Phase 3: Local search
    result3 = _local_search(best, max_iterations=2000, time_limit=30)
    if len(result3) > best_len:
        best = result3
        best_len = len(result3)
    
    if time.time() - t0 > deadline:
        return best
    
    # Phase 4: Simulated annealing
    remaining = deadline - (time.time() - t0)
    if remaining > 5:
        result4 = _simulated_annealing(best, time_limit=remaining - 2)
        if len(result4) > best_len:
            best = result4
            best_len = len(result4)
    
    return best

# EVOLVE-BLOCK-END
