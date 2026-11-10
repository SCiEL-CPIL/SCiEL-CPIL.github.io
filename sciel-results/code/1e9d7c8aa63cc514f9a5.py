import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_build(order):
        """Build a B2[2] set by greedily adding elements in the given order."""
        current_set = []
        counts = {}
        
        for x in order:
            valid = True
            
            for y in current_set:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            
            if valid:
                for y in current_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                
                s2 = 2 * x
                counts[s2] = counts.get(s2, 0) + 1
                
                current_set.append(x)
        
        return current_set
    
    def local_search(initial_set):
        """Improve the set by trying to add elements and swap out elements."""
        current_set = set(initial_set)
        counts = {}
        
        # Rebuild counts from the initial set
        for i, a in enumerate(sorted(current_set)):
            for b in list(current_set)[i:]:
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding elements not in the set
            for x in range(1, 1001):
                if x in current_set:
                    continue
                
                valid = True
                for y in current_set:
                    s = x + y
                    if counts.get(s, 0) >= 2:
                        valid = False
                        break
                
                if valid:
                    s2 = 2 * x
                    if counts.get(s2, 0) >= 2:
                        valid = False
                
                if valid:
                    current_set.add(x)
                    for y in current_set:
                        if y != x:
                            s = x + y
                            counts[s] = counts.get(s, 0) + 1
                    s2 = 2 * x
                    counts[s2] = counts.get(s2, 0) + 1
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one element, add another
                sorted_set = sorted(current_set)
                for i, x in enumerate(sorted_set):
                    if time.time() >= deadline:
                        break
                    
                    # Try removing x
                    # Check if removing x frees up space for any new element
                    trial_set = current_set - {x}
                    trial_counts = {}
                    for a in sorted(trial_set):
                        for b in sorted(trial_set):
                            if a <= b:
                                s = a + b
                                trial_counts[s] = trial_counts.get(s, 0) + 1
                    
                    # Find a candidate to add
                    for y in range(1, 1001):
                        if y in trial_set:
                            continue
                        
                        valid = True
                        for z in trial_set:
                            s = y + z
                            if trial_counts.get(s, 0) >= 2:
                                valid = False
                                break
                        
                        if valid:
                            s2 = 2 * y
                            if trial_counts.get(s2, 0) >= 2:
                                valid = False
                        
                        if valid:
                            # Perform the swap
                            current_set = trial_set
                            counts = trial_counts
                            current_set.add(y)
                            for z in current_set:
                                if z != y:
                                    s = y + z
                                    counts[s] = counts.get(s, 0) + 1
                            s2 = 2 * y
                            counts[s2] = counts.get(s2, 0) + 1
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return list(current_set)
    
    # Try different greedy orderings
    orderings = []
    
    # Random orderings
    for _ in range(30):
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        orderings.append(candidates)
    
    # Ascending
    orderings.append(list(range(1, 1001)))
    
    # Descending
    orderings.append(list(range(1000, 0, -1)))
    
    # Weighted: prefer smaller numbers
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: x + random.random() * 100)
    orderings.append(candidates)
    
    # Sparse: prefer numbers with large gaps
    candidates = list(range(1, 1001))
    random.shuffle(candidates)
    candidates.sort(key=lambda x: random.random())
    orderings.append(candidates)
    
    for order in orderings:
        if time.time() >= deadline:
            break
        
        result = greedy_build(order)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Apply local search to the best set found
    if best_set and time.time() < deadline - 10:
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
