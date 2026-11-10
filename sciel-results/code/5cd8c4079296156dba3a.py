import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds safety margin
    
    best_set = []
    best_size = 0
    
    def greedy_build(candidates, counts_init=None, current_set_init=None):
        """Build a set greedily from candidates in given order."""
        if counts_init is None:
            counts = {}
        else:
            counts = dict(counts_init)
        if current_set_init is None:
            current_set = []
        else:
            current_set = list(current_set_init)
        
        for x in candidates:
            valid = True
            trial_counts = dict(counts)
            
            for y in current_set:
                s = x + y
                trial_counts[s] = trial_counts.get(s, 0) + 1
                if trial_counts[s] > 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                trial_counts[s2] = trial_counts.get(s2, 0) + 1
                if trial_counts[s2] > 2:
                    valid = False
            
            if valid:
                current_set.append(x)
                counts = trial_counts
        
        return current_set, counts
    
    def local_search(set_list, counts_dict, max_iters=50):
        """Try to improve the set by swapping elements."""
        if not set_list:
            return set_list, counts_dict
        
        current_set = list(set_list)
        counts = dict(counts_dict)
        current_size = len(current_set)
        
        # Build a set for fast lookup
        in_set = set(current_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            # Try swapping each element with a candidate not in set
            for i in range(len(current_set)):
                if time.time() > deadline - 2.0:
                    break
                    
                elem = current_set[i]
                # Try to find a replacement
                # Candidates: numbers not in current_set
                # We'll try a few random candidates
                for _attempt in range(3):
                    candidate = random.randint(1, 1000)
                    if candidate in in_set:
                        continue
                    
                    # Check if we can swap elem -> candidate
                    # Remove elem, add candidate
                    # This is complex; let's do a simpler approach
                    # Just check if candidate can be added (ignoring the removal benefit)
                    # Actually, let's try a different approach:
                    # Remove elem, then try to add candidate
                    
                    # Compute new counts after removing elem
                    new_counts = dict(counts)
                    new_set = list(current_set)
                    
                    # Remove elem from set and update counts
                    new_set.pop(i)
                    for y in new_set:
                        s = elem + y
                        if s in new_counts:
                            new_counts[s] -= 1
                            if new_counts[s] <= 0:
                                del new_counts[s]
                    s2 = 2 * elem
                    if s2 in new_counts:
                        new_counts[s2] -= 1
                        if new_counts[s2] <= 0:
                            del new_counts[s2]
                    
                    # Now try to add candidate
                    valid = True
                    trial_counts = dict(new_counts)
                    for y in new_set:
                        s = candidate + y
                        trial_counts[s] = trial_counts.get(s, 0) + 1
                        if trial_counts[s] > 2:
                            valid = False
                            break
                    
                    if valid:
                        s2c = 2 * candidate
                        trial_counts[s2c] = trial_counts.get(s2c, 0) + 1
                        if trial_counts[s2c] > 2:
                            valid = False
                    
                    if valid:
                        # Accept the swap
                        current_set = new_set + [candidate]
                        counts = trial_counts
                        in_set = set(current_set)
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
        
        return current_set, counts
    
    # Phase 1: Initial greedy pass (deterministic, increasing order)
    current_set, counts = greedy_build(range(1, 1001))
    best_set = current_set
    best_size = len(current_set)
    
    # Phase 2: Randomized greedy + local search
    num_iterations = 0
    no_improve_count = 0
    
    while time.time() < deadline - 5.0:
        num_iterations += 1
        random.seed(num_iterations * 7919)  # Different seed multiplier
        
        # Shuffle the range
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set, counts = greedy_build(candidates)
        current_size = len(current_set)
        
        # Local search to improve
        if current_size >= best_size:
            current_set, counts = local_search(current_set, counts, max_iters=20)
            current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If we've had many iterations without improvement, try different strategies
        if no_improve_count > 50:
            no_improve_count = 0
    
    return best_set
