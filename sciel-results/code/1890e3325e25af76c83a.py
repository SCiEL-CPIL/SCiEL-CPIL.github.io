import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (150s total, leave margin)
    
    required = [1, 2, 5, 11]
    required_set = set(required)
    max_val = 284
    
    # Helper to get initial sums for a set as a boolean list for fast lookup
    def get_sums(seq):
        sums = [False] * (2 * max_val + 2)
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                sums[s] = True
        return sums

    # Precompute initial sums for required elements
    initial_sums = get_sums(required)
    
    # Candidates pool (excluding required)
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Best found set
    best_set = required[:]
    best_size = len(best_set)
    
    # Strategy 1: Randomized Greedy
    # We shuffle the candidates and greedily add if valid.
    def randomized_greedy():
        curr = required[:]
        curr_sums = initial_sums[:]  # Copy the list
        
        pool = candidates[:]
        random.shuffle(pool)
        
        for x in pool:
            # Check if x can be added
            valid = True
            # Check x + y for all y in curr
            for y in curr:
                s = x + y
                if curr_sums[s]:
                    valid = False
                    break
            # Check 2*x
            if valid and curr_sums[2 * x]:
                valid = False
            
            if valid:
                curr.append(x)
                # Update sums
                for y in curr[:-1]:
                    curr_sums[x + y] = True
                curr_sums[2 * x] = True
        
        return curr

    # Strategy 2: Local Search (Kick-out)
    # Try to improve the best set by removing one non-essential element and re-filling.
    def local_search_improve(base_set):
        # base_set should be a list
        if len(base_set) <= len(required):
            return base_set
        
        # Create a list of removable elements (non-required)
        removable = [x for x in base_set if x not in required_set]
        if not removable:
            return base_set
            
        # Shuffle removable to try different kicks
        random.shuffle(removable)
        
        best_local = base_set[:]
        
        for rem_val in removable:
            # Try removing rem_val
            new_set = [x for x in base_set if x != rem_val]
            new_sums = get_sums(new_set)
            
            # Now run a fast greedy fill on the remaining candidates
            # Candidates are those not in new_set
            # We can reuse the global candidates list, filtering out those in new_set
            new_set_set = set(new_set)
            pool = [x for x in candidates if x not in new_set_set]
            random.shuffle(pool)
            
            for x in pool:
                valid = True
                for y in new_set:
                    s = x + y
                    if new_sums[s]:
                        valid = False
                        break
                if valid and new_sums[2 * x]:
                    valid = False
                
                if valid:
                    new_set.append(x)
                    for y in new_set[:-1]:
                        new_sums[x + y] = True
                    new_sums[2 * x] = True
            
            if len(new_set) > len(best_local):
                best_local = new_set[:]
                
        return best_local

    # Initial greedy run to establish a baseline
    curr = randomized_greedy()
    if len(curr) > best_size:
        best_size = len(curr)
        best_set = curr[:]

    # Main search loop
    while time.time() < deadline:
        # Alternate between randomized greedy and local search on best
        if random.random() < 0.5:
            curr = randomized_greedy()
            if len(curr) > best_size:
                best_size = len(curr)
                best_set = curr[:]
        else:
            # Local search on the current best
            improved = local_search_improve(best_set)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
            
            # Also try local search on a fresh randomized greedy result sometimes
            if random.random() < 0.2:
                fresh = randomized_greedy()
                if len(fresh) > best_size:
                    best_size = len(fresh)
                    best_set = fresh[:]
                else:
                    # Try to improve the fresh result
                    improved_fresh = local_search_improve(fresh)
                    if len(improved_fresh) > best_size:
                        best_size = len(improved_fresh)
                        best_set = improved_fresh[:]

    # Return sorted list
    if best_set:
        return sorted(best_set)
    else:
        return sorted(required)
