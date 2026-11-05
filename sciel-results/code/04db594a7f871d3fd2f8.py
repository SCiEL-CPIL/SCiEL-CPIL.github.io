import time
import random
import numpy as np

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
    initial_sums_np = np.array(get_sums(required), dtype=bool)
    
    # Candidates pool (excluding required)
    candidates = [x for x in range(1, max_val + 1) if x not in required_set]
    
    # Best found set
    best_set = required[:]
    best_size = len(best_set)
    
    # Strategy 1: Randomized Greedy
    def randomized_greedy(sort_mode=False):
        curr = required[:]
        curr_sums = initial_sums_np.copy()
        
        pool = candidates[:]
        if sort_mode:
            pool.sort()
        else:
            random.shuffle(pool)
            
        for x in pool:
            # Check if x can be added
            # x + y for y in curr
            # Also 2*x
            # Optimization: Use a list for small curr, numpy for large?
            # numpy is generally faster for the indexing check if we vectorize the sum creation
            # But creating the array [x+y for y in curr] is python level.
            # Let's try to keep it simple and fast.
            
            # Check sums
            valid = True
            # 2*x
            if curr_sums[2*x]:
                valid = False
            else:
                for y in curr:
                    if curr_sums[x+y]:
                        valid = False
                        break
            if valid:
                curr.append(x)
                # Update sums
                # We can update the numpy array
                for y in curr[:-1]:
                    curr_sums[x+y] = True
                curr_sums[2*x] = True
        
        return curr

    # Strategy 2: Local Search (Kick-out) with multi-kick
    def local_search_improve(base_set, max_kicks=2):
        if len(base_set) <= len(required):
            return base_set
        
        removable = [x for x in base_set if x not in required_set]
        if not removable:
            return base_set
            
        best_local = base_set[:]
        
        # Try removing 1 element
        random.shuffle(removable)
        for rem_val in removable:
            new_set = [x for x in base_set if x != rem_val]
            new_sums = np.array(get_sums(new_set), dtype=bool)
            new_set_set = set(new_set)
            pool = [x for x in candidates if x not in new_set_set]
            random.shuffle(pool)
            
            for x in pool:
                valid = True
                if new_sums[2*x]:
                    valid = False
                else:
                    for y in new_set:
                        if new_sums[x+y]:
                            valid = False
                            break
                if valid:
                    new_set.append(x)
                    for y in new_set[:-1]:
                        new_sums[x+y] = True
                    new_sums[2*x] = True
            
            if len(new_set) > len(best_local):
                best_local = new_set[:]

        # Try removing 2 elements (if time permits and set is large enough)
        if len(removable) >= 2 and time.time() < deadline - 5:
            # Sample some pairs to remove
            # Don't do all pairs, just random samples
            num_pair_tries = 20
            for _ in range(num_pair_tries):
                if time.time() > deadline - 2:
                    break
                idx1, idx2 = random.sample(range(len(removable)), 2)
                rem1 = removable[idx1]
                rem2 = removable[idx2]
                
                new_set = [x for x in base_set if x != rem1 and x != rem2]
                new_sums = np.array(get_sums(new_set), dtype=bool)
                new_set_set = set(new_set)
                pool = [x for x in candidates if x not in new_set_set]
                random.shuffle(pool)
                
                for x in pool:
                    valid = True
                    if new_sums[2*x]:
                        valid = False
                    else:
                        for y in new_set:
                            if new_sums[x+y]:
                                valid = False
                                break
                    if valid:
                        new_set.append(x)
                        for y in new_set[:-1]:
                            new_sums[x+y] = True
                        new_sums[2*x] = True
                
                if len(new_set) > len(best_local):
                    best_local = new_set[:]

        return best_local

    # Initial greedy run to establish a baseline
    # Try both sorted and shuffled
    curr1 = randomized_greedy(sort_mode=False)
    curr2 = randomized_greedy(sort_mode=True)
    
    curr = curr1 if len(curr1) > len(curr2) else curr2
    if len(curr) > best_size:
        best_size = len(curr)
        best_set = curr[:]

    # Main search loop
    while time.time() < deadline:
        # Alternate between randomized greedy and local search on best
        if random.random() < 0.4:
            # Mix of sorted and shuffled greedy
            if random.random() < 0.5:
                curr = randomized_greedy(sort_mode=False)
            else:
                curr = randomized_greedy(sort_mode=True)
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
            if random.random() < 0.3:
                fresh = randomized_greedy(sort_mode=random.random() < 0.5)
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
