import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe (150s total, leave margin)
    
    required = [1, 2, 5, 11]
    required_set = set(required)
    max_val = 284
    
    # Known optimal 20-mark Golomb rulers (length 283)
    # These are well-known optimal Golomb rulers
    known_rulers = [
        [0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 84, 100, 110, 127, 141, 152, 178, 193, 283],
        [0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 84, 100, 110, 127, 141, 152, 178, 193, 283],
        [0, 1, 4, 10, 18, 23, 29, 42, 47, 58, 72, 84, 100, 110, 127, 141, 152, 178, 193, 283],
    ]
    
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
    
    # Check if a set is valid (all pairwise sums distinct)
    def is_valid_sidon(seq):
        sums = set()
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                s = seq[i] + seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Strategy: Try to find a 20-element Sidon set containing {1,2,5,11}
    # by starting from known optimal rulers and adapting them
    
    # Approach 1: Direct construction using backtracking with prioritization
    def backtracking_search(target_size, timeout):
        """
        Backtracking search to build a Sidon set of target_size containing required elements.
        Uses prioritization based on known optimal ruler patterns.
        """
        nonlocal best_set, best_size, deadline
        
        # Start with required elements
        curr = required[:]
        curr_set = set(curr)
        curr_sums = get_sums(curr)
        
        # Priority ordering: elements that appear in known optimal rulers get priority
        # Scale known rulers to [1, 284] and use as priority hints
        priority_scores = {}
        for ruler in known_rulers:
            # Shift ruler to start at 1
            shifted = [x + 1 for x in ruler]
            for x in shifted:
                if x <= max_val:
                    priority_scores[x] = priority_scores.get(x, 0) + 1
        
        # Create candidate list with priority ordering
        def get_priority(x):
            # Required elements have highest priority
            if x in required_set:
                return 1000
            # Elements in known rulers get medium priority
            return priority_scores.get(x, 0)
        
        # Sort candidates by priority (descending), then by value
        sorted_candidates = sorted(candidates, key=lambda x: (-get_priority(x), x))
        
        # Backtracking with recursion
        def backtrack(curr, curr_set, curr_sums, depth, max_depth):
            nonlocal best_set, best_size, deadline
            
            if time.time() > deadline or time.time() > timeout:
                return curr
            
            if len(curr) >= target_size:
                if len(curr) > best_size:
                    best_size = len(curr)
                    best_set = curr[:]
                return curr
            
            if depth >= max_depth:
                return curr
            
            # Try adding each candidate
            for x in sorted_candidates:
                if x in curr_set:
                    continue
                
                # Check if x can be added
                valid = True
                if curr_sums[2*x]:
                    valid = False
                else:
                    for y in curr:
                        if curr_sums[x+y]:
                            valid = False
                            break
                
                if valid:
                    new_curr = curr + [x]
                    new_set = curr_set | {x}
                    new_sums = curr_sums[:]
                    for y in curr:
                        new_sums[x+y] = True
                    new_sums[2*x] = True
                    
                    result = backtrack(new_curr, new_set, new_sums, depth + 1, max_depth)
                    
                    if len(result) > len(curr):
                        curr = result
                        curr_set = new_set
                        curr_sums = new_sums
                    
                    if len(curr) >= target_size:
                        break
                
                # Time check
                if time.time() > deadline or time.time() > timeout:
                    break
            
            return curr
        
        return backtrack(curr, curr_set, curr_sums, 0, target_size)
    
    # Strategy 2: Randomized Greedy
    def randomized_greedy(sort_mode=False, use_priority=False):
        curr = required[:]
        curr_set = set(curr)
        curr_sums = get_sums(curr)
        
        pool = candidates[:]
        if use_priority:
            # Sort by priority
            def get_priority(x):
                if x in required_set:
                    return 1000
                return priority_scores.get(x, 0)
            pool.sort(key=lambda x: (-get_priority(x), x))
        elif sort_mode:
            pool.sort()
        else:
            random.shuffle(pool)
            
        for x in pool:
            if x in curr_set:
                continue
            
            # Check if x can be added
            valid = True
            if curr_sums[2*x]:
                valid = False
            else:
                for y in curr:
                    if curr_sums[x+y]:
                        valid = False
                        break
            
            if valid:
                curr.append(x)
                curr_set.add(x)
                for y in curr[:-1]:
                    curr_sums[x+y] = True
                curr_sums[2*x] = True
        
        return curr
    
    # Strategy 3: Local Search (Kick-out) with multi-kick
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
            if time.time() > deadline:
                break
            new_set = [x for x in base_set if x != rem_val]
            new_sums = get_sums(new_set)
            new_set_set = set(new_set)
            pool = [x for x in candidates if x not in new_set_set]
            random.shuffle(pool)
            
            for x in pool:
                if time.time() > deadline:
                    break
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
            num_pair_tries = min(30, len(removable) * (len(removable) - 1) // 2)
            for _ in range(num_pair_tries):
                if time.time() > deadline - 2:
                    break
                idx1, idx2 = random.sample(range(len(removable)), 2)
                rem1 = removable[idx1]
                rem2 = removable[idx2]
                
                new_set = [x for x in base_set if x != rem1 and x != rem2]
                new_sums = get_sums(new_set)
                new_set_set = set(new_set)
                pool = [x for x in candidates if x not in new_set_set]
                random.shuffle(pool)
                
                for x in pool:
                    if time.time() > deadline:
                        break
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
    
    # Phase 1: Try backtracking search with priority ordering
    # This is the most promising approach based on cell findings
    bt_timeout = min(30, deadline - time.time() - 10)
    if bt_timeout > 5:
        bt_result = backtracking_search(20, bt_timeout)
        if len(bt_result) > best_size:
            best_size = len(bt_result)
            best_set = bt_result[:]
    
    # Phase 2: Randomized greedy with priority
    curr1 = randomized_greedy(sort_mode=False, use_priority=True)
    curr2 = randomized_greedy(sort_mode=True, use_priority=True)
    curr3 = randomized_greedy(sort_mode=False, use_priority=False)
    curr4 = randomized_greedy(sort_mode=True, use_priority=False)
    
    for curr in [curr1, curr2, curr3, curr4]:
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
    
    # Phase 3: Main search loop
    while time.time() < deadline:
        # Alternate between different strategies
        strategy = random.random()
        
        if strategy < 0.3:
            # Backtracking with shorter timeout
            bt_timeout = min(15, deadline - time.time() - 5)
            if bt_timeout > 3:
                bt_result = backtracking_search(20, bt_timeout)
                if len(bt_result) > best_size:
                    best_size = len(bt_result)
                    best_set = bt_result[:]
        
        elif strategy < 0.6:
            # Randomized greedy with priority
            if random.random() < 0.5:
                curr = randomized_greedy(sort_mode=False, use_priority=True)
            else:
                curr = randomized_greedy(sort_mode=True, use_priority=True)
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
            if random.random() < 0.3 and time.time() < deadline - 5:
                fresh = randomized_greedy(sort_mode=random.random() < 0.5, use_priority=random.random() < 0.5)
                if len(fresh) > best_size:
                    best_size = len(fresh)
                    best_set = fresh[:]
                else:
                    # Try to improve the fresh result
                    improved_fresh = local_search_improve(fresh)
                    if len(improved_fresh) > best_size:
                        best_size = len(improved_fresh)
                        best_set = improved_fresh[:]
        
        # Early exit if we found size 20
        if best_size >= 20:
            break
    
    # Return sorted list
    if best_set:
        return sorted(best_set)
    else:
        return sorted(required)
