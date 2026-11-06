import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    base = [1, 2, 5, 11]
    base_set = set(base)
    max_val = 284
    target_size = 20  # Known maximum
    
    all_candidates = list(range(1, max_val + 1))
    
    def get_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            a = s[i]
            for j in range(i, n):
                sums.add(a + s[j])
        return sums

    def can_add(s, sums, x):
        if 2 * x in sums:
            return False
        for y in s:
            if x + y in sums:
                return False
        return True

    def greedy_fill(s, sums, candidates, bias='random'):
        s = sorted(s)
        sums = get_sums(s)
        s_set = set(s)
        avail = [c for c in candidates if c not in s_set]
        
        if bias == 'random':
            random.shuffle(avail)
        elif bias == 'low':
            avail.sort()
        elif bias == 'high':
            avail.sort(reverse=True)
            
        for x in avail:
            if time.time() > deadline - 2:
                break
            if can_add(s, sums, x):
                # Update sums
                sums.add(2 * x)
                for y in s:
                    sums.add(x + y)
                s.append(x)
        return s, sums

    def polish(s, sums):
        """
        Try to add any remaining valid element to the set.
        This ensures the set is locally maximal with respect to single additions.
        """
        s = list(s)
        sums = get_sums(s)
        s_set = set(s)
        
        for x in range(1, max_val + 1):
            if x in s_set:
                continue
            if can_add(s, sums, x):
                s.append(x)
                s_set.add(x)
                sums.add(2 * x)
                for y in s:
                    if y != x:
                        sums.add(x + y)
        return s, sums

    # Initial greedy solutions to get a good starting point
    best_set = list(base)
    best_sums = get_sums(best_set)
    best_size = len(best_set)
    
    # Try multiple greedy starts
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias='random')
        s, sums = polish(s, sums)
        if len(s) > best_size:
            best_set = s
            best_sums = sums
            best_size = len(s)
            
    # If we already found the target, return
    if best_size >= target_size:
        return sorted(best_set)

    # Backtracking search to find a larger set
    # We will try to find a set of size > best_size
    # We use the best_set found so far to guide the search?
    # Or just do a global backtracking?
    # Global backtracking from base is safer.
    
    # To speed up backtracking, we can use the greedy solution to order candidates
    # Candidates in the best greedy solution are "good"
    greedy_elements = set(best_set)
    
    def score_candidate(x, s, sums):
        # Prefer candidates that are in the greedy solution?
        # Or prefer candidates that are "easy" to add?
        # Let's just return 0 for now, or use greedy membership
        if x in greedy_elements:
            return 1
        return 0

    # Recursive DFS
    # s: sorted list of current elements
    # sums: set of current sums
    # avail: list of available candidates
    # current_size: len(s)
    # best_size: nonlocal variable to track global best
    
    # We need to be careful with recursion depth and time.
    # We will limit the depth of backtracking or use iterative deepening?
    # Since target is 20, and we start with 4, depth is 16.
    # Branching factor is up to 284.
    # This is too large for naive DFS.
    
    # Better approach: Iterated Local Search (ILS) with a strong kick.
    # The current program does this. Why did it stop at 18?
    # Maybe the neighborhood is not rich enough.
    
    # Let's try a specific "Swap 2 for 3" move to increase size by 1.
    
    def try_increase_size(current_set, current_sums):
        """
        Try to increase the size of the set by 1.
        Strategy: Remove k elements, add k+1 elements.
        We try k=1 (remove 1, add 2) and k=2 (remove 2, add 3).
        """
        if time.time() > deadline - 5:
            return None
        
        non_base_elements = [x for x in current_set if x not in base_set]
        if not non_base_elements:
            return None
            
        # Try removing 1 element
        for to_remove in non_base_elements:
            if time.time() > deadline - 5:
                break
            temp_set = [x for x in current_set if x != to_remove]
            temp_sums = get_sums(temp_set)
            
            # Try to add 2 elements
            # We need to find two elements u, v that can be added to temp_set
            # and are compatible with each other.
            
            # Get candidates
            temp_set_set = set(temp_set)
            candidates = [c for c in all_candidates if c not in temp_set_set]
            
            # Filter candidates that can be added to temp_set
            valid_cands = []
            for c in candidates:
                if can_add(temp_set, temp_sums, c):
                    valid_cands.append(c)
                    
            if len(valid_cands) < 2:
                continue
                
            # Shuffle to randomize
            random.shuffle(valid_cands)
            
            # Try pairs
            # To limit complexity, only try a subset of pairs
            max_pairs_to_try = 200
            pairs_tried = 0
            for i in range(len(valid_cands)):
                u = valid_cands[i]
                # Update sums with u
                temp_sums_u = temp_sums.copy()
                temp_sums_u.add(2 * u)
                for y in temp_set:
                    temp_sums_u.add(u + y)
                
                for j in range(i + 1, len(valid_cands)):
                    v = valid_cands[j]
                    # Check if v can be added to temp_set + {u}
                    if can_add(temp_set + [u], temp_sums_u, v):
                        # Success!
                        new_set = temp_set + [u, v]
                        new_set.sort()
                        return new_set
                        
                    pairs_tried += 1
                    if pairs_tried > max_pairs_to_try:
                        break
                if pairs_tried > max_pairs_to_try:
                    break
                    
        # Try removing 2 elements
        # This is more expensive.
        if len(non_base_elements) >= 2:
            # Sample a few pairs to remove
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                to_remove = random.sample(non_base_elements, 2)
                remove_set = set(to_remove)
                temp_set = [x for x in current_set if x not in remove_set]
                temp_sums = get_sums(temp_set)
                
                temp_set_set = set(temp_set)
                candidates = [c for c in all_candidates if c not in temp_set_set]
                
                valid_cands = []
                for c in candidates:
                    if can_add(temp_set, temp_sums, c):
                        valid_cands.append(c)
                        
                if len(valid_cands) < 3:
                    continue
                    
                random.shuffle(valid_cands)
                
                # Try to add 3 elements
                # This is a 3-combination search.
                # Limit search.
                max_triples_to_try = 200
                triples_tried = 0
                for i in range(len(valid_cands)):
                    u = valid_cands[i]
                    temp_sums_u = temp_sums.copy()
                    temp_sums_u.add(2 * u)
                    for y in temp_set:
                        temp_sums_u.add(u + y)
                    
                    for j in range(i + 1, len(valid_cands)):
                        v = valid_cands[j]
                        if not can_add(temp_set + [u], temp_sums_u, v):
                            continue
                        
                        temp_sums_v = temp_sums_u.copy()
                        temp_sums_v.add(2 * v)
                        for y in temp_set + [u]:
                            temp_sums_v.add(v + y)
                        
                        for k in range(j + 1, len(valid_cands)):
                            w = valid_cands[k]
                            if can_add(temp_set + [u, v], temp_sums_v, w):
                                new_set = temp_set + [u, v, w]
                                new_set.sort()
                                return new_set
                                
                            triples_tried += 1
                            if triples_tried > max_triples_to_try:
                                break
                        if triples_tried > max_triples_to_try:
                            break
                    if triples_tried > max_triples_to_try:
                        break
                    if triples_tried > max_triples_to_try:
                        break
                        
        return None

    current_set = best_set[:]
    current_sums = best_sums
    current_size = best_size
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 10:
        
        # Try to increase size
        new_set = try_increase_size(current_set, current_sums)
        
        if new_set is not None:
            if len(new_set) > best_size:
                best_set = new_set
                best_sums = get_sums(new_set)
                best_size = len(new_set)
                current_set = new_set
                current_sums = best_sums
                no_improve_count = 0
                
                if best_size >= target_size:
                    break
            else:
                # Same size, just update current to explore
                current_set = new_set
                current_sums = get_sums(new_set)
                no_improve_count += 1
        else:
            # Try a random kick and refill to escape local optimum
            kick_size = random.choice([2, 3, 4])
            non_base_elements = [x for x in current_set if x not in base_set]
            if len(non_base_elements) >= kick_size:
                to_remove = random.sample(non_base_elements, kick_size)
                remove_set = set(to_remove)
                new_set = [x for x in current_set if x not in remove_set]
                new_sums = get_sums(new_set)
                
                # Refill
                new_set, new_sums = greedy_fill(new_set, new_sums, all_candidates, bias='random')
                new_set, new_sums = polish(new_set, new_sums)
                
                if len(new_set) > current_size:
                    current_set = new_set
                    current_sums = new_sums
                    no_improve_count = 0
                elif len(new_set) == current_size:
                    current_set = new_set
                    current_sums = new_sums
                    no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        if no_improve_count > max_no_improve:
            # Reset to best found
            current_set = best_set[:]
            current_sums = best_sums
            no_improve_count = 0
            # Also try a fresh random start if stuck too long
            if no_improve_count > max_no_improve * 2:
                 s, sums = greedy_fill(list(base), get_sums(list(base)), all_candidates, bias='random')
                 s, sums = polish(s, sums)
                 if len(s) > len(current_set):
                     current_set = s
                     current_sums = sums

    return sorted(best_set)

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
