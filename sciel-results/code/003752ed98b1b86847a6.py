import time
import random
import sys
import math

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    # Fast subset sum computation
    def compute_sums(elements):
        """Compute all non-empty subset sums. Returns set or None if duplicates."""
        sums = set()
        for x in elements:
            new_sums = set()
            for s in sums:
                ns = s + x
                if ns in sums:
                    return None
                new_sums.add(ns)
            if x in sums:
                return None
            new_sums.add(x)
            sums = sums.union(new_sums)
        return sums
    
    # Check divisibility constraint on a set of sums
    def check_divisibility(sums):
        """Check no two distinct sums divide each other. Returns True if valid."""
        s_list = sorted(sums)
        n = len(s_list)
        for i in range(n):
            si = s_list[i]
            for j in range(i+1, n):
                sj = s_list[j]
                if sj % si == 0:
                    return False
        return True
    
    # Full validation
    def is_valid(elements):
        if not elements:
            return True
        if len(elements) != len(set(elements)):
            return False
        if any(x < 1 or x > 256 for x in elements):
            return False
        sums = compute_sums(elements)
        if sums is None:
            return False
        return check_divisibility(sums)
    
    # Incremental check: can we add x to current_elements (which is valid)?
    # current_sums is the set of all non-empty subset sums of current_elements
    def can_add(x, current_elements, current_sums):
        """
        Check if x can be added to current_elements.
        Returns (new_sums, True) if valid, (None, False) otherwise.
        """
        if x in current_elements:
            return None, False
        
        # 1. Check for sum collisions (uniqueness of subset sums)
        # New sums will be {x} U {s + x for s in current_sums}
        # These must be disjoint from current_sums and from each other.
        
        new_sums = set()
        # Add x
        if x in current_sums:
            return None, False
        new_sums.add(x)
        
        # Add s + x
        for s in current_sums:
            ns = s + x
            if ns in current_sums or ns in new_sums:
                return None, False
            new_sums.add(ns)
            
        # 2. Check divisibility
        # The full set of sums is current_sums U new_sums
        # We only need to check pairs where at least one is in new_sums?
        # No, we need to ensure NO two distinct sums in (current_sums U new_sums) divide each other.
        # Since current_sums is already valid, we only need to check:
        # a) new_sums vs new_sums
        # b) new_sums vs current_sums
        
        all_sums = current_sums.union(new_sums)
        
        # Optimization: Check divisibility only involving new_sums
        # If a, b in current_sums, they are already non-dividing.
        # If a, b in new_sums, we must check.
        # If a in new_sums, b in current_sums, we must check.
        
        # To be safe and simple, we can just check all pairs in all_sums if size is small.
        # But for efficiency, let's check only interactions with new_sums.
        
        # Check new_sums internal
        ns_list = sorted(new_sums)
        for i in range(len(ns_list)):
            si = ns_list[i]
            for j in range(i+1, len(ns_list)):
                sj = ns_list[j]
                if sj % si == 0:
                    return None, False
        
        # Check new_sums vs current_sums
        for sn in new_sums:
            for sc in current_sums:
                if sn == sc:
                    return None, False # Already checked for collision, but just in case
                if sn > sc:
                    if sn % sc == 0:
                        return None, False
                else:
                    if sc % sn == 0:
                        return None, False
        
        return all_sums, True

    # DFS with backtracking to find a large valid set
    def dfs_search(target_size, time_limit, max_depth=15):
        """
        Attempt to build a set of size at least target_size.
        Uses randomized backtracking.
        """
        best_found = []
        
        def backtrack(current_elements, current_sums, depth):
            nonlocal best_found
            
            if time.time() > time_limit:
                return
            
            if len(current_elements) > len(best_found):
                best_found = list(current_elements)
                
            if len(current_elements) >= target_size:
                return
            
            if depth >= max_depth:
                return
                
            # Generate candidates
            # Heuristic: Prefer larger numbers? Or numbers that don't create small sums?
            # Let's try random candidates from 1..256 not in current_elements
            available = [x for x in range(1, 257) if x not in current_elements]
            
            # Sort candidates by some heuristic to find solutions faster?
            # Randomizing order helps escape local minima.
            random.shuffle(available)
            
            for x in available:
                if time.time() > time_limit:
                    return
                
                new_sums, valid = can_add(x, current_elements, current_sums)
                if valid:
                    # Recurse
                    backtrack(current_elements + [x], new_sums, depth + 1)
                    
                    # If we found a valid set of target size, we can stop early?
                    # Or continue to see if we can go even larger?
                    # The task asks to maximize size. 
                    # If we reach target_size, we might want to try to exceed it.
                    # But DFS stops when len >= target_size in the base case above.
                    # Let's modify: don't stop immediately, but if we found target_size, 
                    # we can break out of this loop to try other branches? 
                    # Actually, standard DFS: if we found a solution, we might want to 
                    # explore further to see if we can get size > target_size.
                    
                    # For efficiency, if we found target_size, we can return True to 
                    # signal success and let the caller decide if they want more.
                    # But we want MAX size. So we should keep searching if time permits.
                    # However, if we found a set of size 10, and we are looking for 10,
                    # we can stop this branch. But we might find 11 in another branch.
                    
                    # Let's just let the recursion handle it. The base case `if len >= target` 
                    # prevents deeper recursion in that branch, but other branches are explored.
                    
                    # Optimization: If we found a set of size target_size, we can 
                    # potentially prune other branches if we are confident? 
                    # No, we want max size.
                    
                    # If we just found a valid set of size >= target_size, 
                    # we can record it. 
                    
                    # If we want to be faster, we can stop if we find a set of size 
                    # close to the theoretical max or if time is running out.
                    
                    # Let's add a check: if we found a set of size target_size, 
                    # we can break out of the candidate loop to try other top-level 
                    # candidates? No, that would miss larger sets in this branch.
                    
                    # So we just continue.
                    
                    # If we found a valid set, we can return to parent to try next candidate.
                    # But if we want to maximize, we should explore all candidates.
                    
                    # One optimization: If we found a set of size target_size, 
                    # and we are at depth 0, we might want to stop? 
                    # No, we want max.
                    
                    # Let's just let it run until time limit or max depth.
        
        # Start search
        backtrack([], set(), 0)
        
        return best_found

    # Main search logic
    best_set = []
    
    # Try to find a set of increasing sizes
    # We start with a reasonable guess and increase.
    # The max size is likely around 10-15 given the constraints.
    
    # First, a quick greedy to get a baseline
    current_elements = []
    current_sums = set()
    
    # Greedy initialization
    candidates = list(range(256, 1, -1))
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline:
            break
        new_sums, valid = can_add(x, current_elements, current_sums)
        if valid:
            current_elements.append(x)
            current_sums = new_sums
            if len(current_elements) > len(best_set):
                best_set = list(current_elements)
                
    # Now use DFS to try to improve
    # We will try to find a set of size len(best_set) + 1, then +2, etc.
    
    current_target = len(best_set) + 1
    max_target = 20 # Upper bound estimate
    
    while current_target <= max_target:
        if time.time() > deadline - 5:
            break
            
        # Allocate time for this target
        time_alloc = min(deadline - time.time() - 2, 10.0)
        search_deadline = time.time() + time_alloc
        
        result = dfs_search(current_target, search_deadline)
        
        if result and len(result) >= current_target:
            if len(result) > len(best_set):
                best_set = result
                current_target = len(best_set) + 1
            else:
                # Found a set of the same size, maybe try again with different randomization?
                # Or just increment target if we are confident?
                # Let's increment target to push for larger sets.
                current_target += 1
        else:
            # Couldn't find a set of this size.
            # It's likely that this size is not possible or we just missed it.
            # We can try again with more time? Or assume max reached?
            # Let's try one more time with a different seed/randomization if time permits.
            if time.time() < deadline - 5:
                result2 = dfs_search(current_target, time.time() + 5.0)
                if result2 and len(result2) >= current_target:
                    if len(result2) > len(best_set):
                        best_set = result2
                        current_target = len(best_set) + 1
                    else:
                        current_target += 1
                else:
                    # Give up on this size
                    break
            else:
                break

    # Final validation and cleanup
    if best_set:
        if not is_valid(best_set):
            # Fallback to a known valid small set if something went wrong
            # [100, 101] is usually safe
            best_set = [100, 101]
            
    # Ensure types
    best_set = [int(x) for x in best_set if 1 <= x <= 256]
    seen = set()
    unique = []
    for x in best_set:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    best_set = unique
    
    return best_set
