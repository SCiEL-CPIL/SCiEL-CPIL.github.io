import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    # Bitset for sums. Max sum is 284+284=568.
    # We need bits 0..568.
    
    def get_sums_bitset(s):
        sums = 0
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums |= (1 << (lst[i] + lst[j]))
        return sums

    def can_add(x, sums, s_set):
        if x in s_set:
            return False
        # Check 2x
        if (sums >> (2 * x)) & 1:
            return False
        # Check x + y
        for y in s_set:
            if (sums >> (x + y)) & 1:
                return False
        return True

    def update_sums(sums, x, s_set):
        new_sums = sums
        new_sums |= (1 << (2 * x))
        for y in s_set:
            new_sums |= (1 << (x + y))
        return new_sums

    def greedy_fill(prescribed, target_size, randomize=True):
        current_set = set(prescribed)
        current_sums = get_sums_bitset(current_set)
        candidates = list(range(12, 285))
        if randomize:
            random.shuffle(candidates)
        
        for x in candidates:
            if len(current_set) >= target_size:
                break
            if can_add(x, current_sums, current_set):
                current_set.add(x)
                current_sums = update_sums(current_sums, x, current_set)
        return current_set, current_sums

    def backtrack_extend(base_set, base_sums, target_size, deadline):
        current_set = list(base_set)
        current_sums = base_sums
        depth = target_size - len(current_set)
        
        if depth < 0:
            return None
        if depth == 0:
            return current_set
        
        last_val = max(current_set)
        
        # Pruning: Not enough numbers left
        if (284 - last_val) < depth:
            return None
            
        # To speed up, we limit the number of random restarts inside the backtracking?
        # No, we just do one DFS with randomized candidate order.
        # If it fails, the caller will try a different base.
        
        def rec(d, c_set, c_sums, last):
            if d == 0:
                return c_set
            if time.time() > deadline:
                return None
            if (284 - last) < d:
                return None
            
            # Generate candidates > last
            # Optimization: Pre-filter?
            # can_add is O(N). N is small.
            
            # Randomize order to avoid bias
            cands = [x for x in range(last + 1, 285) if x not in c_set]
            random.shuffle(cands)
            
            for x in cands:
                if can_add(x, c_sums, c_set):
                    new_set = c_set + [x]
                    new_sums = update_sums(c_sums, x, c_set)
                    res = rec(d - 1, new_set, new_sums, x)
                    if res is not None:
                        return res
            return None

        return rec(depth, current_set, current_sums, last_val)

    best_set = set(prescribed_set)
    best_size = len(best_set)
    best_sums = get_sums_bitset(best_set)
    
    # Try to find size 20, 19, etc.
    # We allow up to 5 backtracking steps (depth <= 5)
    # So if greedy gets to 15, we can try to reach 20.
    
    for target in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 5:
            break
        
        found = False
        attempts = 0
        max_attempts = 50 # Limit attempts per target to prevent infinite loop if backtracking is slow
        
        while time.time() < deadline - 5 and not found and attempts < max_attempts:
            attempts += 1
            # Greedy fill to target - 5 (so depth is at most 5)
            base_target = max(target - 5, len(prescribed_set))
            base_set, base_sums = greedy_fill(prescribed, base_target, randomize=True)
            
            # If greedy didn't even reach base_target, skip
            if len(base_set) < base_target:
                continue
                
            # If greedy already reached target, we are done
            if len(base_set) >= target:
                if len(base_set) > best_size:
                    best_set = set(base_set)
                    best_size = len(base_set)
                    best_sums = base_sums
                found = True
                if target == 20:
                    return sorted(list(best_set))
                continue

            # Try to extend to target
            res = backtrack_extend(base_set, base_sums, target, deadline - 5)
            if res is not None:
                if len(res) > best_size:
                    best_set = set(res)
                    best_size = len(res)
                    best_sums = get_sums_bitset(best_set)
                found = True
                if target == 20:
                    return sorted(list(best_set))
                
        # If not found, try next target
        # Note: We don't break the outer loop here, we just move to next target
        # But if we found a set of size `target`, we can break early if it's 20.
        
    # Fallback: If we didn't find a good set via backtracking, 
    # we might have a good greedy set. 
    # Or we can run a quick local search on the best_set.
    
    # Quick Local Search Polish
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add
            for x in range(12, 285):
                if can_add(x, best_sums, best_set):
                    best_set.add(x)
                    best_sums = update_sums(best_sums, x, best_set)
                    improved = True
                    break
            if improved:
                continue
                
            # Try swap
            removable = [x for x in best_set if x not in prescribed_set]
            if not removable:
                break
            random.shuffle(removable)
            for x_remove in removable:
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums_bitset(temp_set)
                
                # Try to add any valid y
                for y_add in range(12, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_sums, temp_set):
                        # Accept swap to explore
                        best_set = temp_set
                        best_set.add(y_add)
                        best_sums = update_sums(temp_sums, y_add, temp_set)
                        improved = True
                        break
                if improved:
                    break
                    
    # Final validation
    final_sums = get_sums_bitset(best_set)
    # Check if valid
    # We assume validity is maintained by construction.
    
    return sorted(list(best_set))
