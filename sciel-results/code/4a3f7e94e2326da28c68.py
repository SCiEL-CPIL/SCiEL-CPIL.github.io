import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    candidates = list(range(256, 0, -1))
    n_candidates = len(candidates)
    max_possible_sum = 256 * 16
    
    best_size = 0
    best_set = []
    
    def check_addition(current_sums_list, current_sums_set, candidate):
        # current_sums_list is assumed sorted
        new_sums = [candidate]
        for s in current_sums_list:
            new_sums.append(s + candidate)
        
        new_sums_set = set(new_sums)
        
        # Check collisions
        if new_sums_set.intersection(current_sums_set):
            return None
        
        # Check divisibility: current vs new
        # 1. Does any b in new divide any a in current? (b | a)
        # Since new_sums are generally larger, b | a is unlikely if b > a.
        # We only need to check b <= a.
        # Sort new_sums to check efficiently? Or just iterate.
        # Let's sort new_sums for the internal check later anyway.
        new_sums_sorted = sorted(new_sums)
        
        # 1. b | a check
        # For each b in new_sums, check if any a in current_sums is a multiple of b.
        # Since current_sums_list is sorted, we can stop checking a once a < b.
        for b in new_sums_sorted:
            if b == 0: continue
            # Find first a >= b in current_sums_list
            # Simple linear scan from start? No, current_sums is sorted.
            # We can just iterate current_sums_list and break if a < b?
            # No, if a < b, a cannot be a multiple of b (unless a=0, impossible).
            # So we only care about a >= b.
            for a in current_sums_list:
                if a < b:
                    continue
                if a % b == 0:
                    return None
                # Optimization: if a is much larger, we can't break easily, 
                # but the modulo is cheap.
                
        # 2. Does any a in current divide any b in new? (a | b)
        # Use the multiple array trick.
        is_multiple_of_current = [False] * (max_possible_sum + 1)
        for a in current_sums_list:
            if a > 0:
                for m in range(a, max_possible_sum + 1, a):
                    is_multiple_of_current[m] = True
        
        for b in new_sums:
            if b <= max_possible_sum and is_multiple_of_current[b]:
                return None
        
        # 3. Internal divisibility in new_sums
        # new_sums_sorted is already sorted
        n = len(new_sums_sorted)
        for i in range(n):
            a = new_sums_sorted[i]
            if a == 0: continue
            for j in range(i + 1, n):
                b = new_sums_sorted[j]
                if b % a == 0:
                    return None
        
        return (new_sums, new_sums_set)

    def dfs(idx, current_sums_list, current_sums_set, current_elements):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        if len(current_elements) + (n_candidates - idx) <= best_size:
            return
            
        if idx == n_candidates:
            if len(current_elements) > best_size:
                best_size = len(current_elements)
                best_set = current_elements[:]
            return

        # Skip
        dfs(idx + 1, current_sums_list, current_sums_set, current_elements)
        
        # Include
        candidate = candidates[idx]
        result = check_addition(current_sums_list, current_sums_set, candidate)
        if result is not None:
            new_sums_list, new_sums_set = result
            # Keep current_sums_list sorted for next step?
            # new_sums_list is not necessarily sorted relative to current_sums_list.
            # We should merge them and sort.
            # Actually, check_addition returns new_sums (unsorted).
            # Let's merge and sort.
            merged = current_sums_list + new_sums_list
            merged.sort()
            current_elements.append(candidate)
            dfs(idx + 1, merged, new_sums_set.union(current_sums_set), current_elements)
            current_elements.pop()

    dfs(0, [], set(), [])
    
    if not best_set:
        return [256]
        
    return [int(x) for x in best_set]
