import time
import random
import sys
from functools import lru_cache

DEADLINE_SECONDS = 85.0

def _check_divisibility(sums_set):
    """
    Returns True if there is a divisibility conflict in sums_set.
    sums_set is a set of integers.
    """
    # Optimization: Sort the sums. If s divides t, then t >= s.
    # We can iterate through sums and check multiples.
    # To speed up, we can just check if any s has a multiple in the set.
    # Since the set size is at most 2^k, and k is small (<=15),
    # we can do:
    for s in sums_set:
        # Check if 2s, 3s, ... are in sums_set
        # We don't need to check s itself.
        # Max sum is bounded by 256 * 15 = 3840.
        # So we only check up to max_sum.
        pass # We'll use a faster method in the search loop by maintaining a boolean array or similar if needed, 
             # but for small sets, set membership is O(1).
    
    # A more efficient check:
    # Sort the sums.
    # For each s in sorted_sums:
    #   for multiple in range(2*s, max_sum+1, s):
    #      if multiple in sums_set: return True
    # This is fast if the number of sums is small.
    
    return False

def _construct_validity(sums_set):
    """
    Efficiently check if the set of sums has divisibility conflicts.
    """
    # To make this faster, we can precompute the max sum.
    max_sum = max(sums_set) if sums_set else 0
    # We only need to check divisors up to max_sum // 2
    # For each s in sums_set, if s <= max_sum // 2, check multiples.
    # If s > max_sum // 2, it cannot divide any other distinct sum in the set.
    
    # Optimization: Create a boolean array for O(1) lookup
    # But creating the array every time is O(max_sum).
    # Since max_sum is small (<4000), this is fine.
    present = [False] * (max_sum + 1)
    for s in sums_set:
        if s <= max_sum:
            present[s] = True
            
    for s in sums_set:
        if s > max_sum // 2:
            continue
        # Check multiples
        m = 2 * s
        while m <= max_sum:
            if present[m]:
                return False # Conflict found
            m += s
            
    return True # No conflict

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    best_size = 0
    best_set = []
    
    # Candidate pool
    # We want to find a large set.
    # Let's try searching in different ranges or orders.
    
    # Heuristic: Larger numbers are better?
    # Let's use a backtracking search with ordering.
    
    candidates = list(range(1, 257))
    
    # We will try several orders and run a bounded search.
    
    def search(current_set, current_sums, next_idx, order):
        nonlocal best_size, best_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Pruning: If even adding all remaining candidates doesn't beat best_size
        remaining = len(order) - next_idx
        if len(current_set) + remaining <= best_size:
            return
        
        if time.time() > deadline - 1:
            return
            
        if next_idx >= len(order):
            return
            
        x = order[next_idx]
        
        # Option 1: Include x
        # Check if adding x is valid
        # New sums = current_sums U {x} U {s + x for s in current_sums}
        
        # Optimization: Before computing new sums, check if x conflicts with existing sums?
        # If x is in current_sums, then subset {x} and the subset that formed x have same sum?
        # No, x is an element. The sum {x} is x.
        # If x is in current_sums, it means some other subset sums to x.
        # Then we have two subsets with sum x. This is invalid (sums must be distinct values, 
        # but the rule says "no two distinct resulting values may divide one another".
        # "Equal sum values are counted once."
        # "Two different subsets with the same sum still make the set invalid: removing their common elements leaves two disjoint subsets with the same sum s, and their union has sum 2s, which s divides."
        # So if any two subsets have the same sum, it's invalid.
        # Therefore, all subset sums must be distinct.
        
        # So, if x in current_sums, adding x is invalid.
        # Also, if s + x in current_sums for any s in current_sums, invalid.
        
        if x in current_sums:
            # Must skip x
            search(current_set, current_sums, next_idx + 1, order)
            return
            
        # Check if s+x exists in current_sums
        conflict = False
        for s in current_sums:
            if s + x in current_sums:
                conflict = True
                break
        
        if conflict:
            search(current_set, current_sums, next_idx + 1, order)
            return
            
        # Compute new sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
            
        # Check divisibility
        if _construct_validity(new_sums):
            current_set.append(x)
            search(current_set, new_sums, next_idx + 1, order)
            current_set.pop()
            
        # Option 2: Exclude x
        search(current_set, current_sums, next_idx + 1, order)
        
    # We need to be careful with the recursion depth and time.
    # Branch and bound with 256 items is too deep.
    # We should limit the search to a subset of candidates or use a greedy initialization to raise the bar.
    
    # Phase 1: Greedy to get a good baseline
    def greedy(pool, order_key):
        elems = []
        sums_set = set()
        cands = list(pool)
        if order_key == 'desc':
            cands.sort(reverse=True)
        elif order_key == 'asc':
            cands.sort()
        else:
            random.shuffle(cands)
            
        for x in cands:
            if x in sums_set:
                continue
            conflict = False
            for s in sums_set:
                if s + x in sums_set:
                    conflict = True
                    break
            if conflict:
                continue
                
            new_sums = set(sums_set)
            new_sums.add(x)
            for s in sums_set:
                new_sums.add(s + x)
                
            if _construct_validity(new_sums):
                sums_set = new_sums
                elems.append(x)
                
        return elems

    # Try a few greedy runs to establish a lower bound
    pools = [
        list(range(1, 257)),
        list(range(128, 257)),
        list(range(64, 257)),
    ]
    
    best_greedy_set = []
    for pool in pools:
        for key in ['desc', 'asc', 'random']:
            g_set = greedy(pool, key)
            if len(g_set) > len(best_greedy_set):
                best_greedy_set = g_set
                
    if best_greedy_set:
        best_size = len(best_greedy_set)
        best_set = list(best_greedy_set)
        
    # Phase 2: Local Search / Improvement
    # Try to swap elements or add more.
    
    def local_improve(init_set):
        current_set = list(init_set)
        sums_set = set()
        for e in current_set:
            # Rebuild sums
            pass # Rebuilding is O(k^2 2^k) which is bad.
            # We should maintain sums.
            
        # Let's rebuild sums for the initial set
        sums_set = set()
        for e in current_set:
            new_sums = set(sums_set)
            new_sums.add(e)
            for s in sums_set:
                new_sums.add(s + e)
            sums_set = new_sums
            
        best_local_set = list(current_set)
        best_local_size = len(current_set)
        
        # Try adding random elements
        used = set(current_set)
        remaining = [x for x in range(1, 257) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if x in sums_set:
                continue
            conflict = False
            for s in sums_set:
                if s + x in sums_set:
                    conflict = True
                    break
            if conflict:
                continue
            
            new_sums = set(sums_set)
            new_sums.add(x)
            for s in sums_set:
                new_sums.add(s + x)
                
            if _construct_validity(new_sums):
                sums_set = new_sums
                current_set.append(x)
                best_local_set = list(current_set)
                best_local_size = len(current_set)
                
        # Try swaps
        # This is complex to implement efficiently with the sums set.
        # We'll rely on the greedy + random restarts.
        
        return best_local_set

    # Run several local improvement rounds
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        improved = local_improve(best_set)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            
    return [int(x) for x in best_set]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")
