"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import sys
import time

# EVOLVE-BLOCK-START
def construct():
    N = 256
    best_set = list(range(171, 257))
    best_size = len(best_set)
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin
    
    # Precompute multiples?
    # For a given 'a', conflicts are pairs (b,c) in S such that a | (b+c).
    
    def is_valid(a, S):
        # Check if adding 'a' to set S (where all elements > a? No, S can have anything)
        # Constraint: for all b,c in S, if a < b <= c, a does not divide b+c.
        # Note: b,c must be in S. a is the new element.
        # We assume S contains elements > a? 
        # In our search, we process in descending order? 
        # If we process descending, S contains elements > a.
        # Then b,c > a is guaranteed.
        # So we just check: exists b,c in S such that a | (b+c)?
        # Since b,c > a, b+c > 2a.
        # Multiples of a: 3a, 4a, ...
        
        # Optimization: if 3*a > 2*max(S), return True (valid)
        if not S:
            return True
        max_s = max(S)
        if 3 * a > 2 * max_s:
            return True
            
        # Check multiples
        # b+c = k*a
        # min sum = min(S)+min(S) but b,c distinct? No b<=c.
        # b,c in S.
        # We can iterate k.
        k_min = (2 * min(S) + a - 1) // a + 1 # Actually b,c > a, so min sum is 2*(min(S)) if min(S)>a.
        # Since we add in descending order, S contains elements > a.
        # So min(S) > a.
        # min_sum = 2 * min(S)
        # max_sum = 2 * max(S)
        
        min_s = min(S)
        max_s = max(S)
        
        k_start = (2 * min_s) // a
        if (2 * min_s) % a == 0:
            k_start += 1 # Must be strictly greater? No, b+c can be multiple.
            # Wait, if b+c = k*a, it's a conflict.
            # So we check if any multiple exists.
            
        # Iterate possible sums
        # To speed up, we can use a set of S.
        S_set = set(S)
        
        # Range of sums: [2*min_s, 2*max_s]
        # Multiples of a in this range.
        first_mult = ((2 * min_s) // a + 1) * a
        last_mult = (2 * max_s // a) * a
        
        for sum_val in range(first_mult, last_mult + 1, a):
            # Check if sum_val can be formed by b+c with b,c in S, b<=c
            # b = sum_val - c
            # c ranges from max(sum_val - max_s, min_s) to min(sum_val - min_s, max_s)
            # And b <= c => sum_val - c <= c => 2c >= sum_val => c >= sum_val/2
            c_min = max(sum_val - max_s, min_s, (sum_val + 1) // 2)
            c_max = min(sum_val - min_s, max_s)
            
            for c in range(c_min, c_max + 1):
                b = sum_val - c
                if b in S_set:
                    return False
        return True

    def branch_and_bound(index, current_set, current_size):
        nonlocal best_set, best_size, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if index == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
        
        # Upper bound: current_size + index (if we add all remaining)
        if current_size + index <= best_size:
            return
            
        # Option 1: Include index
        # We process in descending order: 256, 255, ..., 1
        # So 'index' corresponds to value 'index+1' ?
        # Let's map index i to value v = N - i.
        # i=0 -> 256, i=1 -> 255 ... i=255 -> 1.
        
        v = N - index
        
        # Check if v can be added to current_set
        # current_set contains values > v (since we went down from 256)
        # So all b,c in current_set are > v.
        if is_valid(v, current_set):
            new_set = current_set | {v}
            branch_and_bound(index + 1, new_set, current_size + 1)
            
        # Option 2: Exclude v
        branch_and_bound(index + 1, current_set, current_size)

    # Start B&B
    # Initial set is empty.
    # But we know [171, 256] is valid.
    # To speed up, we can start with a greedy solution.
    # Greedy:
    greedy_set = set()
    for v in range(N, 0, -1):
        if is_valid(v, greedy_set):
            greedy_set.add(v)
    
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # Now run B&B to see if we can beat greedy.
    # B&B is expensive. We limit the search.
    # We will only explore if we think we can beat best_size.
    
    # To make B&B faster, we can prune if index is small.
    # Actually, if greedy found 86, and B&B takes too long, it might not finish.
    # Given the constraints, if B&B doesn't find better, it returns greedy.
    
    # A full B&B on 256 is too slow.
    # We will perform a limited search:
    # We try to include small numbers that were excluded by greedy.
    # Greedy excluded 1..170.
    # We can try to swap.
    
    # For safety and to meet the directive's "B&B" requirement without timing out:
    # We run B&B but with a strict node limit.
    
    node_count = 0
    max_nodes = 100000
    
    def bbb(index, current_set, current_size):
        nonlocal best_set, best_size, node_count, start_time, time_limit
        
        if time.time() - start_time > time_limit or node_count > max_nodes:
            return
        
        node_count += 1
        
        if index == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        if current_size + index <= best_size:
            return
            
        v = N - index
        
        # Include v
        if is_valid(v, current_set):
            bbb(index + 1, current_set | {v}, current_size + 1)
            
        # Exclude v
        bbb(index + 1, current_set, current_size)

    # Start from empty set, but this is huge.
    # Better: Start from greedy set and try to improve?
    # No, B&B usually builds from scratch.
    # Given the time limit, if the greedy is optimal, B&B will just verify or fail to find better.
    # If B&B times out, we return the greedy set.
    
    # To avoid wasting time, we only run B&B if we suspect improvement is possible.
    # But we don't know.
    # We will run B&B.
    
    bbb(0, set(), 0)
    
    return best_set
# EVOLVE-BLOCK-END
