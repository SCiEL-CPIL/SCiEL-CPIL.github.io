"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute bad triples and conflict pairs
    bad_pairs_for = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(b, c) == g_ab and gcd(c, a) == g_ab:
                    bad_pairs_for[a].append((b, c))
                    bad_pairs_for[b].append((a, c))
                    bad_pairs_for[c].append((a, b))
                    
    # 2. Initial Greedy Solution to establish a lower bound (best_size)
    # Heuristic: Add elements with fewest conflicts first
    order_greedy = sorted(range(1, N + 1), key=lambda x: len(bad_pairs_for[x]))
    current_set = set()
    for x in order_greedy:
        can_add = True
        for y, z in bad_pairs_for[x]:
            if y in current_set and z in current_set:
                can_add = False
                break
        if can_add:
            current_set.add(x)
            
    best_size = len(current_set)
    best_set = current_set.copy()
    
    # 3. Backtracking Search with Branch and Bound
    # Order variables by descending conflict degree to prune earlier
    order_bt = sorted(range(1, N + 1), key=lambda x: len(bad_pairs_for[x]), reverse=True)
    
    # Optimization: Pre-compute for each x, the list of pairs (y, z)
    # We already have bad_pairs_for
    
    # To speed up conflict check, we can maintain a set of included elements.
    # Conflict check for x: any (y, z) in bad_pairs_for[x] with y, z in included?
    
    # We will use a recursive function.
    # To avoid deep recursion overhead and state copying, we can use a list for included
    # and backtrack by appending/popping. But we need O(1) membership check.
    # A set is better. Copying a set of size ~100 is fast enough for 128 depth?
    # 2^128 is too big. We rely on pruning.
    # If pruning is effective, number of nodes visited is small.
    
    # Let's try to implement the search iteratively or with efficient state management.
    # Given the constraint of 60s, and Python speed, we might need to be very careful.
    
    # Alternative: Use a SAT-like approach? No, stick to Backtracking.
    
    # We'll use a helper to check validity.
    def can_include(x, included):
        for y, z in bad_pairs_for[x]:
            if y in included and z in included:
                return False
        return True

    # Recursive search
    # idx: index in order_bt
    # included: set of included elements
    # count: number of elements included so far (len(included))
    
    # To improve performance, we can pass `included` as a set and use `add/remove` 
    # or pass a new set. Passing a new set is safer but slower. 
    # Given N=128, and likely high pruning, let's try passing a set.
    
    # We can also add a simple heuristic: 
    # If the maximum possible size from this point (count + remaining) <= best_size, prune.
    
    # Let's wrap the search in a timeout or limit on nodes? 
    # The task says "finish within 60s". 
    # If the search is too slow, we should fall back to the greedy result.
    
    import time
    start_time = time.time()
    timeout = 55.0 # Leave 5s for safety
    
    nodes_visited = 0
    
    def search(idx, included, count):
        nonlocal best_size, best_set, nodes_visited
        
        # Check timeout
        if time.time() - start_time > timeout:
            return
            
        # Pruning: Even if we add all remaining, can we beat best_size?
        if count + (len(order_bt) - idx) <= best_size:
            return
            
        if idx == len(order_bt):
            if count > best_size:
                best_size = count
                best_set = included.copy()
            return
            
        x = order_bt[idx]
        
        # Option 1: Include x
        if can_include(x, included):
            included.add(x)
            nodes_visited += 1
            search(idx + 1, included, count + 1)
            included.remove(x)
            
        # Option 2: Exclude x
        nodes_visited += 1
        search(idx + 1, included, count)

    # Start search
    search(0, set(), 0)
    
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
