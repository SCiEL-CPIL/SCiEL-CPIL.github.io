import time
import random

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin
    
    TARGET = 200
    MAX_ELE = 100
    MASK = (1 << (TARGET + 1)) - 1
    TARGET_BIT = 1 << TARGET
    
    def can_add(bitset, x):
        """Check if adding element x would create a subset sum of 200."""
        return not ((bitset << x) & TARGET_BIT)
    
    def add_element(bitset, x):
        """Add element x to the reachable sums bitset."""
        return bitset | ((bitset << x) & MASK)
        
    def greedy_from(start_list, candidates):
        """
        Start with start_list, try to add elements from candidates if possible.
        Returns (list, bitset, size).
        """
        cur = list(start_list)
        cur_set = set(cur)
        bitset = 1
        for x in cur:
            bitset = add_element(bitset, x)
            
        for x in candidates:
            if x in cur_set:
                continue
            if can_add(bitset, x):
                cur.append(x)
                cur_set.add(x)
                bitset = add_element(bitset, x)
                
        return cur, bitset, len(cur)

    best_set = []
    best_len = 0
    best_bitset = 1

    # Strategy 1: Greedy Ascending (1 to 100)
    s1, b1, l1 = greedy_from([], range(1, MAX_ELE + 1))
    if l1 > best_len:
        best_set, best_len, best_bitset = s1, l1, b1
        
    # Strategy 2: Greedy Descending (100 to 1)
    s2, b2, l2 = greedy_from([], range(MAX_ELE, 0, -1))
    if l2 > best_len:
        best_set, best_len, best_bitset = s2, l2, b2
        
    # Strategy 3: Start with known good interval [66, 100], add smaller
    base_interval = list(range(66, MAX_ELE + 1))
    s3, b3, l3 = greedy_from(base_interval, range(65, 0, -1))
    if l3 > best_len:
        best_set, best_len, best_bitset = s3, l3, b3
        
    # Strategy 4: Start with known good interval [66, 100], add smaller (ascending)
    s4, b4, l4 = greedy_from(base_interval, range(1, 66))
    if l4 > best_len:
        best_set, best_len, best_bitset = s4, l4, b4
        
    # Strategy 5: Randomized Greedy
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        candidates = list(range(1, MAX_ELE + 1))
        rng.shuffle(candidates)
        s5, b5, l5 = greedy_from([], candidates)
        if l5 > best_len:
            best_set, best_len, best_bitset = s5, l5, b5
            
    # Strategy 6: Local Search on Best Set (Simple Swap)
    # Try to replace one element with a smaller one to free up sum capacity?
    # Or replace a large one with two small ones? No, we want max cardinality.
    # Just try to add any missing element by removing one existing element.
    cur = list(best_set)
    cur_set = set(cur)
    bitset = best_bitset
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Rebuild bitset to be safe
        bitset = 1
        for x in cur:
            bitset = add_element(bitset, x)
            
        n = len(cur)
        for i in range(n):
            if time.time() > deadline - 5.0:
                break
            removed = cur[i]
            # Bitset without cur[i] is hard to compute directly.
            # Recompute prefix/suffix is O(N^2) per iteration which is O(N^3) total.
            # N is ~50. 50^3 = 125,000 ops. Very fast.
            
            # Compute bitset without cur[i]
            # We can just recompute the whole bitset for cur without i?
            # That is O(N) per check, O(N^2) per pass, O(N^3) total.
            # Let's just do it.
            
            temp_cur = cur[:i] + cur[i+1:]
            temp_bitset = 1
            for x in temp_cur:
                temp_bitset = add_element(temp_bitset, x)
                
            # Try to add any x not in cur
            for x in range(1, MAX_ELE + 1):
                if x in cur_set:
                    continue
                if can_add(temp_bitset, x):
                    # Swap cur[i] with x
                    cur[i] = x
                    cur_set.discard(removed)
                    cur_set.add(x)
                    improved = True
                    break
            if improved:
                break
                
    if len(cur) > best_len:
        best_set = cur
        best_len = len(cur)

    return best_set if best_set else [MAX_ELE]
