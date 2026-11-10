import numpy as np
import time
import random

def _construct_optimized_greedy():
    """
    Optimized greedy construction using NumPy for fast conflict checking.
    Strategy: Iterate through candidates. For each candidate, check if adding it
    creates any sum with > 2 representations. 
    To speed up, we maintain a count array of sums. 
    When adding x, we need to check if any sum (x + y) for y in chosen has count >= 2.
    This is equivalent to checking if (counts[x + chosen] >= 2).any().
    """
    N = 1000
    max_sum = 2 * N
    
    # counts[s] = number of pairs (a, b) with a <= b, a+b=s from chosen set
    counts = np.zeros(max_sum + 1, dtype=np.int8)
    
    chosen = []
    chosen_arr = np.array([], dtype=np.int32)
    
    # We iterate through 1..1000. 
    # A simple ascending order is a good baseline. 
    # But "density first" might be better? 
    # Actually, for Sidon-like sets, spreading out is key.
    # Let's stick to a smart ordering or just optimized ascending.
    # Ascending is fast and produces a valid set.
    
    for x in range(1, N + 1):
        # Check conflicts
        if len(chosen_arr) > 0:
            # Sums created by adding x: x + chosen_arr
            # We need to ensure none of these sums already have count >= 2
            # Also check 2*x
            sums = x + chosen_arr
            # Check if any of these sums have count >= 2
            if np.any(counts[sums] >= 2):
                continue
            if counts[2 * x] >= 2:
                continue
        
        # Safe to add
        chosen.append(x)
        chosen_arr = np.append(chosen_arr, x)
        
        # Update counts
        if len(chosen_arr) > 1:
            # Previous elements
            prev = chosen_arr[:-1]
            new_sums = x + prev
            counts[new_sums] += 1
        counts[2 * x] += 1
        
    return chosen, counts

def _local_search(initial_set, counts, time_limit=10.0):
    """
    Try to improve the set size by swapping elements.
    If we remove x and add y, and size stays same, we might enable future additions?
    Actually, we want to INCREASE size. 
    Local search for size increase is hard with simple swaps (swap keeps size constant).
    However, we can try to "repair" or "re-optimize" the tail.
    
    Alternative Local Search: 
    1. Take the current set.
    2. Identify "bottleneck" sums (sums with count 2).
    3. Try to remove an element involved in a bottleneck sum and replace it with a 
       larger number that doesn't create new bottlenecks, potentially opening up 
       space for more numbers? 
       
    Actually, a simpler and effective local search for maximization is:
    Try to add ANY number from 1..1000 not in the set. If we can't add any, we are stuck.
    The greedy algorithm already tries all 1..1000. If it didn't add them, it's because 
    they conflict. 
    To add a conflicting number, we must remove another number that conflicts with it.
    This is a 1-swap: Remove a, Add b. Size is preserved. 
    But maybe removing a "bad" a allows adding TWO new numbers c and d?
    That's a 2-swap: Remove {a1, a2}, Add {b1, b2}. Net +0? No, we want +1.
    
    Let's try a different angle: 
    The greedy ascending approach might get stuck early. 
    What if we use a randomized greedy with a "lookahead" or "restart" strategy?
    The current program does random shuffles. 
    My optimized greedy is faster, so I can do MANY more random shuffles in the same time.
    
    So the strategy is:
    1. Run optimized greedy on ascending order.
    2. Run optimized greedy on many random permutations.
    3. Keep the best.
    """
    # This function is a placeholder for the actual loop logic in construct()
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Baseline: Optimized Greedy Ascending
    best_set, _ = _construct_optimized_greedy()
    best_size = len(best_set)
    
    # 2. Randomized Greedy Search
    # Since _construct_optimized_greedy is fast (NumPy), we can run many iterations.
    # We need to modify it to accept a permutation.
    
    def _greedy_with_order(order):
        N = 1000
        max_sum = 2 * N
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        chosen = []
        chosen_arr = np.array([], dtype=np.int32)
        
        for x in order:
            if len(chosen_arr) > 0:
                sums = x + chosen_arr
                if np.any(counts[sums] >= 2):
                    continue
                if counts[2 * x] >= 2:
                    continue
            
            chosen.append(x)
            chosen_arr = np.append(chosen_arr, x)
            
            if len(chosen_arr) > 1:
                prev = chosen_arr[:-1]
                new_sums = x + prev
                counts[new_sums] += 1
            counts[2 * x] += 1
            
        return chosen

    random.seed(42)
    base_candidates = list(range(1, 1001))
    
    # Try reverse order first (often good for packing)
    rev_set = _greedy_with_order(base_candidates[::-1])
    if len(rev_set) > best_size:
        best_set = rev_set
        best_size = len(best_set)
    
    # Random shuffles
    # Estimate how many shuffles we can do.
    # Each shuffle takes ~O(N^2) ops in numpy but vectorized. 
    # Let's say ~1ms per shuffle. 100s -> 100,000 shuffles? 
    # Actually, numpy overhead is higher. Let's just run until deadline.
    
    while time.time() < deadline:
        # Strategy: Random permutation
        order = base_candidates[:]
        random.shuffle(order)
        
        current_set = _greedy_with_order(order)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Strategy: Partially sorted / Biased
        # Sometimes, starting with small numbers then random large ones helps?
        # Or vice versa.
        if random.random() < 0.1:
            # Hybrid: First 10% sorted, rest random
            split = 100
            first_part = base_candidates[:split]
            second_part = base_candidates[split:]
            random.shuffle(second_part)
            order = first_part + second_part
        elif random.random() < 0.1:
            # Hybrid: Last 10% sorted, rest random
            split = 900
            first_part = base_candidates[:split]
            second_part = base_candidates[split:]
            random.shuffle(first_part)
            order = first_part + second_part
        else:
            # Pure random already done above, but let's add "Reverse Random"
            # i.e. shuffle, then reverse?
            pass
            
    return best_set

# To make the code cleaner and ensure _greedy_with_order is defined before use in construct
# I will restructure slightly.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def _greedy_with_order(order):
        N = 1000
        max_sum = 2 * N
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        chosen = []
        chosen_arr = np.array([], dtype=np.int32)
        
        for x in order:
            if len(chosen_arr) > 0:
                sums = x + chosen_arr
                if np.any(counts[sums] >= 2):
                    continue
                if counts[2 * x] >= 2:
                    continue
            
            chosen.append(x)
            chosen_arr = np.append(chosen_arr, x)
            
            if len(chosen_arr) > 1:
                prev = chosen_arr[:-1]
                new_sums = x + prev
                counts[new_sums] += 1
            counts[2 * x] += 1
            
        return chosen

    # 1. Baseline: Ascending
    best_set = _greedy_with_order(list(range(1, 1001)))
    best_size = len(best_set)
    
    # 2. Baseline: Descending
    desc_set = _greedy_with_order(list(range(1000, 0, -1)))
    if len(desc_set) > best_size:
        best_set = desc_set
        best_size = len(best_size)
        
    # 3. Randomized Search
    random.seed(42)
    base_candidates = list(range(1, 1001))
    
    # Pre-allocate a list for order to avoid repeated list creation overhead?
    # random.shuffle in place is fine.
    
    while time.time() < deadline:
        order = base_candidates[:]
        random.shuffle(order)
        
        # Occasionally try structured permutations
        if random.random() < 0.05:
            # Interleave small and large?
            small = base_candidates[:500]
            large = base_candidates[500:]
            random.shuffle(small)
            random.shuffle(large)
            # Interleave
            order = []
            i, j = 0, 0
            while i < 500 and j < 500:
                order.append(small[i])
                order.append(large[j])
                i += 1
                j += 1
            order.extend(small[i:])
            order.extend(large[j:])
            
        current_set = _greedy_with_order(order)
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
    return best_set
