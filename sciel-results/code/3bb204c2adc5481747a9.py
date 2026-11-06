"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin before 60s limit
    
    best_set = []
    
    def is_valid_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums."""
        # New triple sums involving x:
        # (x, x, x), (x, x, s_i), (x, s_i, s_j) where s_i, s_j in S
        # We need to check all new sums against existing sums_set and each other
        
        new_sums = []
        
        # (x, x, x)
        s_3x = 3 * x
        new_sums.append(s_3x)
        
        # (x, x, s_i) for each s_i in S
        for s_i in S:
            new_sums.append(2 * x + s_i)
        
        # (x, s_i, s_j) for each pair (s_i, s_j) in S with i <= j
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.append(x + S[i] + S[j])
        
        # Check for duplicates among new sums
        if len(new_sums) != len(set(new_sums)):
            return False
        
        # Check if any new sum is in existing sums_set
        new_sums_set = set(new_sums)
        if new_sums_set & sums_set:
            return False
        
        return True
    
    def compute_all_sums(S):
        """Compute set of all triple sums for S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    sums.add(S[i] + S[j] + S[k])
        return sums
    
    def greedy_search(order, time_limit):
        """Greedy search with given element order."""
        selected = []
        sums_set = set()
        
        for x in order:
            if time.time() > time_limit:
                break
            if is_valid_addition(selected, x, sums_set):
                selected.append(x)
                # Update sums_set with new sums
                # New sums are those involving x
                # (x,x,x), (x,x,s_i), (x,s_i,s_j)
                sums_set.add(3 * x)
                for s_i in selected[:-1]:
                    sums_set.add(2 * x + s_i)
                    for s_j in selected[:-1]:
                        if s_j >= s_i:
                            sums_set.add(x + s_i + s_j)
                        elif s_i > s_j:
                            # Already added when s_i was the larger one? No, we need to be careful
                            # Actually, let's just recompute or add all pairs properly
                            pass
        
        # Let me redo the sums_set update more carefully
        # Actually, let's just recompute for correctness, it's O(n^3) but n is small
        if len(selected) > 0:
            sums_set = compute_all_sums(selected)
        
        return selected, sums_set
    
    # Strategy 1: Greedy with natural order (1, 2, 3, ...)
    order1 = list(range(1, 4097))
    set1, _ = greedy_search(order1, deadline)
    if len(set1) > len(best_set):
        best_set = set1
    
    if time.time() > deadline - 5:
        return best_set
    
    # Strategy 2: Greedy with shuffled orders
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        # Try different ranges and orders
        # Sometimes starting with larger numbers helps
        if attempt % 3 == 0:
            # Natural order but skip some small numbers
            order = list(range(1, 4097))
            random.shuffle(order)
        elif attempt % 3 == 1:
            # Prefer larger numbers first
            order = list(range(4096, 0, -1))
            random.shuffle(order)
        else:
            # Mix of small and large
            small = list(range(1, 200))
            large = list(range(3897, 4097))
            order = small + large
            random.shuffle(order)
        
        set_try, _ = greedy_search(order, deadline - 5)
        if len(set_try) > len(best_set):
            best_set = set_try
    
    # Strategy 3: Local improvement - try to add more elements
    if len(best_set) > 0 and time.time() < deadline - 3:
        current = list(best_set)
        sums_set = compute_all_sums(current)
        
        # Try adding elements from 1 to 4096 that weren't tried
        current_set = set(current)
        for x in range(1, 4097):
            if time.time() > deadline - 3:
                break
            if x in current_set:
                continue
            if is_valid_addition(current, x, sums_set):
                current.append(x)
                current_set.add(x)
                sums_set = compute_all_sums(current)
        
        if len(current) > len(best_set):
            best_set = current
    
    return best_set
# EVOLVE-BLOCK-END
