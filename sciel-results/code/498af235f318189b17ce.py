import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin for 60s limit
    
    n = 512
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    ap_list = []
    conflicts = {i: [] for i in range(1, n + 1)}
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            w = a + 3 * d
            if w > n:
                break
            b = a + d
            c = a + 2 * d
            ap = (a, b, c, w)
            ap_list.append(ap)
            for term in ap:
                conflicts[term].append(ap)
    
    # Function to check if adding x to current_set creates a 4-AP
    def can_add(x, current_set):
        for ap in conflicts[x]:
            # Count how many of the other 3 terms are in current_set
            count = 0
            for term in ap:
                if term != x and term in current_set:
                    count += 1
            if count == 3:
                return False
        return True
    
    # Initialize with a 3-AP-free set (base-3 digits 0 and 2)
    # This is guaranteed 4-AP-free
    initial_set = set()
    for i in range(1, n + 1):
        x = i
        valid = True
        while x > 0:
            if x % 3 == 1:
                valid = False
                break
            x //= 3
        if valid:
            initial_set.add(i)
    
    best_set = set(initial_set)
    best_size = len(best_set)
    
    # Local search with swaps
    # We try to improve the set by adding elements or swapping
    # Strategy: 
    # 1. Try to add any element that doesn't conflict
    # 2. If stuck, try swaps: remove one element, add another
    
    current_set = set(initial_set)
    size = len(current_set)
    
    # Phase 1: Greedy addition
    while (time.time() - start_time) < deadline:
        # Find all elements that can be added
        candidates = [i for i in range(1, n + 1) if i not in current_set and can_add(i, current_set)]
        
        if not candidates:
            break
        
        # Add all possible candidates in this pass
        for x in candidates:
            current_set.add(x)
        
        size = len(current_set)
        if size > best_size:
            best_set = set(current_set)
            best_size = size
    
    # Phase 2: Local search with swaps
    # Try to improve by swapping elements
    # We use a simulated annealing-like approach or simple hill climbing
    
    # For speed, we'll do limited swaps
    # Track how many consecutive no-improvement steps
    no_improve_count = 0
    max_no_improve = 20000  # Limit to prevent infinite loops
    
    while (time.time() - start_time) < deadline and no_improve_count < max_no_improve:
        # Try to find a beneficial swap
        # Strategy: pick a random element to remove, then try to add a new one
        
        # To be efficient, we'll try random swaps
        # Pick a random element from current_set to remove
        if not current_set:
            break
        
        y = random.choice(list(current_set))
        
        # Remove y
        current_set.discard(y)
        
        # Try to add a new element
        # Find candidates that can be added now
        candidates = [i for i in range(1, n + 1) if i not in current_set and can_add(i, current_set)]
        
        if candidates:
            # Pick a random candidate (or the one that maximizes future potential?)
            # For simplicity, pick randomly
            x = random.choice(candidates)
            current_set.add(x)
            
            new_size = len(current_set)
            if new_size > best_size:
                best_set = set(current_set)
                best_size = new_size
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # No candidate to add, restore y
            current_set.add(y)
            no_improve_count += 1
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
