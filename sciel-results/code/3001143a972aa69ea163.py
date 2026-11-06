import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill
    
    TARGET = 200
    N = 100
    MASK = (1 << (TARGET + 1)) - 1
    
    def try_greedy(order):
        """Greedy subset selection: add x if it doesn't create sum TARGET."""
        bits = 1  # Bit 0 set, representing sum 0
        selected = []
        for x in order:
            # Check if adding x creates a subset sum of TARGET
            # 200 is formed if bit (200-x) is set in current reachable bits
            if (bits >> (TARGET - x)) & 1:
                continue
            # Safe to add x
            selected.append(x)
            new_bits = (bits << x) & MASK
            bits |= new_bits
        return selected
    
    # --- Phase 1: Proven base construction {66, ..., 100} ---
    # Proof: any subset of size >= 3 sums to >= 66+67+68 = 201 > 200.
    # Any subset of size 2 sums to at most 99+100 = 199 < 200.
    # Any single element is <= 100 < 200.
    base = list(range(66, 101))  # 35 elements
    
    # Compute reachable sums for the base set using bitset
    base_bits = 1
    for x in base:
        base_bits |= (base_bits << x) & MASK
    
    # --- Phase 2: Greedy additions from [1, 65] ---
    # Try to add elements that don't create sum 200
    # An element x can be added if bit (200-x) is NOT set in reachable bits
    result = base[:]
    reachable = base_bits
    
    # Try smallest first to maximize count
    for x in range(1, 66):
        if time.time() > deadline:
            break
        if (reachable >> (TARGET - x)) & 1:
            continue
        # Safe to add
        result.append(x)
        reachable |= (reachable << x) & MASK
    
    best_set = result
    best_size = len(best_set)
    
    # --- Phase 3: Random restarts with greedy ---
    # Try various orderings to see if we can do better
    orders = []
    orders.append(list(range(100, 0, -1)))
    orders.append(list(range(1, 101)))
    orders.append(sorted(range(1, 101), key=lambda x: x % 2))
    orders.append(sorted(range(1, 101), key=lambda x: x % 3))
    orders.append(sorted(range(1, 101), key=lambda x: x % 5))
    
    base_list = list(range(1, 101))
    for _ in range(30):
        shuffled = base_list[:]
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        selected = try_greedy(order)
        if len(selected) > best_size:
            best_set = selected
            best_size = len(selected)
    
    # --- Phase 4: More random restarts if time permits ---
    while time.time() < deadline:
        random_order = list(range(1, 101))
        random.shuffle(random_order)
        selected = try_greedy(random_order)
        if len(selected) > best_size:
            best_set = selected
            best_size = len(selected)
    
    # --- Phase 5: Local improvement - try swapping elements ---
    # Try to replace elements in best_set with smaller ones to free up sum space
    if time.time() < deadline:
        for _ in range(20):  # Limited improvement passes
            if time.time() > deadline:
                break
            
            # Compute reachable bits for best_set
            bits = 1
            for x in best_set:
                bits |= (bits << x) & MASK
            
            improved = False
            new_set = best_set[:]
            
            # Try removing each element and replacing with a smaller one
            for i, x in enumerate(new_set):
                if time.time() > deadline:
                    break
                
                # Temporarily remove x
                bits_without = 1
                for j, y in enumerate(new_set):
                    if j != i:
                        bits_without |= (bits_without << y) & MASK
                
                # Try adding smaller elements not in new_set
                for candidate in range(1, x):
                    if candidate in new_set:
                        continue
                    # Check if adding candidate creates sum 200
                    if (bits_without >> (TARGET - candidate)) & 1:
                        continue
                    # Safe to swap
                    new_set[i] = candidate
                    improved = True
                    break
                
                if improved:
                    break
            
            if improved:
                best_set = new_set
                best_size = len(best_set)
    
    return best_set
