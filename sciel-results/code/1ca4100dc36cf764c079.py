import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflict mask for all pairs
    # mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # Helper to check compatibility with current mask
    def is_compatible(x, current_mask):
        return (conflict_mask[x] & current_mask) == 0
        
    def count_bits(mask):
        return bin(mask).count('1')
            
    # Conflict counts for each number
    conflict_counts = [0] * (n + 1)
    for i in range(1, n + 1):
        conflict_counts[i] = count_bits(conflict_mask[i])
    
    # Sort numbers by conflict count (fewer conflicts = safer to add)
    order_by_conflict = list(range(1, n + 1))
    order_by_conflict.sort(key=lambda x: conflict_counts[x])
    
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy from empty, sorted by conflict count
    current_mask = 0
    for x in order_by_conflict:
        if is_compatible(x, current_mask):
            current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask
    
    # Strategy 2: Greedy from primes+1 start (original)
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
                
    initial_mask = 0
    for i in range(1, n + 1):
        if is_prime[i] or i == 1:
            initial_mask |= (1 << i)
            
    current_mask = initial_mask
    for x in order_by_conflict:
        if not (current_mask >> x) & 1:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
    sz = count_bits(current_mask)
    if sz > best_size:
        best_size = sz
        best_mask = current_mask
    
    # Multiple randomized greedy restarts from empty set
    rng = random.Random(42)
    remaining_time = deadline - time.time()
    num_restarts = max(1, int(remaining_time / 0.3))  # faster restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Random perturbation of order: shuffle with some bias toward low conflict
        order = list(range(1, n + 1))
        rng.shuffle(order)
        # Sort by (conflict_count + random_noise) to introduce diversity
        order.sort(key=lambda x: conflict_counts[x] + rng.random() * 5)
        
        current_mask = 0
        for x in order:
            if is_compatible(x, current_mask):
                current_mask |= (1 << x)
        sz = count_bits(current_mask)
        if sz > best_size:
            best_size = sz
            best_mask = current_mask
    
    # Local search: add-only pass
    if time.time() < deadline - 8:
        improved = True
        while improved and time.time() < deadline - 6:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if is_compatible(x, best_mask):
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True
    
    # 1-swap local search: try removing y and adding x
    if time.time() < deadline - 4:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, n + 1):
                if (best_mask >> x) & 1:
                    continue
                # x is not in set; check if we can add it by removing one element
                # Find all elements in best_mask that conflict with x
                conflicts_with_x = conflict_mask[x] & best_mask
                if conflicts_with_x == 0:
                    # Can add directly
                    best_mask |= (1 << x)
                    best_size += 1
                    improved = True
                    break
                # Try removing one conflicting element y and adding x
                c = conflicts_with_x
                # Get the list of conflicting elements
                conflicting = []
                while c:
                    lb = c & (-c)
                    y = lb.bit_length() - 1
                    conflicting.append(y)
                    c ^= lb
                
                # If exactly one conflict, we can swap x in for y (size-preserving)
                # Accept if it reduces total conflict count (improves future flexibility)
                if len(conflicting) == 1:
                    y = conflicting[0]
                    new_mask = (best_mask & ~(1 << y)) | (1 << x)
                    if conflict_counts[x] < conflict_counts[y]:
                        best_mask = new_mask
                        improved = True
                        break
                # If multiple conflicts, try removing the "most useful" conflicting element
                # and see if x becomes compatible (this is a 1-remove-1-add that may not be valid)
                # Actually, if x conflicts with multiple elements, we can't just add x.
                # So only the len==1 case is a valid 1-swap.
    
    # 2-swap local search: remove 2 elements, add 1 (net -1) or remove 1, add 2 (net +1)
    # Focus on remove-1-add-2 to increase size
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            # Try to find x,y not in set such that removing z from set allows adding both x and y
            for z in range(1, n + 1):
                if not (best_mask >> z) & 1:
                    continue
                # Temporarily remove z
                temp_mask = best_mask & ~(1 << z)
                # Find candidates x not in set that are compatible with temp_mask
                candidates = []
                for x in range(1, n + 1):
                    if (best_mask >> x) & 1:
                        continue
                    if is_compatible(x, temp_mask):
                        candidates.append(x)
                
                # Try to find two compatible candidates x, y from candidates
                for i in range(len(candidates)):
                    x = candidates[i]
                    # Check if x is compatible with temp_mask
                    if not is_compatible(x, temp_mask):
                        continue
                    for j in range(i + 1, len(candidates)):
                        y = candidates[j]
                        if not is_compatible(y, temp_mask):
                            continue
                        if not is_compatible(x, y):
                            continue
                        # Both x and y are compatible with temp_mask and with each other
                        # Adding both and removing z gives net +1
                        new_mask = temp_mask | (1 << x) | (1 << y)
                        best_mask = new_mask
                        best_size += 1
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            
            # 2-swap: remove 2 elements, add 2 elements (size-preserving) to improve conflict count
            if not improved and time.time() < deadline - 0.5:
                for z1 in range(1, n + 1):
                    if not (best_mask >> z1) & 1:
                        continue
                    for z2 in range(z1 + 1, n + 1):
                        if not (best_mask >> z2) & 1:
                            continue
                        temp_mask = best_mask & ~(1 << z1) & ~(1 << z2)
                        # Find candidates not in set that are compatible with temp_mask
                        candidates = []
                        for x in range(1, n + 1):
                            if (best_mask >> x) & 1:
                                continue
                            if is_compatible(x, temp_mask):
                                candidates.append(x)
                        # Try to find two compatible candidates
                        for i in range(len(candidates)):
                            x = candidates[i]
                            for j in range(i + 1, len(candidates)):
                                y = candidates[j]
                                if not is_compatible(y, temp_mask):
                                    continue
                                if not is_compatible(x, y):
                                    continue
                                new_mask = temp_mask | (1 << x) | (1 << y)
                                # Accept if total conflict count decreases
                                old_conf = conflict_counts[z1] + conflict_counts[z2]
                                new_conf = conflict_counts[x] + conflict_counts[y]
                                if new_conf < old_conf:
                                    best_mask = new_mask
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    # Final add-only pass
    if time.time() < deadline - 0.3:
        for x in range(1, n + 1):
            if not (best_mask >> x) & 1:
                if is_compatible(x, best_mask):
                    best_mask |= (1 << x)
                    best_size += 1
    
    # Convert back to list
    best = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return best
