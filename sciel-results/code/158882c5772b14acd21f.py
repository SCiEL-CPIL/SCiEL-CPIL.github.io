import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    best_set = []
    mask = (1 << 201) - 1 # Bits 0 to 200
    
    orders = []
    orders.append(list(range(100, 0, -1)))
    orders.append(list(range(1, 101)))
    
    base = list(range(1, 101))
    for _ in range(80):
        random.shuffle(base)
        orders.append(base[:])
        
    orders.append(sorted(range(1, 101), key=lambda x: x % 2))
    orders.append(sorted(range(1, 101), key=lambda x: x % 3))
    orders.append(sorted(range(1, 101), key=lambda x: x % 5))
    orders.append(sorted(range(1, 101), key=lambda x: x % 7))
    orders.append(sorted(range(1, 101), key=lambda x: x % 11))
    orders.append(sorted(range(1, 101), key=lambda x: -x % 2))
    orders.append(sorted(range(1, 101), key=lambda x: -x % 3))
    
    for order in orders:
        if time.time() > deadline:
            break
            
        selected = []
        reachable_bits = 1 # Bit 0 is set
        
        for x in order:
            if (reachable_bits >> (200 - x)) & 1:
                continue
            else:
                selected.append(x)
                new_bits = (reachable_bits << x) & mask
                reachable_bits |= new_bits
                
        if len(selected) > len(best_set):
            best_set = selected
            
    # Local search: try to improve best_set by adding elements
    if time.time() < deadline - 5 and best_set:
        best_set = _local_add(best_set, deadline)
        
    return best_set

def _local_add(current_list, deadline):
    current_set = set(current_list)
    all_elements = set(range(1, 101))
    mask = (1 << 201) - 1
    
    def compute_reachable(s):
        bits = 1
        for x in s:
            bits |= (bits << x) & mask
        return bits
    
    changed = True
    max_rounds = 50
    for _ in range(max_rounds):
        if time.time() > deadline - 2:
            break
        if not changed:
            break
        changed = False
        reachable = compute_reachable(current_set)
        for x in sorted(all_elements - current_set):
            if time.time() > deadline - 2:
                break
            if not ((reachable >> (200 - x)) & 1):
                current_set.add(x)
                changed = True
                break
    
    return list(current_set)
