import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 85.0  # 90s wall-clock limit, leave 5s buffer

    # Precompute conflict bitmasks for each x
    # For each x, store list of 3-bit bitmasks (other elements in 4-APs with x)
    # Bit i represents element i+1 (so element 1 is bit 0, element 512 is bit 511)
    conflict = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        masks = []
        # Case 1: x is first: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a, b, c = x + d, x + 2*d, x + 3*d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.append(mask)
        
        # Case 2: x is second: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a, b, c = x - d, x + d, x + 2*d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.append(mask)
        
        # Case 3: x is third: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a, b, c = x - 2*d, x - d, x + d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.append(mask)
        
        # Case 4: x is fourth: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            a, b, c = x - 3*d, x - 2*d, x - d
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.append(mask)
        
        conflict[x] = masks
    
    def can_add(S_bits, x):
        """Check if x can be added to set represented by S_bits without creating a 4-AP."""
        for mask in conflict[x]:
            if (S_bits & mask) == mask:
                return False
        return True
    
    def greedy_fill(S_bits, order):
        """Greedily add elements in given order to S_bits."""
        for x in order:
            if (S_bits >> (x-1)) & 1:
                continue  # already in set
            if can_add(S_bits, x):
                S_bits |= (1 << (x-1))
        return S_bits
    
    def bits_to_set(S_bits):
        return [i+1 for i in range(n) if (S_bits >> i) & 1]
    
    random.seed(42)
    
    best_bits = 0
    best_size = 0
    
    # Phase 1: Many randomized greedy constructions
    all_elements = list(range(1, n + 1))
    
    # Deterministic orders
    for order in [all_elements, list(reversed(all_elements))]:
        S = greedy_fill(0, order)
        sz = bin(S).count('1')
        if sz > best_size:
            best_bits = S
            best_size = sz
    
    # Random orders - run as many as possible
    num_random = 0
    while time.time() - start_time < time_limit * 0.25:
        order = all_elements[:]
        random.shuffle(order)
        S = greedy_fill(0, order)
        sz = bin(S).count('1')
        if sz > best_size:
            best_bits = S
            best_size = sz
        num_random += 1
        if num_random % 50 == 0:
            pass  # progress check
    
    # Phase 2: Local search / hill climbing
    current_bits = best_bits
    current_size = best_size
    
    # Precompute list of elements for quick access
    def try_add_any(S_bits):
        """Try to add any element not in S_bits. Returns (new_S_bits, added_x) or (S_bits, -1)."""
        for x in range(1, n + 1):
            if (S_bits >> (x-1)) & 1:
                continue
            if can_add(S_bits, x):
                return S_bits | (1 << (x-1)), x
        return S_bits, -1
    
    # Randomized local search with kicks
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() - start_time < time_limit:
        # Try to add elements
        improved = True
        while improved:
            S, x = try_add_any(current_bits)
            if x != -1:
                current_bits = S
                current_size += 1
                improved = True
            else:
                improved = False
        
        if current_size > best_size:
            best_bits = current_bits
            best_size = current_size
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            # Do a kick: remove 1-3 random elements, then greedily refill
            elements_in_set = bits_to_set(current_bits)
            if len(elements_in_set) > 3:
                k = random.randint(1, 3)
                to_remove = random.sample(elements_in_set, k)
                for x in to_remove:
                    current_bits &= ~(1 << (x-1))
                current_size -= k
                
                # Greedily refill with random order
                order = all_elements[:]
                random.shuffle(order)
                current_bits = greedy_fill(current_bits, order)
                current_size = bin(current_bits).count('1')
                
                if current_size > best_size:
                    best_bits = current_bits
                    best_size = current_size
                    no_improve_count = 0
                else:
                    no_improve_count = 1  # reset to small value, give it more chances
            
            # If we've been stuck for a long time, try a fresh random restart
            if no_improve_count > 500:
                order = all_elements[:]
                random.shuffle(order)
                current_bits = greedy_fill(0, order)
                current_size = bin(current_bits).count('1')
                if current_size > best_size:
                    best_bits = current_bits
                    best_size = current_size
                no_improve_count = 0
        else:
            no_improve_count = 0
    
    return bits_to_set(best_bits)
