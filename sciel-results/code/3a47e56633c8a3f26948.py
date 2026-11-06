import time
import random

def construct():
    n = 512
    start_time = time.time()
    time_limit = 55.0

    # Precompute conflict triples
    # triples[x] = list of (a, b, c) such that {a,b,c,x} is a 4-AP
    triples = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            triples[x].append((x + d, x + 2 * d, x + 3 * d))
        
        # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            triples[x].append((x - d, x + d, x + 2 * d))
            
        # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            triples[x].append((x - 2 * d, x - d, x + d))
            
        # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            triples[x].append((x - 3 * d, x - 2 * d, x - d))

    def can_add(mask, x):
        for a, b, c in triples[x]:
            # Check if bits a, b, c are set in mask
            if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                return False
        return True

    def greedy_fill(mask, order):
        for x in order:
            if not ((mask >> x) & 1):
                if can_add(mask, x):
                    mask |= (1 << x)
        return mask

    # Initialization: Base-3 Salem-Spencer set
    # Numbers with base-3 digits in {0, 1}
    init_mask = 0
    # Generate all numbers < 3^6 with digits 0,1
    # 3^0=1, 3^1=3, ..., 3^5=243
    powers = [1, 3, 9, 27, 81, 243]
    for i in range(1, 1 << len(powers)):
        val = 0
        for j, p in enumerate(powers):
            if (i >> j) & 1:
                val += p
        if 1 <= val <= n:
            init_mask |= (1 << val)
            
    best_mask = init_mask
    best_size = bin(init_mask).count('1')
    
    random.seed(42)
    
    # Local Search
    current_mask = init_mask
    current_size = best_size
    
    while time.time() - start_time < time_limit:
        # 1. Try to add any element
        added = False
        # Randomize candidates to avoid bias
        candidates = [i for i in range(1, n + 1) if not ((current_mask >> i) & 1)]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size
                break
        
        if not added:
            # 2. Kick: Remove k random elements
            # Choose k based on current size or random
            k = random.randint(1, 5)
            if current_size > k:
                # Get list of elements in current set
                elements_in_set = [i for i in range(1, n + 1) if (current_mask >> i) & 1]
                to_remove = random.sample(elements_in_set, k)
                for x in to_remove:
                    current_mask &= ~(1 << x)
                    current_size -= 1
                
                # 3. Greedy Refill
                # Try to add all possible elements in random order
                candidates = [i for i in range(1, n + 1) if not ((current_mask >> i) & 1)]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add(current_mask, x):
                        current_mask |= (1 << x)
                        current_size += 1
                
                if current_size > best_size:
                    best_mask = current_mask
                    best_size = current_size

    # Convert mask to sorted list
    result = [i for i in range(1, n + 1) if (best_mask >> i) & 1]
    return result
