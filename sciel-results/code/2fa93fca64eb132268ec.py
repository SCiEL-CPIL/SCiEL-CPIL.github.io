import random

def construct():
    n = 512
    
    # Precompute conflict masks for each x
    # conflicts[x] is a list of integers (bitmasks)
    # If (selected & mask) == mask, x cannot be added.
    conflicts = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the 4th term: a, a+d, a+2d, x => a=x-3d
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[x].append(mask)
            
        # x is the 2nd term: a, x, x+d, x+2d => a=x-d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a = x - d
            b = x + d
            c = x + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[x].append(mask)
            
        # x is the 3rd term: a, a+d, x, x+d => a=x-2d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a = x - 2 * d
            b = a + d
            c = x + d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[x].append(mask)
            
        # x is the 1st term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            a = x + d
            b = x + 2 * d
            c = x + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflicts[x].append(mask)
            
    # Greedy function with a given order, using bitsets
    def greedy(order):
        selected = 0
        for x in order:
            can_add = True
            for mask in conflicts[x]:
                if (selected & mask) == mask:
                    can_add = False
                    break
            if can_add:
                selected |= (1 << (x - 1))
        return selected

    # Local search: try to remove one element and add others
    def local_search(selected):
        improved = True
        while improved:
            improved = False
            # Try removing each u in selected
            for u in range(1, n + 1):
                if not (selected & (1 << (u - 1))):
                    continue
                # Remove u
                new_selected = selected & ~(1 << (u - 1))
                # Try to add any v not in selected
                for v in range(1, n + 1):
                    if v == u: continue
                    if new_selected & (1 << (v - 1)): continue
                    # Check if v can be added to new_selected
                    can_add = True
                    for mask in conflicts[v]:
                        if (new_selected & mask) == mask:
                            can_add = False
                            break
                    if can_add:
                        new_selected |= (1 << (v - 1))
                
                # If size increased, accept
                if new_selected.bit_count() > selected.bit_count():
                    selected = new_selected
                    improved = True
                    break # Restart scan with new selected
        return selected

    best_set = 0
    best_size = 0
    
    # Run multiple random orders
    # With bitsets, this is faster. Let's do 3000 runs.
    for _ in range(3000):
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy(order)
        if s.bit_count() > best_size:
            best_size = s.bit_count()
            best_set = s
            
    # Also try deterministic orders
    s = greedy(list(range(1, n + 1)))
    if s.bit_count() > best_size:
        best_size = s.bit_count()
        best_set = s
        
    s = greedy(list(range(n, 0, -1)))
    if s.bit_count() > best_size:
        best_size = s.bit_count()
        best_set = s
        
    # Apply local search to the best set
    improved_set = local_search(best_set)
    if improved_set.bit_count() > best_size:
        best_set = improved_set
        best_size = improved_set.bit_count()
        
    # Convert bitset to sorted list
    result = []
    for i in range(1, n + 1):
        if best_set & (1 << (i - 1)):
            result.append(i)
            
    return result
