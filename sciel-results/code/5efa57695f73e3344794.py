import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin
    n = 256
    
    # 1. Precompute conflict masks
    # mask[i] has bit j set if i and j conflict
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
                
    # 2. Calculate degrees and sort vertices
    degrees = [0] * (n + 1)
    for i in range(1, n + 1):
        degrees[i] = bin(conflict_mask[i]).count('1')
        
    # Sort indices 1..n by degree descending
    order = list(range(1, n + 1))
    order.sort(key=lambda x: degrees[x], reverse=True)
    
    # 3. Initial Greedy Solution (Lower Bound)
    def greedy_solve(start_mask=0, order=order):
        current_mask = start_mask
        for v in order:
            if (current_mask >> v) & 1:
                continue
            if (conflict_mask[v] & current_mask) == 0:
                current_mask |= (1 << v)
        return current_mask

    best_mask = greedy_solve()
    best_size = bin(best_mask).count('1')
    
    # Try a few random greedy starts to improve lower bound
    rng = random.Random(42)
    for _ in range(20):
        rand_order = list(range(1, n + 1))
        rng.shuffle(rand_order)
        # Bias towards low conflict? Or just shuffle?
        # Shuffling creates diverse initial sets
        m = greedy_solve(0, rand_order)
        sz = bin(m).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = m
            
    # 4. Branch and Bound
    # We search for a set of size > best_size
    # available: bitmask of vertices that can still be considered
    # current: bitmask of vertices currently selected
    # count: number of vertices currently selected
    
    # To speed up, we can prune if count + popcount(available) <= best_size
    
    # We need a function to get the "best" vertex to branch on.
    # Heuristic: Pick vertex in available with highest degree in the subgraph induced by available.
    
    def popcount(x):
        return bin(x).count('1')
        
    def branch_and_bound(available, current, count):
        nonlocal best_size, best_mask
        
        # Pruning: Upper bound on size
        if count + popcount(available) <= best_size:
            return
            
        if available == 0:
            if count > best_size:
                best_size = count
                best_mask = current
            return
        
        # Select pivot vertex
        # Iterate through available bits to find the one with max degree in available subgraph
        # This is O(N) per node, but N=256 is small.
        temp_avail = available
        pivot = 0
        max_deg = -1
        
        # Optimization: We can just pick the first one in 'order' that is in available?
        # No, dynamic degree is better for pruning.
        # Let's try a compromise: Pick the vertex from 'order' (global degree sort) that is in available.
        # This is O(1) if we precompute, but 'order' is fixed.
        # Actually, let's just pick the highest degree vertex in the *global* order that is available.
        for v in order:
            if (available >> v) & 1:
                pivot = v
                break
                
        # Alternative: Pick pivot with max (conflict_mask[v] & available).bit_count()
        # Let's stick to global order for speed, it's usually sufficient for N=256.
        
        if pivot == 0:
            return

        # Branch 1: Include pivot
        # New available = available without pivot and without its neighbors
        new_available_include = available & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound(new_available_include, current | (1 << pivot), count + 1)
        
        # Check time
        if time.time() > deadline:
            return

        # Branch 2: Exclude pivot
        new_available_exclude = available & ~(1 << pivot)
        branch_and_bound(new_available_exclude, current, count)
        
    # Start B&B
    # Initial available is all vertices
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    branch_and_bound(all_vertices_mask, 0, 0)
    
    # 5. Final Local Search Polish (in case B&B stopped early or to fix any issues)
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any vertex
            for x in range(1, n + 1):
                if not (best_mask >> x) & 1:
                    if (conflict_mask[x] & best_mask) == 0:
                        best_mask |= (1 << x)
                        best_size += 1
                        improved = True
            # Try 1-swap (remove y, add x) if it increases size? 
            # Actually, 1-swap keeps size same. We want to increase size.
            # The "add" pass above handles size increase.
            # If no vertex can be added, we are locally optimal w.r.t additions.
            # To increase size further, we might need to remove 1 and add 2?
            # That's complex. The B&B should have found the global max if it finished.
            # If it didn't finish, the current best is a lower bound.
            
    # Convert mask to list
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    
    return result
