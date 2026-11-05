import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    all_mask = (1 << N) - 1
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is 1 if i and j+1 conflict
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i != j and (i * j) % (i + j) == 0:
                mask |= (1 << (j - 1))
        conflict_mask[i] = mask

    def bit_count(x):
        return x.bit_count()

    # Greedy builder with a given priority order (list of vertices 1..N)
    def greedy_build(order):
        sol_mask = 0
        rem = all_mask
        for v in order:
            bit = 1 << (v - 1)
            if not (rem & bit):
                continue
            # Check if v can be added: v must not conflict with any in sol_mask
            if conflict_mask[v] & sol_mask:
                continue
            sol_mask |= bit
            # Remove v and its neighbors from consideration
            rem &= ~bit
            rem &= ~conflict_mask[v]
        return sol_mask

    # Polish: add any vertex that doesn't conflict
    def polish(mask):
        size = bit_count(mask)
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if mask & bit:
                continue
            if conflict_mask[v] & mask:
                continue
            mask |= bit
            size += 1
        return mask, size

    # Local Search: 1-for-2 and 2-for-3 swaps
    def local_search(mask, size, time_limit=2.0):
        sol_list = [i + 1 for i in range(N) if mask & (1 << i)]
        
        # Helper to get addable vertices given a mask
        def get_addable(m):
            addable = []
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if m & bit:
                    continue
                if conflict_mask[v] & m:
                    continue
                addable.append(v)
            return addable

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-for-2 swap
            # Try removing each vertex in sol_list
            for v_out in sol_list:
                if time.time() > time_limit:
                    return mask, size
                new_mask = mask & ~(1 << (v_out - 1))
                addable = get_addable(new_mask)
                if len(addable) < 2:
                    continue
                
                # Try to find 2 independent vertices from addable
                found = False
                for i in range(len(addable)):
                    a = addable[i]
                    for j in range(i + 1, len(addable)):
                        b = addable[j]
                        # Check if a and b are independent
                        if not (conflict_mask[a] & (1 << (b - 1))):
                            new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1))
                            # Verify size increase
                            if bit_count(new_mask2) == size + 1:
                                mask = new_mask2
                                size = size + 1
                                sol_list = [i + 1 for i in range(N) if mask & (1 << i)]
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if found:
                    break # Restart loop with new mask

            # 2-for-3 swap
            if not improved and time.time() < time_limit:
                # Try removing pairs
                pairs = []
                for i in range(len(sol_list)):
                    for j in range(i + 1, len(sol_list)):
                        pairs.append((sol_list[i], sol_list[j]))
                
                random.shuffle(pairs)
                # Try a limited number of pairs
                for v1, v2 in pairs[:50]: # Limit to 50 pairs to keep it fast
                    if time.time() > time_limit:
                        return mask, size
                    new_mask = mask & ~(1 << (v1 - 1)) & ~(1 << (v2 - 1))
                    addable = get_addable(new_mask)
                    if len(addable) < 3:
                        continue
                    
                    # Try to find 3 independent vertices from addable
                    found = False
                    if len(addable) <= 20:
                        for i in range(len(addable)):
                            a = addable[i]
                            for j in range(i + 1, len(addable)):
                                b = addable[j]
                                if conflict_mask[a] & (1 << (b - 1)):
                                    continue
                                for k in range(j + 1, len(addable)):
                                    c = addable[k]
                                    if (conflict_mask[a] & (1 << (c - 1))) or (conflict_mask[b] & (1 << (c - 1))):
                                        continue
                                    new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                                    if bit_count(new_mask2) == size + 1:
                                        mask = new_mask2
                                        size = size + 1
                                        sol_list = [i + 1 for i in range(N) if mask & (1 << i)]
                                        improved = True
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                    else:
                        # Random sampling for larger addable sets
                        for _ in range(100):
                            sample = random.sample(addable, 3)
                            a, b, c = sample
                            if (conflict_mask[a] & (1 << (b - 1))) or (conflict_mask[a] & (1 << (c - 1))) or (conflict_mask[b] & (1 << (c - 1))):
                                continue
                            new_mask2 = new_mask | (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                            if bit_count(new_mask2) == size + 1:
                                mask = new_mask2
                                size = size + 1
                                sol_list = [i + 1 for i in range(N) if mask & (1 << i)]
                                improved = True
                                found = True
                                break
                    if found:
                        break

        return mask, size

    best_mask = 0
    best_size = 0

    # Generate diverse initial orders
    base_order = list(range(1, N + 1))
    
    # Calculate static degrees for initial heuristics
    deg = [0] * (N + 1)
    for i in range(1, N + 1):
        deg[i] = bit_count(conflict_mask[i])
    
    # Strategy 1: Min-degree greedy
    order_min_deg = sorted(base_order, key=lambda x: deg[x])
    
    # Strategy 2: Max-degree greedy
    order_max_deg = sorted(base_order, key=lambda x: -deg[x])
    
    # Strategy 3: Natural order
    order_natural = base_order
    
    # Strategy 4: Reverse natural order
    order_reverse = list(reversed(base_order))
    
    orders = [
        order_min_deg,
        order_max_deg,
        order_natural,
        order_reverse
    ]
    
    # Add some random orders
    for _ in range(10):
        rnd_order = base_order[:]
        random.shuffle(rnd_order)
        orders.append(rnd_order)
        
    # Run greedy + local search on each
    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy_build(order)
        mask, size = polish(mask)
        
        # Local search
        mask, size = local_search(mask, size, time_limit=2.0)
        
        if size > best_size:
            best_size = size
            best_mask = mask

    # If we have time, do more random restarts with local search
    while time.time() < deadline:
        rnd_order = base_order[:]
        # Bias towards low degree with random tie-breaking
        rnd_order.sort(key=lambda x: (deg[x], random.random()))
        
        mask = greedy_build(rnd_order)
        mask, size = polish(mask)
        
        # Short local search
        mask, size = local_search(mask, size, time_limit=1.0)
        
        if size > best_size:
            best_size = size
            best_mask = mask

    # Final polish
    best_mask, best_size = polish(best_mask)
    
    final_sol = [i + 1 for i in range(N) if best_mask & (1 << i)]
    if not final_sol:
        final_sol = [1]
        
    return final_sol
