import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 60s

    N = 256

    # Precompute perfect squares up to 2*N = 512
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1

    # Build adjacency masks
    # adj_mask[i] is an integer bitmask where bit j is 1 if (i+1) + (j+1) is a square
    adj_mask = [0] * N
    
    # Identify vertices with self-loops (2*v is a square)
    forbidden = set()
    
    for i in range(N):
        v = i + 1
        if 2 * v in squares:
            forbidden.add(i)
        
        for sq in squares:
            u = sq - v
            if 1 <= u <= N:
                j = u - 1
                if j != i:
                    adj_mask[i] |= (1 << j)

    # Candidates are vertices without self-loops
    candidates = [i for i in range(N) if i not in forbidden]
    
    # Precompute degrees for sorting
    degrees = [bin(adj_mask[i]).count('1') for i in range(N)]

    def get_size(mask):
        return bin(mask).count('1')

    def greedy_solve(order):
        """
        Greedy construction with a given order.
        """
        current_mask = 0
        
        for v in order:
            if v in forbidden:
                continue
            # Check if v conflicts with any u in current_mask
            if not (adj_mask[v] & current_mask):
                current_mask |= (1 << v)
                
        return current_mask

    def local_improve(mask):
        """
        Add vertices that don't conflict.
        """
        improved = True
        while improved:
            improved = False
            for v in candidates:
                if (mask >> v) & 1:
                    continue
                if not (adj_mask[v] & mask):
                    mask |= (1 << v)
                    improved = True
        return mask

    def local_search(mask, max_iterations=100):
        """
        Local search with 1-removal and 2-removal swaps.
        """
        mask = local_improve(mask)
        best_size = get_size(mask)
        best_mask = mask
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iterations and (time.time() - start_time < DEADLINE - 2):
            improved = False
            iter_count += 1
            
            # 1-Removal
            vertices_to_try = [v for v in range(N) if (mask >> v) & 1]
            random.shuffle(vertices_to_try)
            
            for v in vertices_to_try:
                if time.time() - start_time > DEADLINE - 2:
                    return best_mask
                
                new_mask = mask & ~(1 << v)
                new_mask = local_improve(new_mask)
                
                if get_size(new_mask) > best_size:
                    best_size = get_size(new_mask)
                    best_mask = new_mask
                    mask = new_mask
                    improved = True
                    break # Restart outer loop with new mask
            
            if improved:
                continue

            # 2-Removal (if time permits and 1-removal didn't improve)
            if len(vertices_to_try) >= 2:
                verts_in_set = [v for v in range(N) if (mask >> v) & 1]
                random.shuffle(verts_in_set)
                
                # Try a limited number of pairs
                pairs_tried = 0
                max_pairs = 50 # Limit per iteration to keep time low
                for i in range(len(verts_in_set)):
                    if pairs_tried >= max_pairs: break
                    for j in range(i+1, len(verts_in_set)):
                        if pairs_tried >= max_pairs: break
                        pairs_tried += 1
                        v1 = verts_in_set[i]
                        v2 = verts_in_set[j]
                        
                        new_mask = mask & ~(1 << v1) & ~(1 << v2)
                        new_mask = local_improve(new_mask)
                        
                        if get_size(new_mask) > best_size:
                            best_size = get_size(new_mask)
                            best_mask = new_mask
                            mask = new_mask
                            improved = True
                            break
                    if improved:
                        break

        return best_mask

    best_size = 0
    best_mask = 0
    
    # Modular Initialization Strategies
    strategies = []
    
    # Global Low Degree
    strategies.append(sorted(candidates, key=lambda v: degrees[v]))
    # Global High Degree
    strategies.append(sorted(candidates, key=lambda v: -degrees[v]))
    
    # Modular Classes
    for r in range(4):
        class_verts = [v for v in candidates if (v+1) % 4 == r]
        strategies.append(sorted(class_verts, key=lambda v: degrees[v]))
        strategies.append(sorted(class_verts, key=lambda v: -degrees[v]))
        
    # Random
    strategies.append(candidates[:])
    strategies.append(candidates[:])
    strategies.append(candidates[:])

    for i in range(20):
        if time.time() - start_time > DEADLINE - 5:
            break
        strat_idx = i % len(strategies)
        order = strategies[strat_idx][:]
        if i >= 5:
             random.shuffle(order)
        else:
             # Add slight randomization to deterministic strategies to escape symmetries
             if i > 0:
                 for _ in range(3):
                     if len(order) > 1:
                         idx1 = random.randint(0, len(order)-1)
                         idx2 = random.randint(0, len(order)-1)
                         order[idx1], order[idx2] = order[idx2], order[idx1]
        
        s_mask = greedy_solve(order)
        s_mask = local_improve(s_mask)
        
        if get_size(s_mask) > best_size:
            best_size = get_size(s_mask)
            best_mask = s_mask
            
    # Main Loop: Iterated Local Search
    while time.time() - start_time < DEADLINE - 1:
        
        # Choose strategy
        if random.random() < 0.5:
            # Restart from scratch
            order = candidates[:]
            random.shuffle(order)
            s_mask = greedy_solve(order)
            s_mask = local_improve(s_mask)
        else:
            # Perturb current best
            s_mask = best_mask
            # Remove some random elements
            verts_in_set = [v for v in range(N) if (s_mask >> v) & 1]
            random.shuffle(verts_in_set)
            remove_count = max(1, len(verts_in_set) // 10)
            for v in verts_in_set[:remove_count]:
                s_mask &= ~(1 << v)
            s_mask = local_improve(s_mask)
        
        # Local Search
        if time.time() - start_time < DEADLINE - 3:
            s_mask = local_search(s_mask, max_iterations=50)
        
        if get_size(s_mask) > best_size:
            best_size = get_size(s_mask)
            best_mask = s_mask

    # Convert mask back to list of 1-based values
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return sorted(result)

# EVOLVE-BLOCK-END
