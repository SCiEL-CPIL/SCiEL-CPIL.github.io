import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 58.0  # Safety margin before 90s kill
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict sets using bitmasks for speed
    # conflict[u] is a bitmask of vertices v such that (u+1)+(v+1) is a square
    # We use 0-indexed internally
    conflict = [0] * N
    valid_mask = 0
    
    for i in range(N):
        a = i + 1
        # Check self conflict
        if 2 * a in squares:
            continue
        
        valid_mask |= (1 << i)
        
        for s in squares:
            b = s - a
            if 1 <= b <= N:
                j = b - 1
                conflict[i] |= (1 << j)
    
    valid_vertices = [i for i in range(N) if (valid_mask >> i) & 1]
    
    # Convert conflict to a list of lists for faster iteration in some contexts
    conflict_lists = []
    for i in range(N):
        lst = []
        mask = conflict[i]
        while mask:
            low_bit = mask & (-mask)
            idx = low_bit.bit_length() - 1
            lst.append(idx)
            mask ^= low_bit
        conflict_lists.append(lst)
    
    def is_compatible(u, S_set):
        """Check if u can be added to S_set using bitmask."""
        # Check all v in S_set: is v in conflict[u]?
        # S_set is a set of 0-indexed vertices
        cmask = conflict[u]
        for v in S_set:
            if (cmask >> v) & 1:
                return False
        return True
    
    def greedy_construct(order):
        S = set()
        S_mask = 0
        for u in order:
            if not (S_mask & conflict[u]):
                S.add(u)
                S_mask |= (1 << u)
        return S, S_mask
    
    def local_search_add(S, S_mask):
        """Try to add any compatible vertex repeatedly."""
        improved = True
        while improved:
            improved = False
            for u in valid_vertices:
                if (S_mask >> u) & 1:
                    continue
                if not (S_mask & conflict[u]):
                    S.add(u)
                    S_mask |= (1 << u)
                    improved = True
        return S, S_mask
    
    def local_search_swap(S, S_mask, max_swaps=100):
        """Try swap moves: remove one vertex, add multiple."""
        improved = True
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            for u in list(S):
                # Try removing u
                new_S_mask = S_mask & ~(1 << u)
                # Find all vertices that can be added
                addable = []
                for v in valid_vertices:
                    if (new_S_mask >> v) & 1:
                        continue
                    if not (new_S_mask & conflict[v]):
                        addable.append(v)
                
                if len(addable) >= 2:
                    # Add all addable vertices
                    for v in addable:
                        new_S_mask |= (1 << v)
                    # Update S
                    S.discard(u)
                    for v in addable:
                        S.add(v)
                    S_mask = new_S_mask
                    improved = True
                    swap_count += 1
                    break  # Restart the loop after a swap
        
        return S, S_mask
    
    best_S = set()
    best_S_mask = 0
    
    # Strategy 1: Greedy by degree (ascending)
    degrees = {}
    for u in valid_vertices:
        # Count conflicts with valid vertices
        deg = 0
        cmask = conflict[u]
        vmask = valid_mask
        m = cmask & vmask
        while m:
            low_bit = m & (-m)
            m ^= low_bit
            deg += 1
        degrees[u] = deg
    
    order_by_degree = sorted(valid_vertices, key=lambda x: degrees[x])
    S1, M1 = greedy_construct(order_by_degree)
    S1, M1 = local_search_add(S1, M1)
    S1, M1 = local_search_swap(S1, M1)
    if len(S1) > len(best_S):
        best_S = S1
        best_S_mask = M1
    
    # Strategy 2: Greedy by degree (descending) - sometimes helps
    order_by_degree_desc = sorted(valid_vertices, key=lambda x: -degrees[x])
    S2, M2 = greedy_construct(order_by_degree_desc)
    S2, M2 = local_search_add(S2, M2)
    S2, M2 = local_search_swap(S2, M2)
    if len(S2) > len(best_S):
        best_S = S2
        best_S_mask = M2
    
    # Random restarts with diverse strategies
    while time.time() < start_time + DEADLINE:
        # Random shuffle
        random_order = valid_vertices[:]
        random.shuffle(random_order)
        
        current_S, current_M = greedy_construct(random_order)
        current_S, current_M = local_search_add(current_S, current_M)
        current_S, current_M = local_search_swap(current_S, current_M)
        
        if len(current_S) > len(best_S):
            best_S = current_S
            best_S_mask = current_M
        
        # Also try a biased random order: mix of low-degree and random
        if time.time() < start_time + DEADLINE:
            # Weighted random: favor low degree
            weights = [1.0 / (degrees[u] + 1) for u in valid_vertices]
            weighted_order = []
            remaining = valid_vertices[:]
            w_remaining = weights[:]
            for _ in range(len(remaining)):
                total_w = sum(w_remaining)
                r = random.random() * total_w
                cum = 0
                for idx, w in enumerate(w_remaining):
                    cum += w
                    if cum >= r:
                        weighted_order.append(remaining[idx])
                        remaining.pop(idx)
                        w_remaining.pop(idx)
                        break
            current_S2, current_M2 = greedy_construct(weighted_order)
            current_S2, current_M2 = local_search_add(current_S2, current_M2)
            current_S2, current_M2 = local_search_swap(current_S2, current_M2)
            if len(current_S2) > len(best_S):
                best_S = current_S2
                best_S_mask = current_M2
    
    # Convert back to 1-indexed list
    result = [i + 1 for i in best_S]
    # Ensure non-empty
    if not result:
        if valid_vertices:
            return [valid_vertices[0] + 1]
        else:
            return [1]
            
    return result
