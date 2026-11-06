import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = [i*i for i in range(2, 23)]
    
    forbidden = set()
    for s in squares:
        if s % 2 == 0:
            v = s // 2
            if 1 <= v <= N:
                forbidden.add(v)
                
    valid_nodes = [i for i in range(1, N+1) if i not in forbidden]
    node_to_idx = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    adj_bits = [0] * M
    
    for i, u in enumerate(valid_nodes):
        for v in valid_nodes:
            if u >= v: continue
            if u + v in squares:
                j = node_to_idx[v]
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    def greedy_mis_bits():
        remaining_mask = (1 << M) - 1
        chosen = []
        while remaining_mask:
            # Find min degree in remaining
            # This is expensive if done naively.
            # We can just pick a random node or first set bit?
            # Let's stick to min degree for quality, but calculate degree quickly?
            # Degree of u in remaining: (adj_bits[u] & remaining_mask).bit_count()
            
            min_deg = M
            best_u = -1
            # Iterate through set bits in remaining_mask
            r = remaining_mask
            while r:
                # Get lowest set bit
                lsb = r & -r
                u = lsb.bit_length() - 1
                deg = (adj_bits[u] & remaining_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_u = u
                r ^= lsb
                
            if best_u == -1: break
            
            chosen.append(best_u)
            # Remove best_u and neighbors
            to_remove = (1 << best_u) | adj_bits[best_u]
            remaining_mask &= ~to_remove
            
        return chosen

    def local_search_bits(S_mask):
        changed = True
        while changed:
            changed = False
            # Iterate through all u
            for u in range(M):
                if (S_mask >> u) & 1:
                    continue
                conflicts_mask = adj_bits[u] & S_mask
                conflict_count = conflicts_mask.bit_count()
                if conflict_count == 0:
                    S_mask |= (1 << u)
                    changed = True
                elif conflict_count == 1:
                    # Remove the one conflict
                    conflict_bit = conflicts_mask & -conflicts_mask
                    S_mask &= ~conflict_bit
                    S_mask |= (1 << u)
                    changed = True
        return S_mask

    best_mask = greedy_mis_bits()
    best_mask = local_search_bits(best_mask)
    
    # Randomized restarts
    # We need a randomized greedy
    def randomized_greedy_bits():
        remaining_mask = (1 << M) - 1
        chosen = []
        while remaining_mask:
            # Pick random node from remaining? 
            # Or min degree with random tie-breaking?
            # Let's do min degree with random tie-breaking.
            min_deg = M
            candidates = []
            r = remaining_mask
            while r:
                lsb = r & -r
                u = lsb.bit_length() - 1
                deg = (adj_bits[u] & remaining_mask).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    candidates = [u]
                elif deg == min_deg:
                    candidates.append(u)
                r ^= lsb
            
            if not candidates: break
            best_u = random.choice(candidates)
            chosen.append(best_u)
            to_remove = (1 << best_u) | adj_bits[best_u]
            remaining_mask &= ~to_remove
            
        return chosen

    for _ in range(2000):
        if time.time() > deadline:
            break
        mask = randomized_greedy_bits()
        mask = local_search_bits(mask)
        if mask.bit_count() > best_mask.bit_count():
            best_mask = mask

    result = [valid_nodes[i] for i in range(M) if (best_mask >> i) & 1]
    result.sort()
    return result
