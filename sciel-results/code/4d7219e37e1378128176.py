import random
import sys

def has_independent_set_of_size_k(adj_bits, n, k):
    """
    Checks if there is an independent set of size k in the graph using bitsets.
    adj_bits[v] is a bitmask of neighbors of v.
    """
    all_mask = (1 << n) - 1
    
    def backtrack(current_size, candidates_mask):
        if current_size == k:
            return True
        
        # Count bits in candidates_mask
        if candidates_mask.bit_count() < k - current_size:
            return False
        
        if candidates_mask == 0:
            return False
            
        # Pick a vertex. 
        # Heuristic: Pick a vertex that appears in many potential independent sets?
        # Simple heuristic: Pick the first one.
        v_bit = candidates_mask & -candidates_mask
        v = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        new_candidates_with_v = candidates_mask & ~(adj_bits[v] | v_bit)
        if backtrack(current_size + 1, new_candidates_with_v):
            return True
            
        # Branch 2: Exclude v
        new_candidates_without_v = candidates_mask & ~v_bit
        if backtrack(current_size, new_candidates_without_v):
            return True
            
        return False

    return backtrack(0, all_mask)

def construct():
    random.seed(123)
    
    # Try n=39
    n = 39
    # Partition into 5 parts: 8, 8, 8, 8, 7
    part_sizes = [8, 8, 8, 8, 7]
    starts = [0]
    for s in part_sizes:
        starts.append(starts[-1] + s)
        
    for attempt in range(1000):
        adj = [[False] * n for _ in range(n)]
        edges = []
        
        p = 0.5
        # Add random edges between different parts
        for i in range(n):
            part_i = None
            for idx in range(5):
                if starts[idx] <= i < starts[idx+1]:
                    part_i = idx
                    break
            for j in range(i+1, n):
                part_j = None
                for idx in range(5):
                    if starts[idx] <= j < starts[idx+1]:
                        part_j = idx
                        break
                if part_i != part_j:
                    if random.random() < p:
                        adj[i][j] = True
                        adj[j][i] = True
                        edges.append([i, j])
        
        # Convert to bitsets
        adj_bits = [0] * n
        for u in range(n):
            mask = 0
            for v in range(n):
                if adj[u][v]:
                    mask |= (1 << v)
            adj_bits[u] = mask
            
        if not has_independent_set_of_size_k(adj_bits, n, 10):
            return {'n': n, 'edges': edges}
            
    # Fallback
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 3 == 1]
    return {'n': n, 'edges': edges}
