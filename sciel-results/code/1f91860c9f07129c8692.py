import random
import time

def construct():
    n = 20
    # Initialize random graph
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def count_regular_6(adj):
        count = 0
        # Iterate all 6-subsets
        # Using combinations from itertools is slow, use bit tricks or nested loops?
        # Nested loops for 6 indices is fast enough for 38k subsets
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        for m in range(l+1, n):
                            for p in range(m+1, n):
                                mask = (1 << i) | (1 << j) | (1 << k) | (1 << l) | (1 << m) | (1 << p)
                                # Check regularity
                                # Degrees in induced subgraph
                                d_i = (adj[i] & mask).bit_count()
                                d_j = (adj[j] & mask).bit_count()
                                d_k = (adj[k] & mask).bit_count()
                                d_l = (adj[l] & mask).bit_count()
                                d_m = (adj[m] & mask).bit_count()
                                d_p = (adj[p] & mask).bit_count()
                                if d_i == d_j == d_k == d_l == d_m == d_p:
                                    count += 1
        return count

    def is_valid_full(adj):
        # Check all subsets of size >= 6
        # This is expensive, only call when cost6 is 0
        for size in range(6, n + 1):
            # Generate all subsets of this size
            # Using itertools.combinations
            from itertools import combinations
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                # Check regularity
                first_deg = -1
                is_reg = True
                for v in combo:
                    d = (adj[v] & mask).bit_count()
                    if first_deg == -1:
                        first_deg = d
                    elif d != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if (mask >> j) & 1:
                    edges.append([i, j])
        return edges

    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    current_cost = count_regular_6(adj)
    
    # If already valid, return
    if current_cost == 0 and is_valid_full(adj):
        return {'n': n, 'edges': get_edges(adj)}
        
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip edge (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        new_cost = count_regular_6(adj)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost == 0:
                if is_valid_full(adj):
                    return {'n': n, 'edges': get_edges(adj)}
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        # Optional: Random restart if stuck too long?
        # For now, simple hill climbing.
        
    # If we didn't find a valid graph, return the best we have (which might be invalid, but score 0 is better than timeout)
    # Actually, if it's invalid, score is 0.
    # We should try to return a valid graph if possible.
    # If we can't find one, we can return a smaller n that is valid?
    # The task says "Maximize n".
    # If we fail to find n=20, we should try to find a valid n=19, 18, etc.
    # But the search above is for n=20.
    # Let's add a fallback: if we haven't found n=20, try to construct a valid graph for smaller n?
    # Or just return the current state (likely invalid) -> Score 0.
    # Better: If we get stuck, we can try to remove a vertex and see if the remaining n-1 is valid?
    # But that changes n.
    
    # Let's just return the current best effort. If invalid, it scores 0.
    # To be safe, we can check if the current graph is valid.
    if is_valid_full(adj):
        return {'n': n, 'edges': get_edges(adj)}
        
    # Fallback: return a known small valid graph to ensure non-zero score?
    # The seed returns n=5. Score 1.0.
    # If we return invalid n=20, score 0.
    # So we should ensure we return a valid graph.
    # Let's try to find a valid graph for n=5 as a fallback.
    return {'n': 5, 'edges': []}
