import random
import itertools
import time

def _find_violation(n, adj_mask):
    """Find a regular induced subgraph of size >= 6.
    Returns the subset as a tuple if found, else None."""
    vertices = list(range(n))
    # Check sizes from 6 to n
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            mask_S = 0
            for v in subset:
                mask_S |= (1 << v)
            
            degs = []
            for v in subset:
                common = adj_mask[v] & mask_S
                # Count bits
                d = bin(common).count('1')
                degs.append(d)
            
            if len(set(degs)) == 1:
                return subset
    return None

def _check_valid(n, adj_mask):
    return _find_violation(n, adj_mask) is None

def _random_adj_mask(n, p=0.5):
    adj_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
    return adj_mask

def _flip_edge(adj_mask, u, v):
    if adj_mask[u] & (1 << v):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
    else:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

def _local_search(n, time_limit=10):
    """Try to find a valid graph on n vertices using local search."""
    start_time = time.time()
    best_adj = None
    
    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break
            
        adj_mask = _random_adj_mask(n, 0.5)
        
        # Local search phase
        for step in range(500):
            if time.time() - start_time > time_limit:
                break
                
            viol = _find_violation(n, adj_mask)
            if viol is None:
                return adj_mask
            
            # Pick a random edge in the violating subset to flip
            # To be more effective, we can try flipping each edge in the subset and see which one breaks the regularity
            # But we don't know if it creates others.
            # Simple strategy: Flip a random edge in the subset.
            u, v = random.sample(list(viol), 2)
            _flip_edge(adj_mask, u, v)
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_n = 0
    best_edges = []
    
    # Try n=20 first, as it's known to exist
    # Then try n=21
    
    for target_n in [20, 21]:
        if time.time() > deadline - 10:
            break
            
        adj_mask = _local_search(target_n, time_limit=50)
        
        if adj_mask is not None:
            # Verify
            if _check_valid(target_n, adj_mask):
                best_n = target_n
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                best_edges = edges
                # If we found 21, we can stop. If 20, try 21?
                # The loop tries 20 then 21.
                # If 20 is found, we continue to 21.
                # If 21 is found, we update best.
            else:
                # Should not happen if _find_violation is correct
                pass
        else:
            # If local search failed, maybe try a few random restarts?
            # Or just accept failure for this n.
            pass

    # Fallback: if we didn't find 20 or 21, try smaller n?
    # The task says "Maximize n".
    # If we fail to find 20, we should return the best we found.
    # But 20 is known.
    
    # If best_n is 0, we failed.
    # Let's add a fallback to find a small valid graph quickly to ensure non-zero score.
    if best_n == 0:
        # Try n=6, 7...
        for target_n in range(6, 15):
            if time.time() > deadline:
                break
            adj_mask = _local_search(target_n, time_limit=5)
            if adj_mask is not None and _check_valid(target_n, adj_mask):
                best_n = target_n
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                best_edges = edges
                break

    return {'n': best_n, 'edges': best_edges}
