import time
import random
import itertools

# Fast popcount using a table for 8-bit chunks
_POPCOUNT_TABLE = [bin(i).count('1') for i in range(256)]

def _popcount(x):
    count = 0
    while x:
        count += _POPCOUNT_TABLE[x & 0xFF]
        x >>= 8
    return count

def _find_violation(adj, n, start_size=6):
    """Find a subset of size >= start_size that induces a regular graph.
    Returns (subset_list, degree) or None.
    Prioritizes smaller subsets.
    """
    vertices = list(range(n))
    
    # Check sizes from start_size to n
    for size in range(start_size, n + 1):
        for subset in itertools.combinations(vertices, size):
            # Build mask
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Check degrees
            first_deg = -1
            is_reg = True
            for v in subset:
                d = _popcount(adj[v] & mask)
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return list(subset), first_deg
    return None

def _count_violations_size(adj, n, size):
    """Count regular induced subgraphs of a specific size."""
    count = 0
    vertices = list(range(n))
    for subset in itertools.combinations(vertices, size):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        first_deg = -1
        is_reg = True
        for v in subset:
            d = _popcount(adj[v] & mask)
            if first_deg == -1:
                first_deg = d
            elif d != first_deg:
                is_reg = False
                break
        if is_reg:
            count += 1
    return count

def _local_search(n, deadline):
    """Local search to find a valid graph on n vertices."""
    # Initial random graph
    adj = [0] * n
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.5:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
    
    best_adj = adj[:]
    best_edges = edges[:]
    best_violations = float('inf')
    
    # We will try to minimize the number of violations of size 6 first.
    # If we reach 0 violations of size 6, we then check for size > 6.
    
    current_adj = adj[:]
    current_edges = edges[:]
    
    # Calculate initial violations of size 6
    # This might be slow if n is large, but n=20 is manageable.
    # _count_violations_size(current_adj, n, 6)
    
    # To speed up, we just find one violation and try to fix it.
    # This is a greedy hill climbing.
    
    moves = 0
    max_moves = 2000
    
    while moves < max_moves and time.time() < deadline:
        moves += 1
        
        # Find a violation of size 6
        viol = _find_violation(current_adj, n, start_size=6)
        
        if viol is None:
            # No size 6 violations. Check if there are any larger ones.
            # This is expensive, so we only do it if we are "close" to success
            # or periodically.
            # For now, assume if no size 6 violations, we check full validity.
            full_viol = _find_violation(current_adj, n, start_size=7)
            if full_viol is None:
                return current_adj, current_edges
            
            # If there is a larger violation, we need to fix it.
            # But fixing a size 7 violation might create size 6 violations.
            # Let's just pick an edge in the larger violation and flip it.
            subset, deg = full_viol
            # Pick a random edge in the subset
            u, v = random.sample(subset, 2)
            
            # Flip edge
            if current_adj[u] & (1 << v):
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges.remove((u, v))
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append((u, v))
            continue

        # We found a size 6 violation
        subset, deg = viol
        
        # Try to flip an edge in this subset to break regularity
        # We want to change the degree of at least one vertex in the subset
        # such that they are no longer all equal.
        
        # Strategy: Flip an edge incident to one of the vertices in the subset,
        # ideally one that connects two vertices in the subset.
        
        # Get all pairs in subset
        pairs = list(itertools.combinations(subset, 2))
        random.shuffle(pairs)
        
        moved = False
        for u, v in pairs:
            # Tentatively flip
            if current_adj[u] & (1 << v):
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
                current_edges.remove((u, v))
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_edges.append((u, v))
            
            # Check if this specific violation is gone
            # We don't need to check all violations, just this one?
            # If we break this one, we might create another.
            # Let's just accept the move if it doesn't increase the number of size-6 violations too much?
            # Or simpler: Just accept the move. Hill climbing.
            
            # Actually, let's verify if the specific subset is still regular.
            mask = 0
            for x in subset:
                mask |= (1 << x)
            
            is_still_reg = True
            d0 = _popcount(current_adj[subset[0]] & mask)
            for x in subset[1:]:
                if _popcount(current_adj[x] & mask) != d0:
                    is_still_reg = False
                    break
            
            if not is_still_reg:
                # Good move, kept the change
                moved = True
                break
            else:
                # Revert
                if current_adj[u] & (1 << v):
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.remove((u, v))
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))
        
        if not moved:
            # Random move to escape local minimum
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                if current_adj[u] & (1 << v):
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    current_edges.remove((u, v))
                else:
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_edges.append((u, v))

    # If loop finishes without success, return best found (or current)
    # We should track the state with the fewest violations.
    # For simplicity, return current.
    return current_adj, current_edges

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n=21 first, then 20
    for n in [21, 20]:
        if time.time() > deadline:
            break
            
        # Multiple restarts
        for attempt in range(10):
            if time.time() > deadline:
                break
            adj, edges = _local_search(n, deadline)
            
            # Final verification
            if _find_violation(adj, n, start_size=6) is None:
                return {'n': n, 'edges': edges}
                
    # Fallback to n=19 or smaller if 20/21 failed
    for n in range(19, 5, -1):
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            adj, edges = _local_search(n, deadline)
            if _find_violation(adj, n, start_size=6) is None:
                return {'n': n, 'edges': edges}
                
    return {'n': 5, 'edges': []}
