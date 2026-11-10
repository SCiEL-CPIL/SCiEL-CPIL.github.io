import random
import time
from itertools import combinations

def _find_bad_subset(adj, n, max_size=None):
    """Find a regular induced subgraph of size >= 6, or return None if none."""
    if max_size is None:
        max_size = n
    for size in range(6, max_size + 1):
        for comb in combinations(range(n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            first = -1
            regular = True
            for v in comb:
                d = (adj[v] & mask).bit_count()
                if first == -1:
                    first = d
                elif d != first:
                    regular = False
                    break
            if regular:
                return comb
    return None

def _flip_edge(adj, u, v):
    bit_v = 1 << v
    bit_u = 1 << u
    if adj[u] & bit_v:
        adj[u] &= ~bit_v
        adj[v] &= ~bit_u
    else:
        adj[u] |= bit_v
        adj[v] |= bit_u

def _to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append([i, j])
    return edges

def construct():
    start = time.time()
    deadline = start + 145  # safety margin
    
    rng = random.Random(12345)
    best = None  # (n, edges)
    
    def elapsed():
        return time.time() - start
    
    def try_find(n, time_budget):
        """Try to find a valid graph on n vertices within time_budget seconds."""
        target_time = time.time() + time_budget
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > target_time or elapsed() > deadline - 5:
                break
            # Random init with varying density
            density = 0.3 + 0.4 * rng.random()
            adj = [0]*n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search with simulated annealing-like behavior
            current_temp = 1.0
            for it in range(500):
                if time.time() > target_time or elapsed() > deadline - 5:
                    break
                
                # Check for bad subset
                bad = _find_bad_subset(adj, n)
                if bad is None:
                    return adj
                
                # Find a bad edge to flip - prefer edges that break regularity
                # Try flipping edges within the bad subset
                bad_list = list(bad)
                # Pick two vertices from the bad subset
                a = rng.choice(bad_list)
                b = rng.choice(bad_list)
                while a == b:
                    b = rng.choice(bad_list)
                
                # Count how many bad subsets contain this edge
                # For now, just flip randomly
                _flip_edge(adj, a, b)
                
                # Occasionally flip an edge outside the bad subset for diversity
                if rng.random() < 0.1:
                    u = rng.randint(0, n-1)
                    v = rng.randint(0, n-1)
                    while u == v:
                        v = rng.randint(0, n-1)
                    _flip_edge(adj, u, v)
                
                # Cool down temperature
                current_temp *= 0.998
        return None
    
    # Try n=21 first (ambitious), then n=20 (known achievable)
    for n_try in [21, 20]:
        if elapsed() > deadline - 15:
            break
        remaining = deadline - 15 - elapsed()
        if n_try == 21:
            budget = min(60, remaining)
        else:
            budget = min(80, remaining)
        if budget < 5:
            break
        adj = try_find(n_try, budget)
        if adj is not None:
            # Verify
            if _find_bad_subset(adj, n_try) is None:
                edges = _to_edges(adj, n_try)
                best = (n_try, edges)
                if n_try == 21:
                    break  # Found the target, stop
    
    # If we didn't find 21, try to improve 20 or find smaller
    if best is None or best[0] < 20:
        if elapsed() > deadline - 10:
            pass
        else:
            remaining = deadline - 10 - elapsed()
            if remaining > 5:
                budget = min(40, remaining)
                adj = try_find(20, budget)
                if adj is not None:
                    if _find_bad_subset(adj, 20) is None:
                        edges = _to_edges(adj, 20)
                        best = (20, edges)
    
    if best is None:
        # Try n=19
        if elapsed() < deadline - 5:
            remaining = deadline - 5 - elapsed()
            if remaining > 3:
                budget = min(20, remaining)
                adj = try_find(19, budget)
                if adj is not None:
                    if _find_bad_subset(adj, 19) is None:
                        edges = _to_edges(adj, 19)
                        best = (19, edges)
    
    if best is None:
        # Fallback to smaller n
        for n_try in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]:
            if elapsed() > deadline - 3:
                break
            remaining = deadline - 3 - elapsed()
            if remaining < 1:
                break
            adj = try_find(n_try, min(5, remaining))
            if adj is not None:
                if _find_bad_subset(adj, n_try) is None:
                    edges = _to_edges(adj, n_try)
                    best = (n_try, edges)
                    break
    
    if best is None:
        # Absolute fallback: n=5 empty graph
        return {'n': 5, 'edges': []}
    
    return {'n': best[0], 'edges': best[1]}
