import random
import time

def _girth_is_at_least_g(n, edges, g):
    """Check if the girth is at least g using non-backtracking walks from every vertex."""
    if n == 0:
        return True
    # Build adjacency list
    adj = [[] for _ in range(n)]
    # Pre-compute neighbor indices to avoid dictionary lookups if needed, but list is fine
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Map vertex to index in adj list for faster state encoding?
    # State: (v, prev). Encode as v * 10 + prev_idx? 
    # Simpler: just use (v, prev) tuple. Python tuple hashing is fast enough for small N.
    
    for start in range(n):
        # State: (current_vertex, previous_vertex)
        # Start: (start, -1)
        visited = set()
        queue = [(start, -1)]
        visited.add((start, -1))
        depth = 0
        while queue and depth < g - 1:
            next_queue = []
            for u, prev in queue:
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == start and depth + 1 < g:
                        return False
                    if v == start:
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        next_queue.append(state)
            queue = next_queue
            depth += 1
    return True

def _is_valid(n, offsets, k, g):
    if len(offsets) != k // 2:
        return False
    for off in offsets:
        if off % n == 0:
            return False
    
    edges = []
    edge_set = set()
    for i in range(n):
        for off in offsets:
            j = (i + off) % n
            if i < j:
                if (i, j) in edge_set:
                    return False
                edge_set.add((i, j))
                edges.append((i, j))
            else:
                if (j, i) in edge_set:
                    return False
                edge_set.add((j, i))
                edges.append((j, i))
                
    # Check degree (should be k by construction if offsets are distinct and non-zero/non-n/2)
    # But we need to ensure no multi-edges.
    # The set check above handles multi-edges.
    
    if not _girth_is_at_least_g(n, edges, g):
        return False
        
    return True

def construct(k, g):
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g-1)//2 - 1))
        
    max_vertices = 500
    random.seed(42)
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s for overhead
    
    for n in range(max(moore_bound, k+1), max_vertices + 1):
        if time.time() - start_time > timeout:
            break
            
        attempts = 0
        max_attempts = 5000 # Increase attempts if time allows
        found = False
        
        valid_offsets = list(range(1, n))
        if n % 2 == 0:
            valid_offsets.remove(n // 2)
            
        if len(valid_offsets) < k // 2:
            continue
            
        while attempts < max_attempts and not found:
            if time.time() - start_time > timeout:
                break
            attempts += 1
            offsets = random.sample(valid_offsets, k // 2)
            if _is_valid(n, offsets, k, g):
                edges = []
                for i in range(n):
                    for off in offsets:
                        j = (i + off) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    # Fallback
    # ... (keep existing fallback)
