import random
import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit

    rng = random.Random(42)
    
    best_result = None
    best_n = 0

    # Prioritize finding n=20 (known to exist)
    # Then try n=21 if time permits
    # Then fallback to smaller n if search fails
    
    targets = [20, 21, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    
    for target_n in targets:
        if time.time() > deadline - 10:
            break
            
        # If we already have a valid graph on 20 vertices, 
        # only try 21 if we have significant time left.
        if target_n == 21 and best_n >= 20:
            if time.time() > deadline - 40:
                continue
                
        result = _search_for_graph(target_n, deadline - 5, rng)
        
        if result is not None:
            if result['n'] > best_n:
                best_n = result['n']
                best_result = result
                
        # If we found 20, we can stop trying smaller sizes
        # unless we are trying to find 21.
        # The loop handles this by checking best_n.
        # But we can break early if best_n == 20 and target_n < 20.
        if best_n >= 20 and target_n < 20:
            break

    if best_result is None:
        # Fallback to a small valid graph
        best_result = _make_fallback(rng)
        
    return best_result

def _search_for_graph(n, deadline, rng):
    """
    Local search to find a valid graph on n vertices.
    Returns dict with 'n' and 'edges' or None.
    """
    # Try multiple random initializations
    # Fewer restarts for larger n to allow longer search per restart
    max_restarts = 5 if n < 20 else 3
    
    for r in range(max_restarts):
        if time.time() > deadline:
            break
            
        adj = [[0]*n for _ in range(n)]
        
        # Random initialization
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Precompute adjacency masks for fast degree counting
        masks = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
                    
        # Local search
        success = _local_search_loop(adj, masks, n, deadline, rng)
        
        if success:
            # Reconstruct edges list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return None

def _local_search_loop(adj, masks, n, deadline, rng):
    """
    Iteratively find and fix violations.
    Returns True if the graph is valid (no violations).
    """
    # Initial check
    viol = _find_violation(masks, n)
    if viol is None:
        return True

    while time.time() < deadline:
        viol = _find_violation(masks, n)
        if viol is None:
            return True

        # Pick a random edge within the violating subset to flip
        # This breaks the regularity of this specific subset
        u_idx, v_idx = rng.sample(range(len(viol)), 2)
        u = viol[u_idx]
        v = viol[v_idx]
        
        # Flip edge (u, v)
        if adj[u][v]:
            adj[u][v] = 0
            adj[v][u] = 0
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        else:
            adj[u][v] = 1
            adj[v][u] = 1
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
        # Optional: random perturbation to escape local minima
        if rng.random() < 0.1:
            a, b = rng.sample(range(n), 2)
            if adj[a][b]:
                adj[a][b] = 0
                adj[b][a] = 0
                masks[a] &= ~(1 << b)
                masks[b] &= ~(1 << a)
            else:
                adj[a][b] = 1
                adj[b][a] = 1
                masks[a] |= (1 << b)
                masks[b] |= (1 << a)

    # Final check before returning
    return _find_violation(masks, n) is None

def _find_violation(masks, n):
    """
    Find a subset of size >= 6 that induces a regular graph.
    Returns the subset (list of vertices) or None.
    """
    # Check size 6 first, as it's the most restrictive
    # Then 7, etc.
    for size in range(6, n + 1):
        for subset in itertools.combinations(range(n), size):
            if _is_regular_subset(masks, subset):
                return list(subset)
    return None

def _is_regular_subset(masks, subset):
    """
    Check if the induced subgraph on 'subset' is regular.
    """
    s = len(subset)
    if s < 2:
        return True
        
    subset_mask = 0
    for v in subset:
        subset_mask |= (1 << v)
        
    degs = []
    for v in subset:
        d = bin(masks[v] & subset_mask).count('1')
        degs.append(d)
        
    # Check if all degrees are the same
    d0 = degs[0]
    for d in degs[1:]:
        if d != d0:
            return False
    return True

def _make_fallback(rng):
    """
    Generate a verified valid graph on 10 vertices.
    """
    n = 10
    for _ in range(100):
        adj = [[0]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
                    edges.append([i, j])
                    
        masks = [0]*n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
                    
        if _find_violation(masks, n) is None:
            return {'n': n, 'edges': edges}
            
    # Extremely unlikely to fail, but return empty graph if so
    return {'n': 1, 'edges': []}
# EVOLVE-BLOCK-END
