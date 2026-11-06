import time
import random

DEADLINE = 140.0

def _construct_andrasfai(k):
    """Construct the Andrasfai graph A_k on k^2 vertices."""
    n = k * k
    adj = [0] * n
    for i in range(n):
        x1 = i // k
        y1 = i % k
        for j in range(i + 1, n):
            x2 = j // k
            y2 = j % k
            if x1 != x2:
                if (y2 - y1) % k == (x1 * x2) % k:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
    return adj, n

def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets."""
    for i in range(n):
        ai = adj[i]
        j = ai & ~((1 << (i + 1)) - 1)  # only j > i
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            common = ai & adj[jj] & ~((1 << (jj + 1)) - 1)
            if common:
                return True
            j ^= lb
    return False

def _greedy_alpha(adj, n):
    """Greedy independent set size. Returns size of found set."""
    candidates = (1 << n) - 1
    count = 0
    while candidates:
        # Pick vertex with min degree in candidates? Or just first?
        # Min degree is better for independence.
        min_deg = n
        min_v = -1
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            c ^= lb
        if min_v == -1:
            break
        count += 1
        candidates &= ~adj[min_v]
        candidates &= ~(1 << min_v)
    return count

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound. Returns >= limit if alpha >= limit."""
    best = [0]
    
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        
        # Heuristic: pick vertex with max degree in candidates subgraph
        # Actually for independence set, we want to branch on a vertex.
        # Standard B&B for Max Independent Set (which is Max Clique on complement)
        # Here we are finding Max Independent Set in G.
        
        # Pick a vertex v from candidates
        # To prune, we can use the fact that if we include v, we remove its neighbors.
        # If we exclude v, we keep everything.
        
        # Simple branch:
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        
        # Branch 1: Exclude v
        bnb(chosen_count, rest)
        
        # Branch 2: Include v
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
        
    bnb(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    # Quick check
    if _greedy_alpha(adj, n) >= 10:
        return False
    # Exact check
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def _random_maximal_triangle_free(n):
    """Generate a random maximal triangle-free graph on n vertices."""
    adj = [0] * n
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    random.shuffle(pairs)
    
    for u, v in pairs:
        # Check if adding (u, v) creates a triangle
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _random_subgraph_andrasfai(adj_full, n_full, n_target):
    """Pick a random subset of n_target vertices from the full graph."""
    vertices = list(range(n_full))
    random.shuffle(vertices)
    selected = vertices[:n_target]
    
    # Map old indices to new indices 0..n_target-1
    old_to_new = {old: new for new, old in enumerate(selected)}
    
    adj = [0] * n_target
    for i, old_i in enumerate(selected):
        neighbors = adj_full[old_i]
        # Filter neighbors that are in selected set
        new_neighbors = 0
        # Iterate through bits of neighbors
        nb = neighbors
        while nb:
            lb = nb & (-nb)
            old_j = lb.bit_length() - 1
            if old_j in old_to_new:
                new_j = old_to_new[old_j]
                new_neighbors |= (1 << new_j)
            nb ^= lb
        adj[i] = new_neighbors
        
    return adj

def construct():
    start_time = time.time()
    
    # Precompute Andrasfai graph A_9
    k = 9
    adj_full, n_full = _construct_andrasfai(k)
    
    n_target = 40
    
    # Try to find a valid graph on n=40
    for attempt in range(1000):
        if time.time() > start_time + DEADLINE - 5:
            break
            
        # Alternate between strategies
        if attempt % 2 == 0:
            # Strategy 1: Random subset of A_9
            adj = _random_subgraph_andrasfai(adj_full, n_full, n_target)
        else:
            # Strategy 2: Random maximal triangle-free
            adj = _random_maximal_triangle_free(n_target)
            
        # Check validity
        if _is_valid(adj, n_target):
            edges = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n_target, 'edges': edges}
            
    # Fallback to n=39 if 40 fails
    for n_try in range(39, 25, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
        for attempt in range(50):
            if time.time() > start_time + DEADLINE - 5:
                break
            adj = _random_maximal_triangle_free(n_try)
            if _is_valid(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
                
    return {'n': 0, 'edges': []}
