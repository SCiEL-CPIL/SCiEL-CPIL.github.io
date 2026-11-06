"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 150.0
SAFETY_MARGIN = 15.0

def _has_triangle_bitset(adj, n):
    """Check triangle-free using bitsets. adj is list of ints (bitmasks)."""
    for u in range(n):
        # For each neighbor v > u, check if any neighbor of v is also neighbor of u
        neighbors_u = adj[u]
        # Only check edges where u < v
        v = u + 1
        while v < n:
            if (neighbors_u >> v) & 1:
                # Check if adj[v] & neighbors_u has any bit set for w > v
                common = adj[v] & neighbors_u
                # Mask out bits <= v to only check w > v
                common &= ~((1 << (v + 1)) - 1)
                if common:
                    return True
            v += 1
    return False

def _independence_number_bitset(adj, n):
    """Exact maximum independent set size via branch and bound with bitsets."""
    # adj[i] is bitmask of neighbors of i
    best = 0
    
    # Precompute non-neighbors for each vertex (excluding self)
    full_mask = (1 << n) - 1
    
    def branch(candidates_mask, size):
        nonlocal best
        if size + candidates_mask.bit_count() <= best:
            return
        if candidates_mask == 0:
            best = max(best, size)
            return
        
        # Choose vertex with max degree in candidates for better pruning
        # Find vertex in candidates with most neighbors in candidates
        cands = candidates_mask
        best_v = -1
        best_deg = -1
        while cands:
            # Extract lowest set bit
            lsb = cands & (-cands)
            v = lsb.bit_length() - 1
            deg = (adj[v] & candidates_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
            cands ^= lsb
        
        if best_v < 0:
            best = max(best, size)
            return
        
        v = best_v
        v_bit = 1 << v
        
        # Include v: remove v and all its neighbors from candidates
        new_cands = candidates_mask & ~(adj[v] | v_bit)
        branch(new_cands, size + 1)
        
        # Exclude v
        branch(candidates_mask & ~v_bit, size)
    
    branch(full_mask, 0)
    return best

def _is_valid(adj, n):
    """Check if graph is triangle-free and has independence number < 10."""
    if _has_triangle_bitset(adj, n):
        return False
    alpha = _independence_number_bitset(adj, n)
    return alpha < 10

def _edges_to_adj(edges, n):
    """Convert edge list to bitset adjacency."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _adj_to_edges(adj, n):
    """Convert bitset adjacency to edge list."""
    edges = []
    for u in range(n):
        v = u + 1
        while v < n:
            if (adj[u] >> v) & 1:
                edges.append([u, v])
            v += 1
    return edges

def _search_circulant(n):
    """Search for a circulant graph on n vertices that is triangle-free with alpha < 10."""
    # A circulant graph on n vertices is defined by a connection set S ⊂ {1, ..., n//2}
    # where vertex i is connected to i±s mod n for s in S
    # For triangle-free, we need S to be sum-free mod n (no a+b+c ≡ 0 mod n with a,b,c in S∪{-S})
    
    # This is a complex search. Let's try a simpler approach:
    # Use the known Andrasfai graph structure and try to extend it.
    
    # For now, return None to indicate no solution found via this method
    return None

def _random_local_search(n, adj, max_iterations=1000):
    """Local search: try to add edges to increase density while maintaining validity."""
    best_adj = list(adj)
    best_score = 0
    for i in range(n):
        best_score += best_adj[i].bit_count()
    
    for iteration in range(max_iterations):
        # Try adding a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        # Check if edge already exists
        if (best_adj[u] >> v) & 1:
            continue
        
        # Try adding the edge
        new_adj = list(best_adj)
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        
        # Check if still triangle-free
        if _has_triangle_bitset(new_adj, n):
            continue
        
        # Check independence number
        alpha = _independence_number_bitset(new_adj, n)
        if alpha >= 10:
            continue
        
        # Accept the move
        best_adj = new_adj
        best_score += 2
    
    return best_adj

def _greedy_construct(n, time_limit):
    """Greedy construction of a triangle-free graph with alpha < 10."""
    start_time = time.time()
    adj = [0] * n
    
    # Generate all possible edges in random order
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(pairs)
    
    # Add edges greedily, checking validity periodically
    check_interval = max(1, len(pairs) // 20)
    
    for idx, (u, v) in enumerate(pairs):
        if time.time() - start_time > time_limit:
            break
        
        # Check if adding (u,v) creates a triangle
        # Common neighbors of u and v
        common = adj[u] & adj[v]
        if common:
            continue
        
        # Try adding the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Periodically check if we still have alpha < 10
        if idx % check_interval == 0:
            alpha = _independence_number_bitset(adj, n)
            if alpha >= 10:
                # Roll back
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    # Final check
    if _has_triangle_bitset(adj, n):
        return None
    alpha = _independence_number_bitset(adj, n)
    if alpha >= 10:
        return None
    
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_n = 0
    best_edges = []
    
    # Try Andrasfai-type constructions
    for r in range(1, 15):
        if time.time() > deadline:
            break
        n = 3 * r
        adj = [0] * n
        for i in range(n):
            for s in [1, 2]:  # Connect to i±1 and i±2 mod n? No, Andrasfai is specific
                pass
        
        # Standard Andrasfai: i~j iff (j-i) mod 3r ≡ 1 mod 3
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if _is_valid(adj, n):
            if n > best_n:
                best_n = n
                best_edges = _adj_to_edges(adj, n)
    
    # Try to build larger graphs using greedy + local search
    for target_n in range(max(best_n + 1, 28), 41):
        if time.time() > deadline:
            break
        
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # Try multiple random greedy constructions
        for attempt in range(10):
            if time.time() > deadline:
                break
            
            time_per_attempt = remaining_time / 10
            adj = _greedy_construct(target_n, time_per_attempt * 0.7)
            
            if adj is not None:
                # Try to improve with local search
                remaining_for_ls = deadline - time.time()
                if remaining_for_ls > 2:
                    adj = _random_local_search(target_n, adj, max_iterations=500)
                
                if _is_valid(adj, target_n):
                    if target_n > best_n:
                        best_n = target_n
                        best_edges = _adj_to_edges(adj, target_n)
                    break  # Found valid graph for this target_n
    
    # If we haven't found anything, fall back to Andrasfai And(9)
    if best_n <= 0:
        n = 27
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        best_n = n
        best_edges = _adj_to_edges(adj, n)
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
