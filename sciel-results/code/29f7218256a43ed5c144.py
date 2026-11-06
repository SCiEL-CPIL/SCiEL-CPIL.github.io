"""Structural approach: polarity graph base + vertex extension with local search."""
import time
import random

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):
                return True
    return False

def _max_is_bitmask(n, adj, target=11):
    """Check if there exists an independent set of size >= target."""
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        return max(include_val, exclude_val)
    return max_is((1 << n) - 1, target)

def _try_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) preserves C4-freeness."""
    common = adj[u] & adj[v]
    return not (common & (common - 1))

def _build_polarity_graph_q5():
    """
    Build the polarity graph of PG(2,5) using the standard symplectic polarity.
    This gives a C4-free graph on 31 vertices.
    """
    q = 5
    # Enumerate points of PG(2,q) as normalized triples
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: divide by first non-zero component
                if x != 0:
                    inv_x = pow(x, q - 2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)  # Should be 31
    
    # Bilinear form for polarity: B([x:y:z], [x':y':z']) = x*z' + z*x' - y*y'
    def bilinear(p, r):
        return (p[0] * r[2] + p[2] * r[0] - p[1] * r[1]) % q
    
    adj = [0] * n_pts
    edges = []
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if bilinear(points[i], points[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
    
    return adj, edges, n_pts

def _extend_graph(adj, edges, current_n, target_n, deadline):
    """
    Try to extend a C4-free graph from current_n to target_n vertices
    by adding vertices with random neighbor sets.
    Returns (adj, edges, n) or None if failed.
    """
    for attempt in range(100):
        if time.time() > deadline:
            break
        if current_n >= target_n:
            break
        
        # Add a new vertex with random neighbors
        new_v = current_n
        # Choose how many neighbors to give the new vertex
        num_neighbors = random.randint(3, 7)
        
        # Sample random neighbors
        candidates = list(range(current_n))
        random.shuffle(candidates)
        
        new_adj_mask = 0
        for c in candidates:
            if new_adj_mask.bit_count() >= num_neighbors:
                break
            # Check if adding edge (new_v, c) is safe
            # For a new vertex, the only C4 risk is if two of its neighbors
            # have a common neighbor (which would form a C4 with the new vertex)
            # Actually, for a new vertex, adding edge to c is safe as long as
            # no two existing neighbors of c are both neighbors of new_v AND
            # share a common neighbor... 
            # Simpler: just check that for each existing neighbor c, 
            # the set of common neighbors of new_v and c has size <= 1
            # Since new_v is new, common neighbors of new_v and c are
            # neighbors of new_v that are also neighbors of c
            # We need |N(new_v) ∩ N(c)| <= 1 for all c
            # But we're building N(new_v) incrementally, so:
            # For each candidate c, check |{w in N(new_v) : w ~ c}| <= 1
            conflict = 0
            for w in range(current_n):
                if (new_adj_mask >> w) & 1 and (adj[c] >> w) & 1:
                    conflict += 1
                    if conflict > 1:
                        break
            if conflict <= 1:
                new_adj_mask |= (1 << c)
        
        # Also need to check C4-freeness among existing vertices is preserved
        # (it is, since we're only adding edges from new_v)
        
        # Update adj
        adj.append(new_adj_mask)
        for w in range(current_n):
            if (new_adj_mask >> w) & 1:
                adj[w] |= (1 << new_v)
                edges.append([new_v, w])
        
        current_n += 1
    
    # Final C4 check
    if _has_c4_bitmask(current_n, adj):
        return None
    
    return adj, edges, current_n

def _local_search_reduce_alpha(adj, edges, n, deadline):
    """
    Local search: try edge swaps to reduce independence number while maintaining C4-freeness.
    Returns (adj, edges, n) with reduced alpha if successful.
    """
    current_alpha = _max_is_bitmask(n, adj, 11)
    
    for step in range(2000):
        if time.time() > deadline:
            break
        if current_alpha < 11:
            break
        
        # Pick a random edge to remove
        if not edges:
            break
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        
        # Pick a random non-edge to add
        w = random.randint(0, n - 1)
        x = random.randint(0, n - 1)
        if w == x or w == u or w == v or x == u or x == v:
            continue
        if (adj[w] >> x) & 1:
            continue
        
        # Check if adding (w,x) is C4-safe
        if not _try_add_edge(adj, n, w, x):
            continue
        
        # Perform swap
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.pop(idx)
        
        adj[w] |= (1 << x)
        adj[x] |= (1 << w)
        edges.append([w, x])
        
        # Check C4-freeness
        if _has_c4_bitmask(n, adj):
            # Revert
            adj[w] &= ~(1 << x)
            adj[x] &= ~(1 << w)
            edges.pop()
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            continue
        
        new_alpha = _max_is_bitmask(n, adj, 11)
        if new_alpha <= current_alpha:
            current_alpha = new_alpha
        else:
            # Revert
            adj[w] &= ~(1 << x)
            adj[x] &= ~(1 << w)
            edges.pop()
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
    
    return adj, edges, n, current_alpha

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    deadline = time.time() + 140
    
    # Strategy: Build polarity graph of PG(2,5) (31 vertices), extend to 39,
    # then local search to reduce alpha below 11.
    
    for outer_seed in range(20):
        if time.time() > deadline:
            break
        random.seed(outer_seed * 104729 + 31)
        
        # Build base polarity graph
        adj, edges, n_base = _build_polarity_graph_q5()
        
        if _has_c4_bitmask(n_base, adj):
            continue
        
        # Extend to 39 vertices
        result = _extend_graph(adj, edges, n_base, 39, deadline)
        if result is None:
            continue
        
        adj, edges, n_cur = result
        
        # Local search to reduce alpha
        adj, edges, n_cur, alpha = _local_search_reduce_alpha(adj, edges, n_cur, deadline)
        
        if alpha < 11:
            return {'n': n_cur, 'edges': edges}
    
    # Fallback: try random regular graph approach
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed * 7919 + 13)
        n = 39
        adj = [0] * n
        edges = []
        
        for i in range(n):
            candidates = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            random.shuffle(candidates)
            for j in candidates[:5]:
                if _try_add_edge(adj, n, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        if _has_c4_bitmask(n, adj):
            continue
        
        alpha = _max_is_bitmask(n, adj, 11)
        if alpha < 11:
            return {'n': n, 'edges': edges}
        
        # Quick local search
        adj, edges, n, alpha = _local_search_reduce_alpha(adj, edges, n, deadline)
        if alpha < 11:
            return {'n': n, 'edges': edges}
    
    # Final fallback
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
