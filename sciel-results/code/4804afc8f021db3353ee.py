import random
import time

def construct():
    n = 20
    deadline = time.time() + 110  # Safety margin
    
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
    
    # Precompute for each vertex, which triples contain it
    triples_by_vertex = [[] for _ in range(n)]
    for idx, (a, b, c) in enumerate(triples):
        triples_by_vertex[a].append(idx)
        triples_by_vertex[b].append(idx)
        triples_by_vertex[c].append(idx)
    
    def is_valid_adj(adj):
        """Check if adj is K3,3-free. adj is list of ints (bitmasks)."""
        for a, b, c in triples:
            if (adj[a] & adj[b] & adj[c]).bit_count() > 2:
                return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without creating K3,3. Fast check."""
        if (adj[u] >> v) & 1:
            return True  # Already exists
        
        # Only need to check triples containing u or v
        # Triples containing u: (u, x, y) for x != u, y != u, x != y
        # After adding (u,v), N(u) gains v, N(v) gains u
        # For triple (u, x, y): intersection = N(u)&N(x)&N(y)
        #   Before: doesn't include v in N(u) unless v was already neighbor
        #   After: N(u) now includes v
        #   So new intersection = old_intersection | (v if v in N(x) and v in N(y))
        #   But also, if u was in N(x) or N(y), that doesn't change
        # Actually simpler: just check the triples directly
        
        # Check triples containing u
        for x in range(n):
            if x == u:
                continue
            for y in range(n):
                if y == u or y == x:
                    continue
                # Triple (u, x, y)
                common = adj[u] & adj[x] & adj[y]
                # After adding (u,v): N(u) gains v
                # So new common = common | (v if v in adj[x] and v in adj[y])
                if (adj[x] >> v) & 1 and (adj[y] >> v) & 1:
                    common |= (1 << v)
                if common.bit_count() > 2:
                    return False
        
        # Check triples containing v (but not u, since those with u were checked above)
        for x in range(n):
            if x == v or x == u:
                continue
            for y in range(n):
                if y == v or y == u or y == x:
                    continue
                # Triple (v, x, y)
                common = adj[v] & adj[x] & adj[y]
                # After adding (u,v): N(v) gains u
                if (adj[x] >> u) & 1 and (adj[y] >> u) & 1:
                    common |= (1 << u)
                if common.bit_count() > 2:
                    return False
        
        return True
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v) to adj. Return True if successful."""
        if not can_add_edge(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v) from adj."""
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    
    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2
    
    def greedy_construct():
        """Construct a valid graph using greedy edge addition."""
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj
    
    def local_search(adj):
        """Improve adj using local search with add and swap moves."""
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Try adding edges
            missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if not ((adj[i] >> j) & 1)]
            random.shuffle(missing_edges)
            
            for u, v in missing_edges:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
            
            # Try swap moves: remove one edge, add another
            if not improved and time.time() < deadline - 10:
                existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                                if (adj[i] >> j) & 1]
                random.shuffle(existing_edges)
                
                for u1, v1 in existing_edges:
                    if time.time() >= deadline - 10:
                        break
                    # Try removing (u1,v1) and adding (u2,v2)
                    for u2, v2 in missing_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        # Tentatively remove (u1,v1)
                        adj[u1] &= ~(1 << v1)
                        adj[v1] &= ~(1 << u1)
                        
                        # Check if (u2,v2) can be added
                        if can_add_edge(adj, u2, v2):
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            improved = True
                            break
                        else:
                            # Rollback removal
                            adj[u1] |= (1 << v1)
                            adj[v1] |= (1 << u1)
                    
                    if improved:
                        # Rebuild missing_edges
                        missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                                       if not ((adj[i] >> j) & 1)]
                        break
        
        return adj
    
    best_edges = 0
    best_adj = None
    
    # First, do a greedy construction to get a good starting point
    adj = greedy_construct()
    adj = local_search(adj)
    ec = edge_count(adj)
    if ec > best_edges:
        best_edges = ec
        best_adj = adj[:]
    
    # Multiple random restarts
    while time.time() < deadline - 15:
        # Start with a random graph with moderate density
        adj = [0] * n
        p = random.uniform(0.2, 0.4)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Clean up: remove edges until valid
        changed = True
        while changed:
            changed = False
            for a, b, c in triples:
                common = adj[a] & adj[b] & adj[c]
                if common.bit_count() > 2:
                    common_list = [i for i in range(n) if (common >> i) & 1]
                    v = random.choice(common_list)
                    adj[a] &= ~(1 << v)
                    adj[v] &= ~(1 << a)
                    changed = True
                    break
        
        # Local search
        adj = local_search(adj)
        ec = edge_count(adj)
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
    
    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
    
    return result
