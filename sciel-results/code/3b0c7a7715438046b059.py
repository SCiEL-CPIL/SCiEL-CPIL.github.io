import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    N = 20
    # Adjacency as list of sets for fast lookup
    adj = [set() for _ in range(N)]
    
    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
    
    def has_k22_in_induced(vertex_set):
        """Check if induced subgraph on vertex_set contains a K2,2 (C4)."""
        vs = list(vertex_set)
        n = len(vs)
        if n < 4:
            return False
        # Check all pairs of pairs
        for i in range(n):
            for j in range(i+1, n):
                a, b = vs[i], vs[j]
                # Find common neighbors of a and b within vs
                common = adj[a] & adj[b] & vertex_set
                # Need at least 2 common neighbors (c, d) such that a-c, a-d, b-c, b-d all exist
                # common already means a-x and b-x exist for x in common
                # We just need |common| >= 2 and those are in vs
                if len(common) >= 2:
                    return True
        return False
    
    def can_add(u, v):
        """Check if adding edge (u,v) would create a K3,3."""
        if v in adj[u]:
            return False  # already exists
        cn = adj[u] & adj[v]
        # Check if induced subgraph on cn has a K2,2
        return not has_k22_in_induced(cn)
    
    def count_edges():
        return sum(len(s) for s in adj) // 2
    
    def is_valid():
        """Verify no K3,3 exists."""
        # For every pair (u,v), check that induced subgraph on CN(u,v) has no K2,2
        for u in range(N):
            for v in range(u+1, N):
                if v in adj[u]:
                    cn = adj[u] & adj[v]
                    if has_k22_in_induced(cn):
                        return False
        return True
    
    # Phase 1: Greedy construction
    # Start with empty graph, add edges in a smart order
    random.seed(42)
    
    # Try multiple random orderings and pick the best
    best_edges = 0
    best_adj = None
    
    for trial in range(50):
        if time.time() > DEADLINE:
            break
        adj_t = [set() for _ in range(N)]
        
        def add_edge_t(u, v):
            adj_t[u].add(v)
            adj_t[v].add(u)
        
        def can_add_t(u, v):
            if v in adj_t[u]:
                return False
            cn = adj_t[u] & adj_t[v]
            vs = list(cn)
            n = len(vs)
            if n < 4:
                return True
            for i in range(n):
                for j in range(i+1, n):
                    a, b = vs[i], vs[j]
                    common = adj_t[a] & adj_t[b] & cn
                    if len(common) >= 2:
                        return False
            return True
        
        # Random edge ordering
        edges = [(u, v) for u in range(N) for v in range(u+1, N)]
        random.shuffle(edges)
        
        added = 0
        for u, v in edges:
            if can_add_t(u, v):
                add_edge_t(u, v)
                added += 1
        
        if added > best_edges:
            best_edges = added
            best_adj = [s.copy() for s in adj_t]
    
    # Use the best greedy result
    adj = [s.copy() for s in best_adj] if best_adj else [set() for _ in range(N)]
    
    # Phase 2: Local search - try adding edges
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if v not in adj[u] and can_add(u, v):
                    add_edge(u, v)
                    improved = True
    
    # Phase 3: Local search - try swaps (remove one edge, add another)
    for _ in range(200):
        if time.time() > DEADLINE:
            break
        improved = False
        # Try to find a swap that increases edge count
        # Remove a random edge, then try to add two edges
        for attempt in range(1000):
            if time.time() > DEADLINE:
                break
            # Pick a random existing edge to potentially remove
            u = random.randint(0, N-1)
            if not adj[u]:
                continue
            v = random.choice(list(adj[u]))
            
            # Try removing (u,v) and adding two new edges
            remove_edge(u, v)
            
            added_count = 0
            added_edges = []
            for a in range(N):
                for b in range(a+1, N):
                    if b not in adj[a] and can_add(a, b):
                        add_edge(a, b)
                        added_edges.append((a, b))
                        added_count += 1
                        if added_count >= 2:
                            break
                if added_count >= 2:
                    break
            
            if added_count >= 2:
                improved = True
            else:
                # Revert
                for a, b in added_edges:
                    remove_edge(a, b)
                add_edge(u, v)
        
        if not improved:
            # Try single edge additions again
            for u in range(N):
                for v in range(u+1, N):
                    if v not in adj[u] and can_add(u, v):
                        add_edge(u, v)
                        improved = True
            if not improved:
                break
    
    # Final validation
    if not is_valid():
        # Fallback to a known valid graph
        adj = [set() for _ in range(N)]
        for a in range(2):
            for b in range(2, N):
                add_edge(a, b)
    
    result = []
    for u in range(N):
        for v in adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
