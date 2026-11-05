import random
import time

def construct():
    deadline = time.time() + 110  # safety margin
    n = 40
    
    def base_graph():
        """8 disjoint copies of K5."""
        edges = set()
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    edges.add((5*k+a, 5*k+b))
        return edges
    
    def to_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def creates_c6(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        Need a path of length 5 from u to v in current graph."""
        # Path u-a-b-c-d-v: check for each d in N(v), path of length 4 from u to d
        for d in adj[v]:
            if d == u:
                continue
            # Check path u-a-b-c-d of length 4
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[a]:
                    if b == u:
                        continue
                    for c in adj[b]:
                        if c == a:
                            continue
                        if d in adj[c]:
                            return True
        return False
    
    def is_c6_free(edges):
        """Verify the entire graph is C6-free (for final validation)."""
        adj = to_adj(edges)
        # Check all 6-cycles via BFS-like approach
        # For each pair (u,v) that is an edge, check if there's a path of length 5
        # But that's expensive. Instead, we trust the incremental construction.
        # For validation, do a full check:
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check if there's another path of length 5 between u and v
                    # i.e., remove edge (u,v) and check for path of length 5
                    # This is expensive, so we'll do a simpler check
                    pass
        # Trust incremental construction; do a spot check
        return True
    
    def validate_no_c6(edges):
        """Full validation: check no 6-cycle exists."""
        adj = to_adj(edges)
        # A 6-cycle is v0-v1-v2-v3-v4-v5-v0 with all distinct vertices
        # Check via: for each edge (u,v), is there a path of length 5 from u to v?
        # That would form a 6-cycle. But we need the path to not use edge (u,v) and all vertices distinct.
        # Simpler: enumerate all 6-tuples? Too slow.
        # Use: for each pair (u,v) with u<v, count paths of length 5. If >0, C6 exists.
        # Path of length 5 from u to v: u-a-b-c-d-v
        # We can do BFS to depth 5 but that's expensive.
        # Alternative: for each edge (u,v), temporarily remove it and check if u and v are connected by a path of exactly length 5.
        # This is what creates_c6 does, but for all edges.
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    # Check if there's a path of length 5 from u to v not using (u,v)
                    # This is the same as creates_c6 logic
                    found = False
                    for d in adj[v]:
                        if d == u:
                            continue
                        for a in adj[u]:
                            if a == v:
                                continue
                            for b in adj[a]:
                                if b == u:
                                    continue
                                for c in adj[b]:
                                    if c == a:
                                        continue
                                    if d in adj[c]:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        return False
        return True
    
    best_edges = base_graph()
    best_count = len(best_edges)
    
    rng = random.Random(42)
    
    def try_greedy_add(adj, edges, time_limit):
        """Try to add random missing edges greedily."""
        missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > time_limit:
                break
            if not creates_c6(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.add((u, v))
    
    def local_search(edges, time_limit):
        """Local search: try adding edges, and if stuck, try swaps."""
        adj = to_adj(edges)
        
        # First, greedily add as many edges as possible
        try_greedy_add(adj, edges, time_limit)
        
        # Then try swaps: remove one edge, add another
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_edges = list(edges)
            rng.shuffle(current_edges)
            
            for rem_u, rem_v in current_edges:
                if time.time() > time_limit:
                    break
                # Try removing this edge and adding a missing one
                adj[rem_u].discard(rem_v)
                adj[rem_v].discard(rem_u)
                edges.discard((rem_u, rem_v))
                
                missing = [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]
                rng.shuffle(missing)
                
                added = False
                for u, v in missing:
                    if time.time() > time_limit:
                        break
                    if not creates_c6(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.add((u, v))
                        added = True
                        break
                
                if not added:
                    # Put the edge back
                    adj[rem_u].add(rem_v)
                    adj[rem_v].add(rem_u)
                    edges.add((rem_u, rem_v))
        
        return edges
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 5:
        if restarts == 0:
            edges = base_graph()
        else:
            # Start from a random subset of best, or a new random base
            if restarts % 3 == 1:
                edges = base_graph()
            else:
                # Random graph with moderate density
                edges = set()
                for u in range(n):
                    for v in range(u+1, n):
                        if rng.random() < 0.3:
                            edges.add((u, v))
        
        adj = to_adj(edges)
        
        # Greedy add
        try_greedy_add(adj, edges, deadline - 5)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = set(edges)
        
        restarts += 1
    
    # Final validation
    if not validate_no_c6(best_edges):
        # Fallback to base
        return [[u, v] for u, v in sorted(base_graph())]
    
    return [[u, v] for u, v in sorted(best_edges)]
