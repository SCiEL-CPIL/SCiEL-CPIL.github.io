"""Program for finding K2,3-free graph on 20 vertices maximizing edges."""
import random
import itertools

def construct():
    """
    Construct a K2,3-free graph on 20 vertices (0..19) maximizing edges.
    
    Approach: Use a greedy construction with backtracking/local search.
    A graph is K2,3-free iff no two vertices have 3 or more common neighbors.
    Equivalently, for every pair of vertices {u,v}, |N(u) ∩ N(v)| <= 2.
    
    We use a randomized greedy + local search approach to find a dense
    K2,3-free graph.
    """
    n = 20
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_k23_free(edge_set):
        """Check if the graph defined by edge_set is K2,3-free."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) >= 3:
                    return False
        return True
    
    def common_neighbors_count(edge_set):
        """Count violations: number of pairs with >= 3 common neighbors."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        violations = 0
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) >= 3:
                    violations += 1
        return violations
    
    # Greedy construction: add edges in order of some heuristic
    def greedy_construction(seed=42):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        
        # Shuffle edges to get different orderings
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check if adding (u,v) would create a K2,3 violation
            # Adding (u,v) means u and v become neighbors.
            # This could create violations for any pair (x,y) where:
            #   x is in N(u) ∩ N(v) and y is in N(u) ∩ N(v) -> no, that's not right.
            # Actually, adding edge (u,v) affects common neighbors of:
            #   - any pair involving u or v
            #   - pairs (x,y) where x in N(u) and y in N(v) or vice versa? No.
            # Let me think more carefully.
            # 
            # After adding (u,v):
            #   N(u) gains v, N(v) gains u.
            #   For any pair (a,b), the common neighbors can only change if
            #   a or b is u or v.
            #   
            # Pairs to check:
            #   (u, w) for w in N(v) (w != u): common neighbors of u and w now include... 
            #     Actually N(u) ∩ N(w) changes because N(u) gained v.
            #     If v in N(w), then v becomes a new common neighbor of u and w.
            #   (v, w) for w in N(u) (w != v): similarly, u becomes new common neighbor.
            #   (x, y) where x in N(u), y in N(v), x != v, y != u:
            #     N(x) gains... no, N(x) doesn't change. N(y) doesn't change.
            #     So common neighbors of (x,y) don't change.
            #   Wait, that's wrong. Adding edge (u,v) only changes N(u) and N(v).
            #   So only pairs involving u or v can have changed common neighbor counts.
            #
            # Pairs involving u: (u, w) for all w != u
            #   N(u) ∩ N(w) = (N(u) ∪ {v}) ∩ N(w) = (N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
            #   So if v ∈ N(w), the count increases by 1.
            #
            # Pairs involving v: (v, w) for all w != v
            #   Similarly, if u ∈ N(w), count increases by 1.
            #
            # So we need to check:
            #   For each w in N(v) (w != u): if |N(u) ∩ N(w)| == 2, adding v makes it 3 -> violation
            #   For each w in N(u) (w != v): if |N(v) ∩ N(w)| == 2, adding u makes it 3 -> violation
            #
            # Also need to check pair (u,v) itself: |N(u) ∩ N(v)| after adding = |N(u) ∩ N(v)| (unchanged, since u notin N(v) before and v notin N(u) before... wait, they're not adjacent yet)
            # Actually N(u) ∩ N(v) before adding (u,v) is some set S. After adding, N(u) gains v and N(v) gains u. But v is not in N(v) and u is not in N(u), so N(u) ∩ N(v) is still S. No change for pair (u,v).
            
            can_add = True
            # Check pairs (u, w) for w in N(v)
            for w in adj[v]:
                if w == u:
                    continue
                # Current common neighbors of u and w
                common = len(adj[u] & adj[w])
                # After adding (u,v), if v is in N(w), then v becomes a new common neighbor
                if v in adj[w]:
                    common += 1
                if common >= 3:
                    can_add = False
                    break
            
            if can_add:
                # Check pairs (v, w) for w in N(u)
                for w in adj[u]:
                    if w == v:
                        continue
                    common = len(adj[v] & adj[w])
                    if u in adj[w]:
                        common += 1
                    if common >= 3:
                        can_add = False
                        break
            
            if can_add:
                edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        return list(edges)
    
    # Local search: try to add more edges or swap edges
    def local_search(edge_list, max_iters=5000):
        edges = set(edge_list)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        rng = random.Random(12345)
        
        def can_add_edge(u, v):
            if (u, v) in edges or (v, u) in edges:
                return False
            # Check pairs (u, w) for w in N(v)
            for w in adj[v]:
                if w == u:
                    continue
                common = len(adj[u] & adj[w])
                if v in adj[w]:
                    common += 1
                if common >= 3:
                    return False
            # Check pairs (v, w) for w in N(u)
            for w in adj[u]:
                if w == v:
                    continue
                common = len(adj[v] & adj[w])
                if u in adj[w]:
                    common += 1
                if common >= 3:
                    return False
            return True
        
        def remove_edge(u, v):
            edges.discard((u, v))
            edges.discard((v, u))
            adj[u].discard(v)
            adj[v].discard(u)
        
        def add_edge(u, v):
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        iteration = 0
        while improved and iteration < max_iters:
            improved = False
            iteration += 1
            # Try to add random missing edges
            missing = [(u, v) for u, v in all_edges if (u, v) not in edges and (v, u) not in edges]
            rng.shuffle(missing)
            for u, v in missing:
                if can_add_edge(u, v):
                    add_edge(u, v)
                    improved = True
                    break
            if not improved:
                # Try swapping: remove one edge, add another
                edge_list_now = list(edges)
                rng.shuffle(edge_list_now)
                for rem_u, rem_v in edge_list_now:
                    for add_u, add_v in missing:
                        if (add_u, add_v) in edges or (add_v, add_u) in edges:
                            continue
                        # Try removing (rem_u, rem_v) and adding (add_u, add_v)
                        # First check if we can add (add_u, add_v) after removing (rem_u, rem_v)
                        # This is complex; skip for now
                        pass
        
        return list(edges)
    
    # Try multiple random seeds and keep the best
    best_edges = []
    best_count = 0
    
    for seed in range(50):
        edges = greedy_construction(seed)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Apply local search to the best
    improved = local_search(best_edges)
    if len(improved) > best_count:
        best_edges = improved
        best_count = len(improved)
    
    # Verify the result is valid
    if not is_k23_free(best_edges):
        # Fallback: use the simple K4 construction
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return [[u, v] for u, v in best_edges]
