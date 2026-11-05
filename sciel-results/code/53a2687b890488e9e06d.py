import time
import random
from collections import defaultdict

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        """
        Decompose edges into minimum number of edge-disjoint simple paths.
        Uses a matching-based approach:
        1. Find an initial path decomposition
        2. Iteratively try to merge paths using a maximum matching formulation
        """
        if not edges:
            return []
        
        edges_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Initial solution: greedy with multiple random restarts
        def greedy_init(rng=None):
            if rng is None:
                rng = random.Random()
            
            remaining = set(edges)
            paths = []
            
            def get_neighbors(u):
                return [v for e in remaining if e[0] == u for v in [e[1]] if v != u] + \
                       [v for e in remaining if e[1] == u for v in [e[0]] if v != u]
            
            def remove_edge(u, v):
                if (u, v) in remaining:
                    remaining.remove((u, v))
                else:
                    remaining.remove((v, u))
            
            while remaining:
                # Pick vertex with max degree
                degs = defaultdict(int)
                for u, v in remaining:
                    degs[u] += 1
                    degs[v] += 1
                max_deg = max(degs.values())
                candidates = [v for v, d in degs.items() if d == max_deg]
                start = rng.choice(candidates)
                
                neighbors = get_neighbors(start)
                if not neighbors:
                    continue
                end = rng.choice(neighbors)
                
                path = [start, end]
                remove_edge(start, end)
                
                # Extend
                changed = True
                while changed:
                    changed = False
                    for pos in [-1, 0]:  # extend from end or start
                        if pos == -1:
                            cur = path[-1]
                        else:
                            cur = path[0]
                        nbrs = get_neighbors(cur)
                        valid = [n for n in nbrs if n not in path]
                        if valid:
                            # Choose neighbor that leads to fewer options (dead-end preference)
                            nxt = min(valid, key=lambda x: len(get_neighbors(x)))
                            if pos == -1:
                                path.append(nxt)
                            else:
                                path.insert(0, nxt)
                            remove_edge(cur, nxt)
                            changed = True
                
                paths.append(path)
            
            return paths
        
        # Try multiple random restarts to get a good initial solution
        best_paths = None
        best_count = float('inf')
        
        for trial in range(100):
            if time.time() > deadline - 10:
                break
            rng = random.Random(hash((tuple(edges), trial)))
            paths = greedy_init(rng)
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
        
        if best_paths is None:
            best_paths = greedy_init()
            best_count = len(best_paths)
        
        # Now try to improve by merging paths
        # Two paths can be merged if they can be reconnected to form a single simple path
        # This is equivalent to finding a matching in an auxiliary graph
        
        def path_edges(path):
            """Get set of edges in a path."""
            return {(min(path[i], path[i+1]), max(path[i], path[i+1])) for i in range(len(path)-1)}
        
        def try_merge(paths):
            """
            Try to merge two paths into one.
            Returns (new_paths, merged) where merged is True if a merge was successful.
            """
            n = len(paths)
            if n < 2:
                return paths, False
            
            # For each pair of paths, check if they can be merged
            # Two paths P1 and P2 can be merged if there exist vertices
            # a, b in P1 and c, d in P2 such that:
            # - (a,c) and (b,d) are edges not in P1 or P2
            # - The reconnection forms a simple path
            # 
            # Simpler approach: try to connect an endpoint of P1 to an endpoint of P2
            # via an unused edge, then check if the result is a simple path
            
            for i in range(n):
                for j in range(i+1, n):
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    edges_p1 = path_edges(p1)
                    edges_p2 = path_edges(p2)
                    
                    # Try all combinations of connecting endpoints
                    endpoints1 = [p1[0], p1[-1]]
                    endpoints2 = [p2[0], p2[-1]]
                    
                    for e1 in endpoints1:
                        for e2 in endpoints2:
                            edge_key = (min(e1, e2), max(e1, e2))
                            if edge_key in edges_set and edge_key not in edges_p1 and edge_key not in edges_p2:
                                # Check if we can merge
                                # Determine how to combine
                                if p1[-1] == e1 and p2[0] == e2:
                                    new_path = p1 + p2[1:]
                                elif p1[0] == e1 and p2[0] == e2:
                                    new_path = p1[::-1] + p2[1:]
                                elif p1[-1] == e1 and p2[-1] == e2:
                                    new_path = p1 + p2[::-1][1:]
                                elif p1[0] == e1 and p2[-1] == e2:
                                    new_path = p1[::-1] + p2[::-1][1:]
                                else:
                                    continue
                                
                                # Check if new path is simple
                                if len(set(new_path)) == len(new_path):
                                    # Verify all edges are valid
                                    valid = True
                                    for k in range(len(new_path) - 1):
                                        ek = (min(new_path[k], new_path[k+1]), max(new_path[k], new_path[k+1]))
                                        if ek not in edges_set:
                                            valid = False
                                            break
                                    if valid:
                                        # Success! Replace p1 and p2 with new_path
                                        new_paths = [paths[k] for k in range(n) if k != i and k != j]
                                        new_paths.append(new_path)
                                        return new_paths, True
            
            return paths, False
        
        # Iteratively try to merge paths
        current_paths = best_paths
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            current_paths, improved = try_merge(current_paths)
        
        # Also try a more aggressive approach: use matching to find multiple merges at once
        # But first, let's see if we can do better with a different strategy
        
        # Alternative: use the fact that we can pair up "half-edges" at each vertex
        # This is equivalent to finding a maximum matching in a multigraph
        
        def decompose_via_matching():
            """
            Decompose edges into minimum paths by finding a maximum matching
            in the line graph (or an auxiliary graph).
            
            The key insight: at each vertex v of degree d, we need to decide
            how to pair up the d incident edges. Each pair forms a "through"
            connection, and unpaired edges are path endpoints.
            
            The number of path endpoints at v is d - 2*(number of pairs at v).
            Total path endpoints = sum over all v of (d(v) - 2*pairs(v))
            Number of paths = total_path_endpoints / 2
            
            To minimize paths, we maximize total pairs.
            This is equivalent to finding a maximum matching in a graph where
            each vertex v is represented by d(v) "ports" (one per incident edge),
            and we can match ports at the same vertex.
            """
            # Build the auxiliary graph
            # Each edge (u,v) in the original graph has two endpoints: one at u, one at v
            # At each vertex, we can pair up incident edges
            # This is equivalent to finding a maximum matching in a graph where:
            # - Nodes are the "half-edges" (edge, vertex) pairs
            # - Two half-edges can be matched if they share a vertex
            
            # Actually, this is simpler: at each vertex, we just need to decide
            # how many pairs to form. The constraint is that the resulting
            # structure must decompose into simple paths (no cycles).
            
            # For now, use a greedy approach: at each vertex, pair up as many
            # edges as possible, but avoid creating cycles.
            
            # This is complex, so let's use a different approach:
            # Start with all edges as separate paths, then merge using matching.
            
            # Each edge is a path of length 1
            paths = [[u, v] for u, v in edges]
            
            # Build a graph where nodes are path endpoints
            # Two endpoints can be connected if there's an edge between them
            # that's not already used
            
            used_edges = set()
            for u, v in edges:
                used_edges.add((min(u,v), max(u,v)))
            
            # Actually, let's just use the iterative merge approach with more iterations
            return paths
        
        # Use the iterative merge approach, but with more aggressive searching
        # Try all pairs of paths, not just endpoints
        
        def try_merge_internal(paths):
            """
            Try to merge two paths by reconnecting internal edges.
            Path P1: [a, b, c, d, e]
            Path P2: [f, g, h, i, j]
            Remove edges (b,c) from P1 and (g,h) from P2
            Add edges (c,g) and (b,h) if they exist
            New path: [a, b, h, i, j] + [f, g, c, d, e] reversed appropriately
            """
            n = len(paths)
            if n < 2:
                return paths, False
            
            for i in range(n):
                for j in range(i+1, n):
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    
                    edges_p1 = path_edges(p1)
                    edges_p2 = path_edges(p2)
                    
                    # Try all pairs of internal edges
                    for k in range(len(p1) - 1):
                        a, b = p1[k], p1[k+1]
                        for l in range(len(p2) - 1):
                            c, d = p2[l], p2[l+1]
                            
                            # Option 1: connect b-c and d-a
                            edge_bc = (min(b,c), max(b,c))
                            edge_da = (min(d,a), max(d,a))
                            if edge_bc in edges_set and edge_da in edges_set:
                                if edge_bc not in edges_p1 and edge_bc not in edges_p2:
                                    if edge_da not in edges_p1 and edge_da not in edges_p2:
                                        # Form new path
                                        # P1: ...a, b...  P2: ...c, d...
                                        # Remove (a,b) and (c,d), add (b,c) and (d,a)
                                        # New: ...a, d, ...P2 from d onwards... , c, b, ...P1 from b onwards...
                                        # Wait, let me think about this more carefully
                                        # P1: p1[0], ..., p1[k]=a, p1[k+1]=b, ..., p1[-1]
                                        # P2: p2[0], ..., p2[l]=c, p2[l+1]=d, ..., p2[-1]
                                        # Remove edges (a,b) and (c,d)
                                        # Add edges (b,c) and (d,a)
                                        # New path: p1[0..k] + [b, c] + p2[l-1..0] reversed? No...
                                        # 
                                        # Actually: 
                                        # Segment 1: p1[0] to a (inclusive)
                                        # Segment 2: b to p1[-1]
                                        # Segment 3: p2[0] to c (inclusive)  
                                        # Segment 4: d to p2[-1]
                                        # 
                                        # Connect: segment1 - (a,d) - segment4 - ... - segment3 - (c,b) - segment2
                                        # New path: p1[0..k] + [d] + p2[l+1:] + p2[l-1::-1] + [b] + p1[k+2:]
                                        # Wait, that's not right either.
                                        # 
                                        # Let me just construct it directly:
                                        # p1_prefix = p1[:k+1]  # includes a
                                        # p1_suffix = p1[k+1:]   # starts with b
                                        # p2_prefix = p2[:l+1]  # includes c
                                        # p2_suffix = p2[l+1:]   # starts with d
                                        # 
                                        # New path: p1_prefix + p2_suffix + p2_prefix[::-1] + p1_suffix
                                        # Check: ...a, d, ..., c, b, ...
                                        # Edge (a,d) exists, edge (c,b) exists
                                        new_path = p1[:k+1] + p2[l+1:] + p2[:l+1][::-1] + p1[k+1:]
                                        
                                        # Check simplicity
                                        if len(set(new_path)) == len(new_path):
                                            # Verify all edges
                                            valid = True
                                            for m in range(len(new_path) - 1):
                                                em = (min(new_path[m], new_path[m+1]), max(new_path[m], new_path[m+1]))
                                                if em not in edges_set:
                                                    valid = False
                                                    break
                                            if valid:
                                                new_paths = [paths[x] for x in range(n) if x != i and x != j]
                                                new_paths.append(new_path)
                                                return new_paths, True
                    
                    # Option 2: connect a-d and b-c
                    for k in range(len(p1) - 1):
                        a, b = p1[k], p1[k+1]
                        for l in range(len(p2) - 1):
                            c, d = p2[l], p2[l+1]
                            
                            edge_ad = (min(a,d), max(a,d))
                            edge_bc = (min(b,c), max(b,c))
                            if edge_ad in edges_set and edge_bc in edges_set:
                                if edge_ad not in edges_p1 and edge_ad not in edges_p2:
                                    if edge_bc not in edges_p1 and edge_bc not in edges_p2:
                                        # New path: p1[:k] + p2[l+1:] + p2[:l+1][::-1] + p1[k+1:]
                                        # Wait: ...a, d, ..., c, b, ...
                                        # p1_prefix = p1[:k]  # up to before a
                                        # p1_suffix = p1[k+1:]  # starts with b
                                        # p2_prefix = p2[:l+1]  # includes c
                                        # p2_suffix = p2[l+1:]  # starts with d
                                        # New: p1[:k] + [a, d] + p2[l+2:] + ... no this is getting messy
                                        # 
                                        # Let me just try: p1[:k+1] + p2[l+1:] + p2[:l+1][::-1] + p1[k+1:]
                                        # Same as before but with different edge connections
                                        new_path = p1[:k+1] + p2[l+1:] + p2[:l+1][::-1] + p1[k+1:]
                                        
                                        if len(set(new_path)) == len(new_path):
                                            valid = True
                                            for m in range(len(new_path) - 1):
                                                em = (min(new_path[m], new_path[m+1]), max(new_path[m], new_path[m+1]))
                                                if em not in edges_set:
                                                    valid = False
                                                    break
                                            if valid:
                                                new_paths = [paths[x] for x in range(n) if x != i and x != j]
                                                new_paths.append(new_path)
                                                return new_paths, True
            
            return paths, False
        
        # Try both merge strategies
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            current_paths, improved = try_merge(current_paths)
            if not improved:
                current_paths, improved = try_merge_internal(current_paths)
        
        # Final validation
        def validate_paths(paths, edges):
            if not paths:
                return len(edges) == 0
            
            edge_set = set(edges)
            used_edges = set()
            
            for path in paths:
                if len(set(path)) != len(path):
                    return False
                
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set:
                        return False
                    if e in used_edges:
                        return False
                    used_edges.add(e)
            
            if used_edges != edge_set:
                return False
            
            return True
        
        if not validate_paths(current_paths, edges):
            # Fallback to best valid solution
            current_paths = best_paths
        
        return current_paths
    
    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(set(path)) != len(path):
                return False
            
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        if used_edges != edge_set:
            return False
        
        return True
    
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        # Validate
        if not validate_paths(paths, edges):
            # Fallback: each edge as a path
            paths = [[u, v] for u, v in edges]
        
        results.append(paths)
        
        if time.time() > deadline - 1:
            break
    
    while len(results) < 3:
        n, p, seed = specs[len(results)]
        edges = instance(n, p, seed)
        results.append([[u, v] for u, v in edges])
    
    return results

if __name__ == "__main__":
    res = construct()
    total = sum(len(r) for r in res)
    print(f"Total paths: {total}")
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(r)} paths")
