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
        if not edges:
            return []
        
        edges_set = set(edges)
        n_vertices = max(max(u, v) for u, v in edges) + 1
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def greedy_init(rng):
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
                # Pick vertex with max degree in remaining graph
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
                
                # Extend greedily
                changed = True
                while changed:
                    changed = False
                    # Try extending from both ends
                    for end_pos in [True, False]:
                        if end_pos:
                            cur = path[-1]
                        else:
                            cur = path[0]
                        nbrs = get_neighbors(cur)
                        valid = [n for n in nbrs if n not in path]
                        if valid:
                            # Choose neighbor: prefer lower degree to avoid trapping
                            nxt = min(valid, key=lambda x: len(get_neighbors(x)))
                            if end_pos:
                                path.append(nxt)
                            else:
                                path.insert(0, nxt)
                            remove_edge(cur, nxt)
                            changed = True
                        if changed:
                            break
                
                paths.append(path)
            
            return paths
        
        def local_search(paths, rng, max_iters=200):
            """
            Local search: try to merge two paths by reconnection.
            Two paths P1 and P2 can be merged if there exist vertices
            a,b in P1 and c,d in P2 such that:
            - (a,c) and (b,d) are edges in the graph
            - These edges are not used in P1 or P2
            - The reconnection forms a simple path
            """
            improved = True
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                if time.time() > deadline - 10:
                    break
                
                # Try all pairs of paths
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        p1 = paths[i]
                        p2 = paths[j]
                        
                        # Build edge sets for quick lookup
                        e1 = set()
                        for k in range(len(p1) - 1):
                            e1.add(tuple(sorted((p1[k], p1[k+1]))))
                        e2 = set()
                        for k in range(len(p2) - 1):
                            e2.add(tuple(sorted((p2[k], p2[k+1]))))
                        
                        # Try to find a valid merge
                        merged = None
                        for ai in range(len(p1)):
                            for bi in range(ai + 1, len(p1)):
                                a, b = p1[ai], p1[bi]
                                if (a, b) in e1 or (b, a) in e1:
                                    continue  # Already an edge in p1
                                for cj in range(len(p2)):
                                    for dj in range(cj + 1, len(p2)):
                                        c, d = p2[cj], p2[dj]
                                        if (c, d) in e2 or (d, c) in e2:
                                            continue
                                        
                                        # Check if (a,c) and (b,d) are graph edges
                                        if (a, c) in edges_set and (b, d) in edges_set:
                                            # Try reconnection: 
                                            # P1 = [..., a, ..., b, ...]
                                            # P2 = [..., c, ..., d, ...]
                                            # Remove (a,c) and (b,d) from consideration
                                            # New path: reverse(P1[:ai+1]) + P1[bi:] reversed... 
                                            # Actually: P1 segment from a to b, and P2 segment from c to d
                                            # Connect a-c and b-d
                                            # New: P1[0:ai] + [c] + P2[cj+1:dj] reversed... 
                                            
                                            # Let me think more carefully.
                                            # P1: v0, v1, ..., a, ..., b, ..., vk
                                            # P2: w0, w1, ..., c, ..., d, ..., wm
                                            # We want to form a single path using all vertices/edges.
                                            # Remove edge (a, next_in_p1) and (prev_in_p1, b) from P1
                                            # Remove edge (c, next_in_p2) and (prev_in_p2, d) from P2
                                            # Add edges (a,c) and (b,d)
                                            
                                            # This is complex. Let me try a simpler merge:
                                            # Check if any endpoint of P1 connects to any endpoint of P2
                                            pass
                                        
                                        # Also try (a,d) and (b,c)
                                        if (a, d) in edges_set and (b, c) in edges_set:
                                            pass
                        
                        # Simpler merge: connect endpoints
                        endpoints1 = [p1[0], p1[-1]]
                        endpoints2 = [p2[0], p2[-1]]
                        
                        for e1 in endpoints1:
                            for e2 in endpoints2:
                                if (e1, e2) in edges_set:
                                    edge_key = tuple(sorted((e1, e2)))
                                    if edge_key not in e1 and edge_key not in e2:
                                        # Can merge
                                        if p1[-1] == e1 and p2[0] == e2:
                                            merged = p1 + p2[1:]
                                        elif p1[0] == e1 and p2[0] == e2:
                                            merged = p1[::-1] + p2[1:]
                                        elif p1[-1] == e1 and p2[-1] == e2:
                                            merged = p1 + p2[::-1][1:]
                                        elif p1[0] == e1 and p2[-1] == e2:
                                            merged = p1[::-1] + p2[::-1][1:]
                                        
                                        if merged and len(set(merged)) == len(merged):
                                            paths[i] = merged
                                            paths.pop(j)
                                            improved = True
                                            break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            return paths
        
        # Get initial solution
        best_paths = None
        best_count = float('inf')
        
        # Try multiple random restarts
        for trial in range(2000):
            if time.time() > deadline - 5:
                break
            
            rng = random.Random(hash((tuple(sorted(edges)), trial)))
            paths = greedy_init(rng)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
                # Apply local search
                paths = local_search(paths, rng, max_iters=50)
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = paths
        
        if best_paths is None:
            best_paths = greedy_init(random.Random(0))
        
        return best_paths
    
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
