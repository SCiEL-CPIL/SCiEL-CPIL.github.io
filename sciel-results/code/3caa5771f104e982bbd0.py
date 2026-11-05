import time
import random
from collections import defaultdict, deque

def construct():
    """
    Decomposes the edges of three fixed instance graphs into the minimum number of
    edge-disjoint simple paths.
    
    Returns:
        A list of three lists of paths. Each path is a list of vertices.
    """
    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        """
        Minimum path decomposition via greedy construction followed by iterative merging.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        n_vertices = max(max(u, v) for u, v in edge_set) + 1
        
        # Build adjacency
        adj = defaultdict(list)
        for (u, v) in edge_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # Compute degrees and odd vertices
        degree = {v: len(adj[v]) for v in range(n_vertices)}
        odd_vertices = [v for v in range(n_vertices) if degree[v] % 2 == 1]
        lower_bound = max(1, len(odd_vertices) // 2) if edge_set else 0

        def greedy_decompose(edge_set, adj, rng, deadline):
            """
            One greedy pass: build maximal simple paths.
            """
            remaining = set(edge_set)
            paths = []
            
            # Build remaining adjacency
            rem_adj = defaultdict(list)
            for u, v in remaining:
                rem_adj[u].append(v)
                rem_adj[v].append(u)
            
            def rem_degree(v):
                return len(rem_adj[v]) if v in rem_adj else 0
            
            while remaining and time.time() < deadline:
                # Choose start vertex: prefer odd-degree in remaining graph
                rem_odd = [u for u in rem_adj if rem_degree(u) % 2 == 1]
                if rem_odd:
                    candidates = sorted(rem_odd, key=lambda u: -rem_degree(u))[:3]
                    start = rng.choice(candidates)
                else:
                    candidates = [u for u in rem_adj if rem_degree(u) > 0]
                    if not candidates:
                        break
                    candidates = sorted(candidates, key=lambda u: -rem_degree(u))[:3]
                    start = rng.choice(candidates)
                
                # Build a maximal simple path
                path = [start]
                in_path = {start}
                current = start
                
                while True:
                    neighbors = [nb for nb in rem_adj[current] 
                                 if (min(current, nb), max(current, nb)) in remaining]
                    if not neighbors:
                        break
                    
                    non_cycle = [nb for nb in neighbors if nb not in in_path]
                    if non_cycle:
                        non_cycle.sort(key=lambda v: -rem_degree(v))
                        if rng.random() < 0.3 and len(non_cycle) > 1:
                            next_v = non_cycle[rng.randrange(1, min(3, len(non_cycle)))]
                        else:
                            next_v = non_cycle[0]
                    else:
                        break
                    
                    key = (min(current, next_v), max(current, next_v))
                    remaining.discard(key)
                    path.append(next_v)
                    in_path.add(next_v)
                    
                    if next_v in rem_adj[current]:
                        rem_adj[current].remove(next_v)
                    if current in rem_adj[next_v]:
                        rem_adj[next_v].remove(current)
                    if not rem_adj[current]:
                        del rem_adj[current]
                    if not rem_adj[next_v]:
                        del rem_adj[next_v]
                    
                    current = next_v
                
                if len(path) > 1:
                    paths.append(path)
            
            # Handle any leftover edges
            while remaining and time.time() < deadline:
                e = next(iter(remaining))
                u, v = e
                paths.append([u, v])
                remaining.discard(e)
                if v in rem_adj.get(u, []):
                    rem_adj[u].remove(v)
                if u in rem_adj.get(v, []):
                    rem_adj[v].remove(u)
                if u in rem_adj and not rem_adj[u]:
                    del rem_adj[u]
                if v in rem_adj and not rem_adj[v]:
                    del rem_adj[v]
            
            return paths

        def merge_paths(paths, adj, deadline):
            """
            Iteratively merge paths if they share an endpoint and the union is simple.
            """
            if len(paths) <= 1:
                return paths
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_paths = []
                used = [False] * len(paths)
                
                for i in range(len(paths)):
                    if used[i]:
                        continue
                    
                    path_i = paths[i]
                    start_i = path_i[0]
                    end_i = path_i[-1]
                    
                    merged = False
                    for j in range(len(paths)):
                        if j == i or used[j]:
                            continue
                        
                        path_j = paths[j]
                        start_j = path_j[0]
                        end_j = path_j[-1]
                        
                        # Option 1: i ends at j's start
                        if end_i == start_j:
                            candidate = path_i + path_j[1:]
                            if len(set(candidate)) == len(candidate):
                                new_paths.append(candidate)
                                used[i] = True
                                used[j] = True
                                merged = True
                                changed = True
                                break
                        
                        # Option 2: i ends at j's end
                        if end_i == end_j:
                            candidate = path_i + list(reversed(path_j[1:]))
                            if len(set(candidate)) == len(candidate):
                                new_paths.append(candidate)
                                used[i] = True
                                used[j] = True
                                merged = True
                                changed = True
                                break
                        
                        # Option 3: i starts at j's start
                        if start_i == start_j:
                            candidate = list(reversed(path_i[1:])) + path_j
                            if len(set(candidate)) == len(candidate):
                                new_paths.append(candidate)
                                used[i] = True
                                used[j] = True
                                merged = True
                                changed = True
                                break

                        # Option 4: i starts at j's end
                        if start_i == end_j:
                            candidate = path_j + path_i[1:]
                            if len(set(candidate)) == len(candidate):
                                new_paths.append(candidate)
                                used[i] = True
                                used[j] = True
                                merged = True
                                changed = True
                                break

                    if not merged:
                        new_paths.append(path_i)
                        used[i] = True
                
                paths = new_paths
            
            return paths

        def validate_paths(paths, edge_set):
            """Validate that paths are simple, edge-disjoint, and cover all edges."""
            if not paths:
                return len(edge_set) == 0
            
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

        best_paths = None
        best_count = float('inf')
        
        max_restarts = 30
        for trial in range(max_restarts):
            if time.time() > deadline - 2:
                break
            
            rng = random.Random(trial * 7919 + 13)
            raw_paths = greedy_decompose(edge_set, adj, rng, deadline)
            
            # Merge
            merged_paths = merge_paths(raw_paths, adj, deadline)
            
            count = len(merged_paths)
            if count < best_count:
                best_count = count
                best_paths = merged_paths
                
            if best_count <= lower_bound:
                break
        
        if best_paths is None or not validate_paths(best_paths, edge_set):
            # Fallback: each edge as a path
            best_paths = [[u, v] for u, v in edge_set]
        
        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = decompose(edges)
        
        # Validate
        edge_set = set(tuple(sorted(e)) for e in edges)
        if not validate_paths(paths, edge_set):
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
