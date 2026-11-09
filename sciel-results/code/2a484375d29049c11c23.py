import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_optimal(edges, n, time_limit=20.0):
        """
        Decompose edges into minimum number of edge-disjoint simple paths.
        Uses odd-degree vertex pairing + greedy path extension + local search.
        """
        if not edges:
            return []
        
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        total_edges = len(edge_set)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        def get_degrees(remaining):
            deg = [0] * n
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            return deg
        
        def find_path(start, end, remaining, adj, max_len=None):
            """Find a simple path from start to end using only edges in remaining."""
            if start == end:
                return [start]
            
            # BFS with parent tracking
            from collections import deque
            parent = {start: None}
            queue = deque([start])
            
            while queue:
                curr = queue.popleft()
                if curr == end:
                    # Reconstruct path
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                
                for neighbor in adj[curr]:
                    if neighbor not in parent:
                        e_key = (min(curr, neighbor), max(curr, neighbor))
                        if e_key in remaining:
                            parent[neighbor] = curr
                            queue.append(neighbor)
            
            return None
        
        def find_any_path(start, remaining, adj, max_len=None):
            """Find a simple path starting from start, trying to be long."""
            if max_len is None:
                max_len = len(remaining) + 1
            
            # DFS with backtracking, try to extend as far as possible
            best_path = [start]
            
            def dfs(current, visited, path, remaining_local):
                nonlocal best_path
                if len(path) > len(best_path):
                    best_path = path[:]
                
                if len(path) >= max_len:
                    return
                
                # Try neighbors in order of fewest remaining options (Warnsdorff)
                candidates = []
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        e_key = (min(current, neighbor), max(current, neighbor))
                        if e_key in remaining_local:
                            # Count options for neighbor
                            options = 0
                            for nb in adj[neighbor]:
                                if nb not in visited:
                                    e2 = (min(neighbor, nb), max(neighbor, nb))
                                    if e2 in remaining_local:
                                        options += 1
                            candidates.append((options, neighbor, e_key))
                
                if not candidates:
                    return
                
                candidates.sort(key=lambda x: x[0])
                
                for _, neighbor, e_key in candidates:
                    new_visited = visited | {neighbor}
                    new_remaining = remaining_local - {e_key}
                    new_path = path + [neighbor]
                    dfs(neighbor, new_visited, new_path, new_remaining)
                    # Early termination if we found a path using all edges
                    if len(new_path) - 1 == len(remaining_local):
                        return
            
            dfs(start, {start}, [start], remaining)
            return best_path
        
        # Phase 1: Greedy decomposition using odd-degree vertex pairing
        remaining = set(edge_set)
        paths = []
        
        while remaining:
            deg = get_degrees(remaining)
            odd_vertices = [i for i in range(n) if deg[i] % 2 == 1 and deg[i] > 0]
            
            if odd_vertices:
                # Pick two odd vertices and find a path between them
                # Choose the pair that gives the longest path
                best_path = None
                best_len = -1
                # Limit pairs to check for efficiency
                if len(odd_vertices) > 10:
                    # Sample some pairs
                    import itertools
                    sample = random.sample(odd_vertices, min(10, len(odd_vertices)))
                    pairs = list(itertools.combinations(sample, 2))
                else:
                    pairs = list(__import__('itertools').combinations(odd_vertices, 2))
                
                for u, v in pairs:
                    path = find_path(u, v, remaining, adj)
                    if path and len(path) > best_len:
                        best_len = len(path)
                        best_path = path
                
                if best_path is None:
                    # Fallback: find any path from an odd vertex
                    u = odd_vertices[0]
                    best_path = find_any_path(u, remaining, adj)
                elif best_len < 2:
                    # Path too short, try any path
                    u = odd_vertices[0]
                    best_path = find_any_path(u, remaining, adj)
            else:
                # All even degrees: find a long path
                # Pick a vertex with highest degree
                start = max(range(n), key=lambda x: deg[x]) if any(deg) else 0
                best_path = find_any_path(start, remaining, adj)
            
            if best_path is None or len(best_path) < 2:
                # Fallback: single edge
                u, v = next(iter(remaining))
                best_path = [u, v]
            
            # Remove edges from path
            for i in range(len(best_path) - 1):
                e_key = (min(best_path[i], best_path[i+1]), max(best_path[i], best_path[i+1]))
                remaining.discard(e_key)
            
            paths.append(best_path)
        
        # Phase 2: Local search to improve (merge/swap paths to reduce count)
        # Try to merge two paths into one if possible
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    # Try to merge paths[i] and paths[j]
                    # Check if we can connect end of one to start of other
                    p1 = paths[i]
                    p2 = paths[j]
                    
                    # Try all 4 combinations: p1+p2, p1+reversed(p2), reversed(p1)+p2, reversed(p1)+reversed(p2)
                    merged = None
                    for rev1 in [False, True]:
                        for rev2 in [False, True]:
                            q1 = p1[::-1] if rev1 else p1
                            q2 = p2[::-1] if rev2 else p2
                            # Check if end of q1 connects to start of q2
                            if q1[-1] == q2[0]:
                                # Check if all vertices in q2 are not in q1 (simple path)
                                if not any(v in q1 for v in q2[1:]):
                                    merged = q1 + q2[1:]
                                    break
                        if merged:
                            break
                    if merged:
                        # Validate merged path uses only original edges
                        valid = True
                        for k in range(len(merged) - 1):
                            e_key = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                            if e_key not in edge_set:
                                valid = False
                                break
                        if valid:
                            paths = paths[:i] + paths[i+1:j] + paths[j+1:] + [merged]
                            improved = True
                            break
                if improved:
                    break
        
        return paths
    
    def validate_and_fix(paths, edges, n):
        """Validate paths and fix any issues."""
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            if len(path) < 2:
                continue
            valid = True
            path_edges = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e_key = (min(u, v), max(u, v))
                if e_key not in edge_set:
                    valid = False
                    break
                if e_key in used_edges:
                    valid = False
                    break
                path_edges.append(e_key)
            
            if valid:
                used_edges.update(path_edges)
                valid_paths.append(path)
            else:
                for e in path_edges:
                    if e not in used_edges:
                        used_edges.add(e)
                        valid_paths.append([e[0], e[1]])
        
        for e in edge_set:
            if e not in used_edges:
                used_edges.add(e)
                valid_paths.append([e[0], e[1]])
        
        return valid_paths
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        n = max(max(e) for e in edges) + 1 if edges else 0
        
        paths = decompose_optimal(edges, n, time_limit=40.0)
        paths = validate_and_fix(paths, edges, n)
        
        results.append(paths)
    
    return results
