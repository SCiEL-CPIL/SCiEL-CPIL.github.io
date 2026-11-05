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
    deadline = start_time + 110.0  # Safety margin for 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def validate_paths(paths, edges):
        """Validate that paths are simple, edge-disjoint, and cover all edges."""
        if not paths:
            return len(edges) == 0
        
        edge_set = set(edges)
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            # Check simple path
            if len(set(path)) != len(path):
                return False
            
            # Check edges exist and are not reused
            for j in range(len(path) - 1):
                u, v = path[j], path[j+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set:
                    return False
                if e in used_edges:
                    return False
                used_edges.add(e)
        
        # Check all edges are covered
        if used_edges != edge_set:
            return False
        
        return True

    def max_matching_bipartite(left_nodes, right_nodes, adjacency):
        """
        Find maximum bipartite matching using Hopcroft-Karp algorithm.
        left_nodes, right_nodes: lists of node IDs
        adjacency: dict mapping left node -> set of right nodes
        Returns: list of (left, right) pairs in the matching
        """
        if not left_nodes or not right_nodes:
            return []
        
        # Build node index maps
        left_index = {node: i for i, node in enumerate(left_nodes)}
        right_index = {node: i for i, node in enumerate(right_nodes)}
        
        n_left = len(left_nodes)
        n_right = len(right_nodes)
        
        # Build adjacency list in indexed form
        adj = [[] for _ in range(n_left)]
        for u in left_nodes:
            ui = left_index[u]
            for v in adjacency.get(u, set()):
                if v in right_index:
                    adj[ui].append(right_index[v])
        
        # Hopcroft-Karp
        match_left = [-1] * n_left
        match_right = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if match_right[v] == -1:
                        found = True
                    elif dist[match_right[v]] == float('inf'):
                        dist[match_right[v]] = dist[u] + 1
                        queue.append(match_right[v])
            return found
        
        def dfs(u):
            for v in adj[u]:
                if match_right[v] == -1 or (dist[match_right[v]] == dist[u] + 1 and dfs(match_right[v])):
                    match_left[u] = v
                    match_right[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        pass
        
        matching = []
        for u in range(n_left):
            if match_left[u] != -1:
                matching.append((left_nodes[u], right_nodes[match_left[u]]))
        
        return matching

    def greedy_initial_decomposition(edges, seed):
        """
        Create an initial path decomposition using greedy approach.
        Returns list of paths.
        """
        rng = random.Random(seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        paths = []
        
        while remaining:
            # Pick a random edge to start
            edge = rng.choice(list(remaining))
            remaining.remove(edge)
            u, v = edge
            path = [u, v]
            
            # Extend the path greedily
            changed = True
            while changed:
                changed = False
                # Try extending from both ends
                for end in [0, 1]:  # 0 = front, 1 = back
                    if end == 0:
                        current = path[0]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.insert(0, nb)
                            changed = True
                    else:
                        current = path[-1]
                        candidates = [nb for nb in adj[current] 
                                      if nb not in path and ((current, nb) in remaining or (nb, current) in remaining)]
                        if candidates:
                            nb = rng.choice(candidates)
                            if (current, nb) in remaining:
                                remaining.remove((current, nb))
                            else:
                                remaining.remove((nb, current))
                            path.append(nb)
                            changed = True
                
                if not changed:
                    break
            
            paths.append(path)
        
        return paths

    def merge_paths_with_matching(paths, edges, adj, used_edges, max_iterations=50):
        """
        Iteratively merge path endpoints using maximum matching.
        paths: list of paths (each is a list of vertices)
        edges: set of all edges
        adj: adjacency dict
        used_edges: set of edges already used in paths
        Returns: merged list of paths
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Build auxiliary graph for matching
            # Each path has two endpoints: first and last vertex
            # We can match endpoint i of path A with endpoint j of path B if:
            # 1. There's an unused edge between them
            # 2. They're from different paths
            # 3. The merged path is simple
            
            nodes = []
            for i, path in enumerate(current_paths):
                if len(path) >= 2:
                    nodes.append((i, 0))
                    nodes.append((i, 1))
            
            if len(nodes) < 4:  # Need at least 2 paths with 2 endpoints each
                break
            
            # Build adjacency for matching - only include edges that result in simple merged paths
            match_adj = defaultdict(set)
            
            for idx_a in range(len(nodes)):
                for idx_b in range(idx_a + 1, len(nodes)):
                    a = nodes[idx_a]
                    b = nodes[idx_b]
                    path_a, pos_a = a
                    path_b, pos_b = b
                    
                    # Skip if same path
                    if path_a == path_b:
                        continue
                    
                    vertex_a = current_paths[path_a][0] if pos_a == 0 else current_paths[path_a][-1]
                    vertex_b = current_paths[path_b][0] if pos_b == 0 else current_paths[path_b][-1]
                    
                    # Check if there's an unused edge between them
                    e = tuple(sorted((vertex_a, vertex_b)))
                    if e in current_used:
                        continue
                    if e not in edges:
                        continue
                    
                    # Check if connecting would create a cycle (i.e., if the two paths share any vertices)
                    set_a = set(current_paths[path_a])
                    set_b = set(current_paths[path_b])
                    if set_a & set_b:
                        continue
                    
                    match_adj[a].add(b)
                    match_adj[b].add(a)
            
            # Find maximum matching using Hopcroft-Karp
            left_nodes = list(match_adj.keys())
            right_nodes = list(match_adj.keys())
            
            if not left_nodes:
                break
            
            matching = max_matching_bipartite(left_nodes, right_nodes, match_adj)
            
            if not matching:
                break
            
            # Apply all matches
            new_paths = {i: list(p) for i, p in enumerate(current_paths)}
            removed_paths = set()
            
            for node_a, node_b in matching:
                path_a, pos_a = node_a
                path_b, pos_b = node_b
                
                if path_a in removed_paths or path_b in removed_paths:
                    continue
                
                vertex_a = new_paths[path_a][0] if pos_a == 0 else new_paths[path_a][-1]
                vertex_b = new_paths[path_b][0] if pos_b == 0 else new_paths[path_b][-1]
                
                # Determine how to merge
                path1 = new_paths[path_a]
                path2 = new_paths[path_b]
                
                # Determine orientation
                if pos_a == 1:
                    path1 = list(reversed(path1))
                
                if pos_b == 1:
                    path2 = list(reversed(path2))
                
                # Now we're connecting path1[-1] to path2[0]
                if path1[-1] != vertex_a or path2[0] != vertex_b:
                    continue
                
                # Merge
                merged = path1 + path2[1:]
                
                # Check if the merged path is simple
                if len(set(merged)) != len(merged):
                    continue
                
                # Check if all edges in the merged path are valid
                valid = True
                for j in range(len(merged) - 1):
                    e = tuple(sorted((merged[j], merged[j+1])))
                    if e not in edges:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Update
                new_paths[path_a] = merged
                del new_paths[path_b]
                removed_paths.add(path_b)
                current_used.add(tuple(sorted((vertex_a, vertex_b))))
            
            # Rebuild current_paths
            current_paths = [new_paths[i] for i in sorted(new_paths.keys())]
            
            if len(current_paths) >= len(paths):
                break
        
        return current_paths

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        # Build adjacency and degrees
        adj = defaultdict(set)
        degrees = defaultdict(int)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            
        odd_vertices = [v for v in degrees if degrees[v] % 2 != 0]
        num_odd = len(odd_vertices)
        
        # Theoretical Lower Bound
        if num_odd == 0:
            lb = 1 if len(edges) == 1 else 2
        else:
            lb = num_odd // 2
        
        best_paths = None
        best_count = float('inf')
        
        # Strategy: Try multiple random initial decompositions, then merge with matching
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            seed_val = seed * 10000 + attempt
            initial_paths = greedy_initial_decomposition(edges, seed_val)
            
            if not initial_paths:
                continue
            
            # Track used edges
            used_edges = set()
            for path in initial_paths:
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e = tuple(sorted((u, v)))
                    used_edges.add(e)
            
            # Merge with matching
            merged_paths = merge_paths_with_matching(initial_paths, edges, adj, used_edges)
            
            # Validate
            if validate_paths(merged_paths, edges):
                if len(merged_paths) < best_count:
                    best_count = len(merged_paths)
                    best_paths = merged_paths
                    
                    # If we reached the lower bound, we can stop
                    if len(merged_paths) <= lb:
                        break
        
        # Fallback to simple greedy if matching approach didn't work well
        if best_paths is None:
            for i in range(50):
                if time.time() > deadline - 5:
                    break
                paths = greedy_initial_decomposition(edges, seed * 10000 + 10000 + i)
                if validate_paths(paths, edges):
                    if len(paths) < best_count:
                        best_count = len(paths)
                        best_paths = paths
            
            if best_paths is None:
                best_paths = [[u, v] for u, v in edges]
                best_count = len(best_paths)

        return best_paths

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    results = []
    
    for spec in specs:
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        paths = solve_instance(n, p, seed)
        results.append(paths)
        
    # Fallback
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
