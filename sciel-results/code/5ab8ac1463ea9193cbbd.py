import time
import random
from collections import defaultdict
from itertools import combinations

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
            queue = []
            for u in range(n_left):
                if match_left[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            
            found = False
            while queue:
                u = queue.pop(0)
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
        
        matching = []
        while bfs():
            for u in range(n_left):
                if match_left[u] == -1:
                    if dfs(u):
                        pass
        
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

    def merge_paths_with_matching(paths, edges, adj, used_edges, max_iterations=200):
        """
        Iteratively merge path endpoints using maximum matching on a bipartite auxiliary graph.
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
            
            # Each path with >= 2 vertices has two endpoints
            # Build list of (path_idx, endpoint_pos) for all endpoints
            endpoint_list = []
            for i, path in enumerate(current_paths):
                if len(path) >= 2:
                    endpoint_list.append((i, 0))
                    endpoint_list.append((i, 1))
            
            if len(endpoint_list) < 4:
                break
            
            # Build the set of valid merge edges: pairs of endpoints from different paths
            # that are connected by an unused graph edge and whose merge produces a simple path
            valid_merges = []  # list of ((path_a, pos_a), (path_b, pos_b), edge)
            
            # Precompute vertex sets for each path for simplicity check
            path_vertex_sets = [set(p) for p in current_paths]
            
            for idx_a, (path_a, pos_a) in enumerate(endpoint_list):
                for idx_b, (path_b, pos_b) in enumerate(endpoint_list[idx_a + 1:], start=idx_a + 1):
                    if path_a == path_b:
                        continue
                    
                    vertex_a = current_paths[path_a][0] if pos_a == 0 else current_paths[path_a][-1]
                    vertex_b = current_paths[path_b][0] if pos_b == 0 else current_paths[path_b][-1]
                    
                    e = tuple(sorted((vertex_a, vertex_b)))
                    if e in current_used:
                        continue
                    if e not in edges:
                        continue
                    
                    # Check if merging would create a non-simple path
                    # The merged path would be: path_a (possibly reversed) + path_b (possibly reversed)
                    # They share no vertices if the sets are disjoint
                    if path_vertex_sets[path_a] & path_vertex_sets[path_b]:
                        continue
                    
                    valid_merges.append(((path_a, pos_a), (path_b, pos_b), e))
            
            if not valid_merges:
                break
            
            # Try multiple random bipartitions to find a good matching
            best_matching_size = 0
            best_matches = None
            
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                
                # Random bipartition of endpoints
                rng_attempt = random.Random(iteration * 100 + attempt)
                left_nodes = []
                right_nodes = []
                left_to_endpoint = {}
                right_to_endpoint = {}
                
                for (path_idx, pos) in endpoint_list:
                    if rng_attempt.random() < 0.5:
                        left_nodes.append((path_idx, pos))
                        left_to_endpoint[(path_idx, pos)] = (path_idx, pos)
                    else:
                        right_nodes.append((path_idx, pos))
                        right_to_endpoint[(path_idx, pos)] = (path_idx, pos)
                
                # Ensure each path has at most one endpoint on each side
                # If both endpoints of a path are on the same side, move one to the other
                for path_idx in range(len(current_paths)):
                    if len(current_paths[path_idx]) >= 2:
                        ep0 = (path_idx, 0)
                        ep1 = (path_idx, 1)
                        both_left = ep0 in left_to_endpoint and ep1 in left_to_endpoint
                        both_right = ep0 in right_to_endpoint and ep1 in right_to_endpoint
                        if both_left:
                            # Move ep1 to right
                            left_nodes.remove(ep1)
                            del left_to_endpoint[ep1]
                            right_nodes.append(ep1)
                            right_to_endpoint[ep1] = ep1
                        elif both_right:
                            # Move ep1 to left
                            right_nodes.remove(ep1)
                            del right_to_endpoint[ep1]
                            left_nodes.append(ep1)
                            left_to_endpoint[ep1] = ep1
                
                # Build adjacency: left endpoint -> set of right endpoints it can connect to
                left_adj = defaultdict(set)
                for (path_a, pos_a), (path_b, pos_b), e in valid_merges:
                    if (path_a, pos_a) in left_to_endpoint and (path_b, pos_b) in right_to_endpoint:
                        left_adj[(path_a, pos_a)].add((path_b, pos_b))
                    elif (path_b, pos_b) in left_to_endpoint and (path_a, pos_a) in right_to_endpoint:
                        left_adj[(path_b, pos_b)].add((path_a, pos_a))
                
                # Find maximum matching
                matching = max_matching_bipartite(left_nodes, right_nodes, left_adj)
                
                if len(matching) > best_matching_size:
                    best_matching_size = len(matching)
                    # Convert matching back to merge operations
                    matches = []
                    for left_ep, right_ep in matching:
                        path_a, pos_a = left_ep
                        path_b, pos_b = right_ep
                        
                        vertex_a = current_paths[path_a][0] if pos_a == 0 else current_paths[path_a][-1]
                        vertex_b = current_paths[path_b][0] if pos_b == 0 else current_paths[path_b][-1]
                        e = tuple(sorted((vertex_a, vertex_b)))
                        
                        matches.append(((path_a, pos_a), (path_b, pos_b), e))
                    best_matches = matches
            
            if not best_matches:
                break
            
            # Apply all matches
            new_paths = {i: list(p) for i, p in enumerate(current_paths)}
            removed_paths = set()
            
            for (path_a, pos_a), (path_b, pos_b), edge in best_matches:
                if path_a in removed_paths or path_b in removed_paths:
                    continue
                
                vertex_a = new_paths[path_a][0] if pos_a == 0 else new_paths[path_a][-1]
                vertex_b = new_paths[path_b][0] if pos_b == 0 else new_paths[path_b][-1]
                
                path1 = list(new_paths[path_a])
                path2 = list(new_paths[path_b])
                
                # Orient path1 so that vertex_a is at the end
                if pos_a == 0:
                    path1 = list(reversed(path1))
                
                # Orient path2 so that vertex_b is at the start
                if pos_b == 1:
                    path2 = list(reversed(path2))
                
                # Now path1 ends at vertex_a and path2 starts at vertex_b
                if path1[-1] != vertex_a or path2[0] != vertex_b:
                    continue
                
                merged = path1 + path2[1:]
                
                # Verify simplicity
                if len(set(merged)) != len(merged):
                    continue
                
                new_paths[path_a] = merged
                del new_paths[path_b]
                removed_paths.add(path_b)
                current_used.add(edge)
            
            current_paths = [new_paths[i] for i in sorted(new_paths.keys())]
            
            if len(current_paths) == len(paths):
                break
        
        return current_paths

    def local_search_improve(paths, edges, adj, used_edges, max_iterations=50):
        """
        Try to reduce path count by swapping segments between paths.
        """
        if not paths:
            return []
        
        current_paths = [list(p) for p in paths]
        current_used = set(used_edges)
        best_count = len(current_paths)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            improved = False
            
            # Try to find two paths where we can swap a suffix/prefix
            # to create two shorter paths or one longer path
            for i in range(len(current_paths)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(current_paths)):
                    path_i = current_paths[i]
                    path_j = current_paths[j]
                    
                    if len(path_i) < 2 or len(path_j) < 2:
                        continue
                    
                    # Try cutting path_i at various positions and path_j at various positions
                    # and see if we can reconnect to form valid paths
                    for cut_i in range(1, len(path_i) - 1):
                        for cut_j in range(1, len(path_j) - 1):
                            # Split path_i into A = path_i[:cut_i], B = path_i[cut_i:]
                            # Split path_j into C = path_j[:cut_j], D = path_j[cut_j:]
                            A = path_i[:cut_i]
                            B = path_i[cut_i:]
                            C = path_j[:cut_j]
                            D = path_j[cut_j:]
                            
                            # Option 1: Connect A+D and B+C (if edges exist)
                            # A ends at A[-1], D starts at D[0]
                            # B ends at B[-1], C starts at C[0]
                            e1 = tuple(sorted((A[-1], D[0])))
                            e2 = tuple(sorted((B[-1], C[0])))
                            
                            if e1 in edges and e2 in edges:
                                # Check if these edges are currently used
                                # They should be the edges we're removing from the original paths
                                orig_e1 = tuple(sorted((path_i[cut_i-1], path_i[cut_i])))
                                orig_e2 = tuple(sorted((path_j[cut_j-1], path_j[cut_j])))
                                
                                # The new edges must not be in current_used (they're new connections)
                                # Actually, we need to check that A+D and B+C are simple paths
                                new_path1 = A + D[1:]
                                new_path2 = B + C[1:]
                                
                                if len(set(new_path1)) == len(new_path1) and len(set(new_path2)) == len(new_path2):
                                    # Check that the edges used in new paths are valid
                                    valid = True
                                    new_used = set()
                                    for path in [new_path1, new_path2]:
                                        for k in range(len(path) - 1):
                                            u, v = path[k], path[k+1]
                                            e = tuple(sorted((u, v)))
                                            if e not in edges or e in current_used:
                                                valid = False
                                                break
                                            new_used.add(e)
                                        if not valid:
                                            break
                                    
                                    if valid and new_used <= edges:
                                        # Check that we're not creating duplicate edges
                                        if len(new_used) == len(new_path1) - 1 + len(new_path2) - 1:
                                            # This swap is valid
                                            # Remove old edges, add new edges
                                            old_edges = set()
                                            for path in [path_i, path_j]:
                                                for k in range(len(path) - 1):
                                                    u, v = path[k], path[k+1]
                                                    e = tuple(sorted((u, v)))
                                                    old_edges.add(e)
                                            
                                            # The new used edges should be a subset of edges
                                            # and we should have the same number of edges
                                            if len(new_used) == len(old_edges):
                                                current_paths[i] = new_path1
                                                current_paths[j] = new_path2
                                                current_used = new_used
                                                improved = True
                                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
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
            
            # Try local search improvement
            if len(merged_paths) > lb:
                improved_paths = local_search_improve(merged_paths, edges, adj, used_edges)
                if validate_paths(improved_paths, edges) and len(improved_paths) < len(merged_paths):
                    merged_paths = improved_paths
            
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
            for i in range(100):
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
    
    # Allocate time more evenly, with more time for larger instances
    time_alloc = [30, 35, 45]  # Total ~110s with safety margin
    
    for idx, spec in enumerate(specs):
        n, p, seed = spec
        # Check time before starting instance
        if time.time() > deadline - 10:
            # Not enough time, use trivial decomposition for remaining
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        # Set a local deadline for this instance
        instance_deadline = min(deadline - 5, time.time() + time_alloc[idx])
        
        # Temporarily override the global deadline for this instance
        global_deadline_backup = deadline
        # We need to pass the deadline to solve_instance, but it uses the closure
        # Instead, let's just rely on the global deadline check in solve_instance
        
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
