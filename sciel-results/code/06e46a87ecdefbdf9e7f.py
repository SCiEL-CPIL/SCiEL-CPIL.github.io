import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)

        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Calculate lower bound
        deg = {i: 0 for i in range(n)}
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 != 0)
        lower_bound = max(1, odd_count // 2)

        remaining_edges = set(edges)
        best_paths = None
        best_count = float('inf')

        # Improved heuristic: Greedy with Warnsdorff's rule and backtracking
        # We try multiple random restarts to find a minimal path cover

        max_restarts = 80
        if n <= 20:
            max_restarts = 120
        elif n <= 25:
            max_restarts = 100
        else:
            max_restarts = 80

        def find_longest_path_warnsdorff(start_node, curr_edges_set, adj_dict, max_depth=None):
            """Find longest simple path using Warnsdorff's rule with backtracking."""
            if max_depth is None:
                max_depth = len(adj_dict)
            
            best_path = [start_node]
            visited = {start_node}
            
            # Use iterative deepening with Warnsdorff's heuristic
            path = [start_node]
            visited_set = {start_node}
            
            def extend(node, path, visited_set):
                nonlocal best_path
                if len(path) > len(best_path):
                    best_path = path[:]
                
                # Find all valid next nodes
                candidates = []
                for nb in adj_dict[node]:
                    if nb in visited_set:
                        continue
                    e = (min(node, nb), max(node, nb))
                    if e in curr_edges_set:
                        # Count available moves from nb (Warnsdorff's rule)
                        move_count = 0
                        for m in adj_dict[nb]:
                            if m in visited_set:
                                continue
                            e2 = (min(nb, m), max(nb, m))
                            if e2 in curr_edges_set:
                                move_count += 1
                        candidates.append((move_count, nb))
                
                if not candidates:
                    return
                
                # Sort by Warnsdorff's rule: prefer nodes with fewer remaining moves
                candidates.sort()
                
                # Try the best candidate first
                for _, nb in candidates:
                    new_path = path + [nb]
                    new_visited = visited_set | {nb}
                    extend(nb, new_path, new_visited)
                    # After backtracking, continue trying other candidates
                    # This ensures we explore multiple paths
            
            extend(start_node, path, visited_set)
            return best_path

        def find_longest_path_beam(start_node, curr_edges_set, adj_dict, beam_width=15, max_depth=None):
            """Find longest simple path using beam search."""
            if max_depth is None:
                max_depth = len(adj_dict)
            
            beam = [([start_node], {start_node})]
            best_path = [start_node]
            
            for d in range(max_depth):
                new_beam = []
                for path, visited in beam:
                    node = path[-1]
                    valid_neighbors = []
                    for nb in adj_dict[node]:
                        if nb in visited:
                            continue
                        e = (min(node, nb), max(node, nb))
                        if e in curr_edges_set:
                            # Score by Warnsdorff's rule
                            count = 0
                            for m in adj_dict[nb]:
                                if m in visited:
                                    continue
                                e2 = (min(nb, m), max(nb, m))
                                if e2 in curr_edges_set:
                                    count += 1
                            valid_neighbors.append((count, nb))
                    
                    if not valid_neighbors:
                        if len(path) > len(best_path):
                            best_path = path[:]
                        continue
                    
                    valid_neighbors.sort()
                    
                    for _, nb in valid_neighbors[:beam_width]:
                        new_path = path + [nb]
                        new_visited = visited | {nb}
                        new_beam.append((new_path, new_visited))
                
                if not new_beam:
                    break
                
                # Keep only the longest paths
                new_beam.sort(key=lambda x: -len(x[0]))
                beam = new_beam[:beam_width]
                
                longest = max(len(p) for p, v in beam)
                if longest > len(best_path):
                    best_path = beam[0][0][:]
            
            return best_path

        def try_path_decomposition(curr_edges, adj, n, time_limit=0.5):
            """Try to decompose edges into minimum number of paths using greedy with restarts."""
            paths = []
            remaining = set(curr_edges)
            start_time_local = time.time()
            
            while remaining:
                if time.time() - start_time_local > time_limit:
                    return None
                
                # Find active nodes (those with remaining edges)
                active_nodes = set()
                for u, v in remaining:
                    active_nodes.add(u)
                    active_nodes.add(v)
                
                if not active_nodes:
                    break
                
                # Try to find the longest path
                best_path = None
                best_len = 0
                
                # Try starting from each active node (limit to improve speed)
                # Use Warnsdorff's rule to prioritize high-degree nodes
                active_list = list(active_nodes)
                # Sort by degree in remaining graph
                active_list.sort(key=lambda x: -sum(1 for y in adj[x] if (min(x, y), max(x, y)) in remaining))
                
                # Try a subset of starts
                for s in active_list[:min(20, len(active_list))]:
                    # Use beam search for speed
                    path = find_longest_path_beam(s, remaining, adj, beam_width=12, max_depth=n)
                    if path and len(path) > best_len:
                        best_path = path
                        best_len = len(path)
                
                if best_path and len(best_path) > 1:
                    # Verify path validity
                    valid = True
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in remaining:
                            valid = False
                            break
                    if valid:
                        paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i + 1]
                            e = (min(u, v), max(u, v))
                            remaining.discard(e)
                    else:
                        # Fallback: take a single edge
                        e = next(iter(remaining))
                        paths.append(list(e))
                        remaining.discard(e)
                else:
                    # Fallback: take a single edge
                    e = next(iter(remaining))
                    paths.append(list(e))
                    remaining.discard(e)
            
            return paths

        # Try multiple strategies
        for r in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            if best_count == lower_bound:
                break
            
            # Strategy 1: Greedy with random restarts
            curr_edges = set(remaining_edges)
            curr_paths = []
            
            # Randomize edge processing order
            edge_list = list(curr_edges)
            random.shuffle(edge_list)
            
            # Greedy extraction with Warnsdorff's rule
            while curr_edges:
                if time.time() > deadline - 2.0:
                    curr_paths = None
                    break
                
                # Find active nodes
                active_nodes = set()
                for u, v in curr_edges:
                    active_nodes.add(u)
                    active_nodes.add(v)
                
                if not active_nodes:
                    break
                
                # Try to find longest path from each active node
                best_path = None
                best_len = 0
                
                # Prioritize high-degree nodes
                active_list = list(active_nodes)
                active_list.sort(key=lambda x: -sum(1 for y in adj[x] if (min(x, y), max(x, y)) in curr_edges))
                
                # Try subset of starts
                for s in active_list[:min(15, len(active_list))]:
                    path = find_longest_path_beam(s, curr_edges, adj, beam_width=10, max_depth=n)
                    if path and len(path) > best_len:
                        best_path = path
                        best_len = len(path)
                
                if best_path and len(best_path) > 1:
                    # Verify path
                    valid = True
                    for i in range(len(best_path) - 1):
                        u, v = best_path[i], best_path[i + 1]
                        e = (min(u, v), max(u, v))
                        if e not in curr_edges:
                            valid = False
                            break
                    if valid:
                        curr_paths.append(best_path)
                        for i in range(len(best_path) - 1):
                            u, v = best_path[i], best_path[i + 1]
                            e = (min(u, v), max(u, v))
                            curr_edges.discard(e)
                    else:
                        e = next(iter(curr_edges))
                        curr_paths.append(list(e))
                        curr_edges.discard(e)
                else:
                    e = next(iter(curr_edges))
                    curr_paths.append(list(e))
                    curr_edges.discard(e)
            
            if curr_paths and len(curr_paths) < best_count:
                best_count = len(curr_paths)
                best_paths = curr_paths
        
        # If no best_paths found due to timeout, fallback to edge list
        if best_paths is None:
            best_paths = [list(e) for e in edges]
        
        all_results.append(best_paths)

    return all_results
