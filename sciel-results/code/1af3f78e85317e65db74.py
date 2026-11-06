import time
import random
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []

    for i, spec in enumerate(specs):
        time_left = deadline - time.time()
        if time_left <= 0:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n_vertices = max(max(u, v) for u, v in edges) + 1 if edges else 0
        if n_vertices == 0:
            results.append([])
            continue

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Build cost matrix for Hungarian algorithm
        # This formulation finds a minimum-cost maximum matching where
        # each matched edge (u->v) represents chaining u to v in a path.
        # Minimizing cost maximizes the number of chained edges, thus minimizing paths.
        INF_COST = 10000
        cost_matrix = [[INF_COST] * n_vertices for _ in range(n_vertices)]
        
        for u, v in edges:
            cost_matrix[u][v] = 0
            cost_matrix[v][u] = 0
        
        # Run Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matched edges (only those with cost 0 are valid graph edges)
        matched_edges = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r][c] == 0:
                matched_edges.add((r, c))
        
        # Build directed graph from matched edges
        out_edges = defaultdict(list)
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for u, v in matched_edges:
            out_edges[u].append(v)
            out_degree[u] += 1
            in_degree[v] += 1
        
        # Reconstruct paths
        # A path starts at a node with in_degree 0 and follows out_edges
        # until out_degree is 0
        paths = []
        visited_in_path = set()
        
        for start in range(n_vertices):
            if in_degree[start] == 0 and out_degree[start] > 0:
                path = [start]
                current = start
                while out_degree[current] > 0:
                    current = out_edges[current][0]
                    path.append(current)
                    out_degree[current] -= 1
                paths.append(path)
        
        # Handle any remaining edges that were not covered
        # This can happen if the matching didn't cover all edges
        covered_edges = set()
        for path in paths:
            for idx in range(len(path) - 1):
                u, v = path[idx], path[idx + 1]
                covered_edges.add((min(u, v), max(u, v)))
        
        remaining_edges = set(edges) - covered_edges
        for e in remaining_edges:
            paths.append(list(e))

        # Phase 2: Local search improvement
        # Try to improve the decomposition by finding longer paths
        # that can replace multiple shorter paths
        
        if time.time() < deadline - 5.0:
            # Sort paths by length (shortest first)
            path_indices = list(range(len(paths)))
            path_indices.sort(key=lambda i: len(paths[i]))
            
            improved = True
            while improved and time.time() < deadline - 3.0:
                improved = False
                for idx in path_indices[:10]:  # Try to improve up to 10 shortest
                    if time.time() > deadline - 3.0:
                        break
                    
                    old_path = paths[idx]
                    if len(old_path) <= 1:
                        continue
                    
                    # Remove old path's edges
                    old_edges = set()
                    for j in range(len(old_path) - 1):
                        u, v = old_path[j], old_path[j + 1]
                        e = (min(u, v), max(u, v))
                        old_edges.add(e)
                    
                    # Try to find a longer path that includes some of these edges
                    # and possibly other edges from nearby short paths
                    
                    # Build a subgraph from old_path edges plus edges from other short paths
                    candidate_edges = set(old_edges)
                    for other_idx in path_indices[:10]:
                        if other_idx == idx:
                            continue
                        other_path = paths[other_idx]
                        for j in range(len(other_path) - 1):
                            u, v = other_path[j], other_path[j + 1]
                            e = (min(u, v), max(u, v))
                            candidate_edges.add(e)
                    
                    # Try to find a long path in the subgraph formed by candidate_edges
                    sub_adj = defaultdict(set)
                    for u, v in candidate_edges:
                        sub_adj[u].add(v)
                        sub_adj[v].add(u)
                    
                    # Beam search for longest path in subgraph
                    best_sub_path = None
                    active_vertices = set()
                    for u, v in candidate_edges:
                        active_vertices.add(u)
                        active_vertices.add(v)
                    
                    if active_vertices:
                        vertices_list = list(active_vertices)
                        random.shuffle(vertices_list)
                        max_starts = min(20, len(vertices_list))
                        
                        for start in vertices_list[:max_starts]:
                            if not sub_adj[start]:
                                continue
                            
                            beam = [([start], {start}, start)]
                            for depth in range(50):
                                if time.time() > deadline - 2.0:
                                    break
                                new_beam = []
                                for path, visited, node in beam:
                                    valid_neighbors = []
                                    for n in sub_adj[node]:
                                        if n in visited:
                                            continue
                                        e = (min(node, n), max(node, n))
                                        if e in candidate_edges:
                                            valid_neighbors.append(n)
                                    
                                    if not valid_neighbors:
                                        if best_sub_path is None or len(path) > len(best_sub_path):
                                            best_sub_path = path[:]
                                        continue
                                    
                                    # Warnsdorff: prefer neighbors with fewer options
                                    scored = []
                                    for n in valid_neighbors:
                                        count = 0
                                        for m in sub_adj[n]:
                                            if m in visited:
                                                continue
                                            e = (min(n, m), max(n, m))
                                            if e in candidate_edges:
                                                count += 1
                                        scored.append((count, n))
                                    scored.sort()
                                    
                                    for _, n in scored[:5]:
                                        new_path = path + [n]
                                        new_visited = visited | {n}
                                        new_beam.append((new_path, new_visited, n))
                                
                                if not new_beam:
                                    break
                                
                                new_beam.sort(key=lambda x: -len(x[0]))
                                beam = new_beam[:5]
                                
                                if beam:
                                    longest = max(len(p) for p, v, n in beam)
                                    if best_sub_path is None or longest > len(best_sub_path):
                                        best_sub_path = beam[0][0][:]
                            
                            for path, visited, node in beam:
                                if best_sub_path is None or len(path) > len(best_sub_path):
                                    best_sub_path = path[:]
                    
                    # If we found a longer path, replace the old path
                    if best_sub_path and len(best_sub_path) > len(old_path):
                        # Check if the new path uses only edges that are either
                        # in old_path or in other paths we're considering
                        new_path_edges = set()
                        for j in range(len(best_sub_path) - 1):
                            u, v = best_sub_path[j], best_sub_path[j + 1]
                            new_path_edges.add((min(u, v), max(u, v)))
                        
                        # Only accept if all edges in new path are in candidate_edges
                        if new_path_edges.issubset(candidate_edges):
                            paths[idx] = best_sub_path
                            improved = True
        
        results.append(paths)

    return results
