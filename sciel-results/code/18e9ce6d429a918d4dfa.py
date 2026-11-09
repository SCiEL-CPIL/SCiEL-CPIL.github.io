"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def greedy_path_cover(edges, n):
        """
        Greedy approach: repeatedly extract a long simple path from remaining edges.
        Uses a heuristic to extend the path greedily without backtracking to ensure speed.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        remaining_edges = set(edge_set)
        paths = []
        
        # Precompute adjacency for the full graph to quickly check neighbors
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        while remaining_edges:
            # Find a starting vertex with the highest degree in the remaining subgraph
            best_start = -1
            best_deg = -1
            
            # To speed up, we can iterate through vertices that still have edges
            # But iterating 0..n is fine for n<=30
            for v in range(n):
                deg = 0
                for u in adj[v]:
                    if (min(u,v), max(u,v)) in remaining_edges:
                        deg += 1
                if deg > best_deg:
                    best_deg = deg
                    best_start = v
            
            if best_start == -1:
                break # Should not happen if remaining_edges is not empty
            
            # Build a path greedily starting from best_start
            current = best_start
            path = [current]
            visited = {current}
            
            while True:
                # Find the next vertex to extend the path
                next_vertex = -1
                max_next_deg = -1
                
                for neighbor in adj[current]:
                    if neighbor in visited:
                        continue
                    edge = tuple(sorted((current, neighbor)))
                    if edge not in remaining_edges:
                        continue
                    
                    # Heuristic: prefer neighbors with higher degree in remaining graph
                    # to keep options open or to consume high-degree nodes early?
                    # Actually, just picking any valid neighbor is fast.
                    # Let's pick the one with the highest degree to try to make the path longer.
                    deg = 0
                    for u in adj[neighbor]:
                        if (min(u,neighbor), max(u,neighbor)) in remaining_edges:
                            deg += 1
                    
                    if deg > max_next_deg:
                        max_next_deg = deg
                        next_vertex = neighbor
                
                if next_vertex == -1:
                    break # No more edges to extend from current
                
                # Move to next vertex
                current = next_vertex
                path.append(current)
                visited.add(current)
                
                # Remove the edge we just traversed
                prev = path[-2]
                edge = tuple(sorted((prev, current)))
                remaining_edges.discard(edge)
            
            paths.append(path)
            
            # If we didn't remove any edges (should not happen if best_start was valid), break
            if len(path) < 2:
                 # This implies best_start had no available edges, which contradicts best_deg > 0
                 # But just in case, remove one edge manually
                 if remaining_edges:
                     e = next(iter(remaining_edges))
                     remaining_edges.discard(e)
                     paths.append([e[0], e[1]])
                 else:
                     break

        return paths

    def try_merge_paths(paths, edge_set):
        """Try to merge paths that share endpoints."""
        if len(paths) <= 1:
            return paths
        
        improved = True
        while improved:
            improved = False
            new_paths = []
            merged_indices = set()
            
            for i in range(len(paths)):
                if i in merged_indices:
                    continue
                path_i = paths[i]
                merged_flag = False
                
                for j in range(i + 1, len(paths)):
                    if j in merged_indices:
                        continue
                    path_j = paths[j]
                    
                    # Try all 4 combinations of orientations
                    merged_candidate = None
                    for p1 in [path_i, path_i[::-1]]:
                        for p2 in [path_j, path_j[::-1]]:
                            # Check if we can concatenate
                            if p1[-1] == p2[0]:
                                candidate = p1 + p2[1:]
                                if len(set(candidate)) == len(candidate):
                                    # Verify all edges exist
                                    valid = True
                                    for k in range(len(candidate) - 1):
                                        u, v = candidate[k], candidate[k+1]
                                        if tuple(sorted((u, v))) not in edge_set:
                                            valid = False
                                            break
                                    if valid:
                                        merged_candidate = candidate
                                        break
                        if merged_candidate:
                            break
                            
                    if merged_candidate:
                        new_paths.append(merged_candidate)
                        merged_indices.add(j)
                        improved = True
                        merged_flag = True
                        break
                
                if not merged_flag:
                    new_paths.append(path_i)
            
            paths = new_paths
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback to trivial solution if out of time
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        n = max(max(e) for e in edges) + 1 if edges else 0
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Use greedy longest path extraction
        paths = greedy_path_cover(edges, n)
        
        # Try to merge paths
        if time.time() < deadline - 5:
            paths = try_merge_paths(paths, edge_set)
        
        # Verify and clean up
        valid_paths = []
        used_edges = set()
        
        for path in paths:
            if len(path) < 2:
                continue
            path_edges = []
            valid = True
            seen = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = tuple(sorted((u, v)))
                if edge not in edge_set:
                    valid = False
                    break
                if u in seen or v in seen:
                    valid = False
                    break
                seen.add(u)
                seen.add(v)
                path_edges.append(edge)
            
            if not valid:
                continue
            if any(e in used_edges for e in path_edges):
                continue
            
            used_edges.update(path_edges)
            valid_paths.append(path)
        
        # Add remaining edges as single-edge paths
        for u, v in (edge_set - used_edges):
            valid_paths.append([u, v])
        
        results.append(valid_paths)
    
    return results
# EVOLVE-BLOCK-END
