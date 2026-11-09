import time
from random import Random

def construct():
        start_time = time.time()
        DEATHLINE = 110.0
        
        def instance(n, p, seed):
            rng = Random(seed)
            order = list(range(n))
            rng.shuffle(order)
            edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
            edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
            return sorted(edges)

        specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
        results = []
        
        for n, p, seed in specs:
            edges = instance(n, p, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            remaining_edges = set(edges)
            paths = []
            rng = Random(seed + 1000)
            
            def decompose_low_degree(adj, remaining_edges, n):
                # Handles graphs with max degree <= 2
                # Returns list of paths
                res_paths = []
                visited_edges = set()
                
                for start in range(n):
                    if not adj[start]:
                        continue
                    
                    # Start a walk
                    curr = start
                    path = [curr]
                    
                    while True:
                        # Find an unvisited edge from curr
                        next_node = None
                        for neighbor in adj[curr]:
                            edge = tuple(sorted((curr, neighbor)))
                            if edge not in visited_edges:
                                next_node = neighbor
                                break
                                
                        if next_node is None:
                            break
                            
                        visited_edges.add(tuple(sorted((curr, next_node))))
                        path.append(next_node)
                        curr = next_node
                        
                        # If we return to start, it's a cycle.
                        # But we need simple paths.
                        # If curr == start, we have a cycle.
                        # We can break the cycle by removing the last edge?
                        # Actually, the walk stops when no unvisited edges.
                        # If it's a cycle, it will stop when we are back at start and all edges visited.
                        # The path will be [v0, v1, ..., vk, v0].
                        # This is a cycle, not a simple path (v0 repeated).
                        # We need to output a path.
                        # A cycle can be represented as a path by dropping one endpoint?
                        # No, a path cover of a cycle is 1 path? No, a cycle is not a path.
                        # A path is $v_1...v_k$ with distinct vertices.
                        # A cycle $v_1...v_k v_1$ has repeated vertex.
                        # To cover a cycle with paths, we can take $v_1...v_k$ (length k-1 edges) and leave $v_k-v_1$?
                        # No, we must cover ALL edges.
                        # A cycle cannot be covered by a single simple path.
                        # It requires at least 1 path? No.
                        # Example: Triangle 0-1-2-0.
                        # Path 0-1-2 covers (0,1) and (1,2). Edge (2,0) remains.
                        # Path 2-0 covers (2,0).
                        # So a cycle of length k requires 1 path of length k-1 and 1 path of length 1?
                        # Or just 1 path if we allow non-simple? No, "simple" means distinct vertices.
                        # So yes, cycles need to be broken.
                        
                        # However, if max degree is 2, components are paths or cycles.
                        # If it's a path (endpoints deg 1), the walk produces a valid simple path.
                        # If it's a cycle (all deg 2), the walk produces a cycle.
                        # We need to detect if it's a cycle.
                        # If path[0] == path[-1], it's a cycle.
                        # In that case, we can output path[:-1] as one path, and the last edge as another?
                        # Wait, path[:-1] is $v_0...v_{k-1}$. Edges covered: $(v_0,v_1)...(v_{k-2},v_{k-1})$.
                        # Remaining edge: $(v_{k-1}, v_0)$.
                        # So we add `path[:-1]` and `[path[-1], path[0]]`?
                        # No, `path` is $v_0...v_k v_0$.
                        # `path[:-1]` is $v_0...v_k$.
                        # The last edge in the walk was $(v_k, v_0)$.
                        # So `path[:-1]` covers edges $0..k-1$.
                        # The edge $(v_k, v_0)$ is covered by the walk step that closed it.
                        # But we can't use it in the simple path $v_0...v_k$.
                        # So we should output `path[:-1]` and `[v_k, v_0]`?
                        # No, `[v_k, v_0]` is a path of length 1.
                        # But we already "used" that edge in the walk logic.
                        # We just need to output the paths that cover the edges.
                        # So if cycle: output `path[:-1]` and `[path[-1], path[0]]`?
                        # Wait, `path` is $v_0, v_1, ..., v_k, v_0$.
                        # Edges: $(v_0,v_1), ..., (v_{k-1},v_k), (v_k,v_0)$.
                        # Path 1: $v_0...v_k$. Edges: $(v_0,v_1)...(v_{k-1},v_k)$.
                        # Path 2: $v_k...v_0$? No, $v_k, v_0$. Edge $(v_k,v_0)$.
                        # So yes, 2 paths.
                        
                        if curr == start and len(path) > 2:
                             # It's a cycle
                             # The current path is a cycle.
                             # We break it into two paths?
                             # Actually, we can just append the path minus the last node,
                             # and then the edge closing it as a separate path?
                             # But we are in a "decompose" function that returns paths.
                             # We can just return `path[:-1]` and `[path[-1], path[0]]`?
                             # No, `path[:-1]` is a valid simple path.
                             # The last edge is `(path[-2], path[-1])`? No.
                             # Let's trace:
                             # path = [0, 1, 2, 0]
                             # path[:-1] = [0, 1, 2]. Edges: (0,1), (1,2).
                             # Missing edge: (2,0).
                             # So we add [2, 0].
                             res_paths.append(path[:-1])
                             res_paths.append([path[-1], path[0]])
                             # And we must mark all these edges as visited in the outer scope?
                             # The function returns paths, so the caller must update state.
                             # This makes the function slightly complex.
                             # For simplicity, I will just let the greedy handle it, 
                             # or modify the walk to stop before closing the cycle?
                             # If we stop before closing, we have a path $v_0...v_k$.
                             # The edge $(v_k, v_0)$ remains.
                             # Then the next walk will pick it up as a single edge path.
                             # This is valid and simple.
                             pass
                        else:
                            res_paths.append(path)
                            
                return res_paths

            # Main loop
            while remaining_edges:
                if time.time() > DEATHLINE - 5:
                    # Fallback: just add remaining edges as single paths
                    for e in remaining_edges:
                        paths.append(list(e))
                    break

                # Check if max degree <= 2
                max_deg = max((len(adj[u]) for u in range(n)), default=0)
                if max_deg <= 2:
                    # Decompose
                    # We need to be careful with the visited_edges logic in decompose_low_degree
                    # It's better to just do a simple traversal here.
                    local_visited = set()
                    for start in range(n):
                        if not adj[start]:
                            continue
                        curr = start
                        path = [curr]
                        while True:
                            next_node = None
                            for neighbor in adj[curr]:
                                edge = tuple(sorted((curr, neighbor)))
                                if edge not in local_visited and edge in remaining_edges:
                                    next_node = neighbor
                                    break
                            if next_node is None:
                                break
                            local_visited.add(tuple(sorted((curr, next_node))))
                            path.append(next_node)
                            curr = next_node
                            if curr == start and len(path) > 2:
                                # Cycle detected
                                # Append path without closing edge
                                # The closing edge is (path[-2], path[-1])? No, (path[-1], start)
                                # path is [v0...vk v0]
                                # We want [v0...vk]
                                # The edge (vk, v0) is in local_visited.
                                # We will output [v0...vk]
                                # The edge (vk, v0) will be handled by a subsequent start?
                                # No, it's in local_visited.
                                # So we must output it separately.
                                paths.append(path[:-1])
                                paths.append([path[-1], path[0]])
                                # Remove these edges from remaining_edges
                                for i in range(len(path[:-1])-1):
                                     e = tuple(sorted((path[:-1][i], path[:-1][i+1])))
                                     remaining_edges.remove(e)
                                     adj[path[:-1][i]].remove(path[:-1][i+1])
                                     adj[path[:-1][i+1]].remove(path[:-1][i])
                                e_close = tuple(sorted((path[-1], path[0])))
                                remaining_edges.remove(e_close)
                                adj[path[-1]].remove(path[0])
                                adj[path[0]].remove(path[-1])
                                break
                            else:
                                # Normal path
                                paths.append(path)
                                for i in range(len(path)-1):
                                    e = tuple(sorted((path[i], path[i+1])))
                                    remaining_edges.remove(e)
                                    adj[path[i]].remove(path[i+1])
                                    adj[path[i+1]].remove(path[i])
                                break
                    continue

                # Greedy step
                best_path = []
                active_nodes = [i for i in range(n) if adj[i]]
                
                for start in active_nodes:
                    for _ in range(20):
                        curr = start
                        path = [curr]
                        used_edges = set()
                        
                        while True:
                            candidates = []
                            for neighbor in adj[curr]:
                                edge = tuple(sorted((curr, neighbor)))
                                if edge in remaining_edges and edge not in used_edges:
                                    candidates.append((len(adj[neighbor]), neighbor))
                            
                            if not candidates:
                                break
                            
                            candidates.sort(key=lambda x: x[0], reverse=True)
                            top_k = candidates[:3]
                            _, next_node = rng.choice(top_k)
                            
                            edge = tuple(sorted((curr, next_node)))
                            used_edges.add(edge)
                            path.append(next_node)
                            curr = next_node
                            
                        if len(path) > len(best_path):
                            best_path = path
                            
                if not best_path or len(best_path) < 2:
                    # Should not happen
                    break
                    
                paths.append(best_path)
                
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    edge = tuple(sorted((u, v)))
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    
            results.append(paths)
            
        return results
