import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    TIMEOUT = 110.0  # Leave 10s margin for 120s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, time_limit):
        # Build adjacency list
        # edges is a list of (u, v) with u < v
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        # Helper to remove edges from adj and remaining_edges
        def remove_edge(u, v):
            e = tuple(sorted((u, v)))
            if e in remaining_edges:
                remaining_edges.remove(e)
                adj[u].remove(v)
                adj[v].remove(u)
                
        # Helper to check if graph has any cycle
        def has_cycle():
            # If all degrees are <= 1, no cycle
            # Simple DFS to detect cycle
            visited = {i: False for i in range(n)}
            
            def dfs(u, parent):
                visited[u] = True
                for v in adj[u]:
                    if not visited[v]:
                        if dfs(v, u):
                            return True
                    elif v != parent:
                        return True
                return False
                
            for i in range(n):
                if not visited[i]:
                    if dfs(i, -1):
                        return True
            return False

        # Heuristic to find a long cycle
        # Strategy: Try multiple random starting edges, do greedy DFS to find a path back to start
        def find_long_cycle():
            if not remaining_edges:
                return None
            
            best_cycle = None
            best_len = 0
            
            # Sample a few random edges to start
            # We want to find a cycle, so we need a start edge (u,v) and a path v->...->u
            sample_edges = list(remaining_edges)
            random.shuffle(sample_edges)
            num_attempts = min(20, len(sample_edges))
            
            for u, v in sample_edges[:num_attempts]:
                # Try to find path from v to u in graph without edge (u,v)
                # We simulate this by doing DFS from v, avoiding u until the end?
                # No, we can visit u only if we have formed a path.
                # Standard DFS for path from v to u.
                
                # To bias for long paths, we can use a heuristic ordering
                # Sort neighbors by degree (descending) to explore "rich" areas first?
                # Or just random?
                
                # Let's try a simple DFS that tracks path.
                # Since N is small, we can just do a standard DFS and if we find u, we have a cycle.
                # To find a LONG cycle, we can try to maximize depth.
                
                # Heuristic: 
                # 1. Start at v.
                # 2. Next node: pick neighbor with highest degree (unvisited).
                # 3. If stuck, backtrack.
                
                # This is essentially finding a long simple path from v to u.
                # We'll implement a limited DFS that prefers high-degree nodes.
                
                def dfs_long(current, target, path, visited):
                    nonlocal best_cycle, best_len
                    if time.time() > time_limit:
                        return None
                    if current == target:
                        # Found a cycle: path + [u]
                        cycle = list(path) + [u]
                        # Check if distinct
                        if len(cycle) == len(set(cycle)):
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                        return True # Found a cycle, but we might want longer?
                        # For now, we return to try other branches if we want strictly longest,
                        # but greedy usually finds decent ones. Let's just return the first found
                        # in this branch to save time, or continue?
                        # Given N=30, continuing to find longest is feasible.
                        # But let's just take the first valid one to keep it simple and fast,
                        # relying on multiple random starts to find long ones.
                        # Actually, if we return immediately, we might miss a longer path in the same tree.
                        # Let's just return True and let the outer loop handle other starts.
                        # Or we can keep searching? 
                        # Let's keep searching for a longer one within this start?
                        # No, that's expensive. Let's just return the first one found.
                        # The "best" is updated.
                        pass 
                    # If we found a cycle, we record it. We can stop this DFS branch.
                    # But we might find a longer one if we continue?
                    # Let's just return True to stop this branch.
                    
                    # To find long cycles, we should prefer neighbors that lead to dead ends less?
                    # Let's just use random order or degree order.
                    
                    neighbors = list(adj[current])
                    # Filter out visited
                    neighbors = [x for x in neighbors if x not in visited]
                    # Sort by degree descending
                    neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                    
                    for next_node in neighbors:
                        visited.add(next_node)
                        path.append(next_node)
                        dfs_long(next_node, target, path, visited)
                        path.pop()
                        visited.remove(next_node)
                        if time.time() > time_limit:
                            return True

                # We need to be careful: the edge (u,v) is still in the graph for adjacency.
                # We must treat it as removed for the path search.
                # So in dfs_long, when current is v, we should not go to u immediately?
                # No, the path is v -> ... -> u. The edge (u,v) is the closing edge.
                # So in the DFS, we are looking for a path from v to u.
                # The edge (u,v) is NOT used in the path v->...->u.
                # So we should temporarily remove (u,v) from adj?
                # Or just check: if current == v and next_node == u, skip?
                # Actually, if we are at v, and we go to u, that's length 1 path (edge v-u).
                # Cycle length would be 2 (v-u-v). But simple cycles must be length >= 3.
                # So we should NOT allow direct v->u if we want a simple cycle of length >= 3?
                # The problem says "cycle given by at least three distinct vertices".
                # So a 2-cycle is not allowed.
                # Thus, we must find a path of length >= 2 from v to u.
                
                # Let's modify dfs_long:
                # If current == v and next_node == u: skip (would be 2-cycle)
                
                # We'll implement a specific DFS for this.
                
                def dfs_path(start, target):
                    nonlocal best_cycle, best_len
                    if time.time() > time_limit:
                        return
                    
                    stack = [(start, [start], {start})]
                    while stack:
                        if time.time() > time_limit:
                            return
                        node, path, visited = stack.pop()
                        if node == target:
                            if len(path) >= 3: # path is v...u, cycle is v...u-v, length len(path)+1? 
                                # Wait. Path is [v, ..., u]. 
                                # Cycle is [v, ..., u, v].
                                # Distinct vertices in cycle: v, ..., u.
                                # If len(path) >= 3, then vertices are distinct and count >= 3.
                                # Example: path [v, a, u]. Cycle v-a-u-v. Vertices v,a,u. Distinct. OK.
                                # Example: path [v, u]. Cycle v-u-v. Vertices v,u. Not distinct enough? 
                                # "at least three distinct vertices". So len(path) must be >= 3.
                                cycle = list(path) + [start] # start is v? No, target is u.
                                # Cycle: path + [start] ? No.
                                # Path: v -> ... -> u.
                                # Cycle: v -> ... -> u -> v.
                                # So cycle vertices: path + [v].
                                # But v is already in path[0].
                                # So cycle = path + [path[0]].
                                # Let's just store the path.
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_cycle = list(path)
                            # Don't return, might find longer?
                            # For now, continue to find longest in this DFS tree?
                            # This DFS is not strictly "longest path" because it's LIFO.
                            # But it explores.
                            continue
                        
                        neighbors = list(adj[node])
                        # Exclude the edge (u,v) if we are at v and neighbor is u?
                        # And generally, we cannot use the edge (u,v) in the interior.
                        # But we are searching for a path from v to u.
                        # The edge (u,v) connects u and v.
                        # If we are at v, we shouldn't go to u directly (2-cycle).
                        # If we are at u, we stop.
                        # So we just need to ensure we don't use (u,v) as an intermediate step?
                        # Since it's a simple path, we only visit u at the end.
                        # So we just need to prevent v->u direct step.
                        
                        # Filter neighbors
                        nexts = []
                        for nb in neighbors:
                            if nb in visited:
                                continue
                            if node == start and nb == target:
                                continue # Skip direct edge to target from start
                            nexts.append(nb)
                        
                        # Heuristic: sort by degree
                        nexts.sort(key=lambda x: len(adj[x]), reverse=True)
                        
                        for nb in nexts:
                            stack.append((nb, path + [nb], visited | {nb}))
                            
                dfs_path(v, u)
                
            if best_cycle:
                # best_cycle is a path [v, ..., u]
                # The cycle is best_cycle + [v]
                # But we need to return the list of vertices in the cycle.
                # The closing edge is implied.
                # So we return best_cycle + [best_cycle[0]]?
                # "A cycle given by at least three distinct vertices (the closing edge is implied)."
                # Usually this means [v1, v2, ..., vk] implies edges (v1,v2), ..., (vk-1,vk), (vk,v1).
                # So if we found path v->a->u, the cycle is v-a-u-v.
                # Vertices: [v, a, u].
                # So we should return [v, a, u].
                return best_cycle
            return None

        while remaining_edges:
            if time.time() > time_limit:
                break
                
            # Check if any cycle exists
            if not has_cycle():
                # All remaining edges are single pieces
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Try to find a long cycle
            cycle = find_long_cycle()
            
            if cycle:
                # cycle is a list of vertices [v0, v1, ..., vk-1]
                # Edges are (v0,v1), (v1,v2), ..., (vk-1,v0)
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    remove_edge(u, v)
            else:
                # Fallback: find ANY cycle using standard DFS
                # This ensures we don't leave a cycle behind as single edges
                # If find_long_cycle failed, it might be because it was too slow or heuristic failed.
                # Let's do a simple DFS to find any cycle.
                visited = {i: False for i in range(n)}
                parent = {i: -1 for i in range(n)}
                found_cycle = None
                
                def dfs_any(u):
                    nonlocal found_cycle
                    visited[u] = True
                    for v in adj[u]:
                        if not visited[v]:
                            parent[v] = u
                            dfs_any(v)
                        elif v != parent[u]:
                            # Cycle found
                            # Reconstruct path from u to v
                            path = [v]
                            cur = u
                            while cur != v:
                                path.append(cur)
                                cur = parent[cur]
                            path.append(v) # Close the loop?
                            # The cycle is path[1:] ?
                            # Let's trace:
                            # u -> ... -> v. And edge v-u exists.
                            # Cycle: u -> ... -> v -> u.
                            # Vertices: [u, ..., v].
                            # So we return the list of vertices in the cycle.
                            cyc = [u]
                            cur = parent[u]
                            while cur != v:
                                cyc.append(cur)
                                cur = parent[cur]
                            cyc.append(v)
                            found_cycle = cyc
                            return True
                    return False
                
                for i in range(n):
                    if not visited[i]:
                        if dfs_any(i):
                            break
                            
                if found_cycle:
                    pieces.append(found_cycle)
                    for i in range(len(found_cycle)):
                        u = found_cycle[i]
                        v = found_cycle[(i + 1) % len(found_cycle)]
                        remove_edge(u, v)
                else:
                    # Should not happen if has_cycle() was true
                    # But if it does, take one edge
                    e = next(iter(remaining_edges))
                    pieces.append(list(e))
                    remove_edge(e[0], e[1])
                    
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        # Set a per-graph time limit
        graph_limit = time.time() + (TIMEOUT - (time.time() - start_time)) / 3.0
        pieces = decompose_graph(edges, n, graph_limit)
        results.append(pieces)
        
    return results

from random import Random
