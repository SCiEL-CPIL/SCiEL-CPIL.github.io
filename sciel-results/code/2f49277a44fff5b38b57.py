import sys
import time
from collections import defaultdict, deque
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145  # 150s limit, keep 5s margin

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def solve_graph(n, edges):
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degrees
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
            
        odd_vertices = [i for i in range(n) if deg[i] % 2 == 1]
        
        if not odd_vertices:
            # Eulerian graph
            # Find Eulerian circuit
            # Use Hierholzer's algorithm
            # We need to return paths
            # Since it's Eulerian, we can cover it with 2 paths (1 long path + 1 edge)
            # Or potentially 1 path if we are lucky? No, simple path cannot be a cycle.
            # So we find an Eulerian circuit, break it at an edge.
            
            # If graph is empty, return []
            if not edges:
                return []
                
            # Find Eulerian circuit
            # We need to be careful with recursion depth for N=30, but it's fine.
            # Actually, we can just return 2 paths: one is the circuit minus one edge, the other is that edge.
            
            # To get the circuit:
            # Since N is small, we can use a simple backtracking or Hierholzer.
            # Hierholzer requires mutable adjacency or careful stack usage.
            
            # Let's just find any edge (u,v).
            # The rest of the graph is a path from v to u? No, it's a path from v to u in the remaining graph?
            # If we remove (u,v), the graph becomes a graph where u and v have odd degree, and others even.
            # It might be disconnected.
            # But since the original was Eulerian, removing one edge leaves a graph with exactly two odd vertices (u and v).
            # Such a graph has an Eulerian trail from u to v.
            # So we can find an Eulerian trail from u to v in the graph without (u,v).
            # This trail is a simple path? No, it might repeat vertices.
            # But we need simple paths.
            
            # This is getting complicated.
            # For N=30, we can just use a simple heuristic:
            # Find a cycle. Remove it. Repeat.
            # Each cycle of length L requires 2 paths?
            # If we have multiple cycles, we can combine them?
            
            # Simpler:
            # If O=0, we return 2 paths.
            # Path 1: A long path.
            # Path 2: The remaining edge.
            
            # Let's find an Eulerian trail in the graph with one edge removed.
            # Pick any edge e=(u,v).
            # Remove e.
            # Find Eulerian trail from u to v in remaining graph.
            # This trail uses all remaining edges.
            # The trail is a sequence of vertices.
            # We need to convert this trail into simple paths.
            # A trail might repeat vertices.
            # We can decompose a trail into simple paths?
            # Yes, by finding cycles in the trail and removing them.
            # But the number of paths might increase.
            
            # Given the small size, let's just return the edges as paths if we can't do better quickly.
            # But we know O=0 means we can do 2.
            # Let's try to construct 2 paths.
            
            if len(edges) == 1:
                return [[edges[0][0], edges[0][1]]] # 1 path?
                # Wait, 1 edge is 1 path.
                # But O=0 implies even degrees. 1 edge implies degrees 1,1. O=2.
                # So if O=0, len(edges) >= 2 (cycle).
                
            # Pick an edge
            e0 = edges[0]
            u0, v0 = e0
            remaining_edges = set(edges) - {tuple(sorted(e0))}
            
            # Find Eulerian trail from u0 to v0 in remaining_edges
            # This is possible because remaining_edges has exactly two odd vertices u0, v0.
            # We can use Hierholzer's or simple DFS with backtracking (N=30 is small).
            
            def find_eulerian_trail(start, end, rem_edges, adj_rem):
                # Returns list of vertices
                path = [start]
                visited_edges = set()
                
                def dfs(curr):
                    if curr == end:
                        return True
                    for nb in adj_rem[curr]:
                        edge = tuple(sorted((curr, nb)))
                        if edge in rem_edges and edge not in visited_edges:
                            visited_edges.add(edge)
                            path.append(nb)
                            if dfs(nb):
                                return True
                            path.pop()
                            visited_edges.remove(edge)
                    return False
                    
                if dfs(start):
                    return path
                return None

            # Build adj for remaining
            adj_rem = defaultdict(list)
            for u, v in remaining_edges:
                adj_rem[u].append(v)
                adj_rem[v].append(u)
                
            trail = find_eulerian_trail(u0, v0, remaining_edges, adj_rem)
            
            if trail:
                # Trail is a sequence of vertices covering all remaining edges.
                # We need to decompose this trail into simple paths.
                # A trail is a walk.
                # We can just return the trail as one path?
                # No, it might not be simple.
                # But for the purpose of the score, if we return a non-simple path, is it invalid?
                # "Paths must be simple".
                # So we must decompose.
                
                # Heuristic for decomposition:
                # If the trail is simple, return [trail, [u0, v0]]
                # If not, we might need more paths.
                # But for N=30, trails are often simple or close.
                # Let's check if trail is simple.
                if len(set(trail)) == len(trail):
                    return [trail, [u0, v0]]
                else:
                    # Fallback: decompose into simple paths
                    # This is complex. Let's just return edges as paths for safety?
                    # No, that's bad score.
                    # Let's try to extract simple paths from the trail.
                    paths = []
                    used_edges = set()
                    # This is too slow to implement correctly in the scratchpad.
                    # Let's assume the trail is simple or can be treated as such for the "simple" check in the verifier?
                    # The verifier will check if vertices are distinct.
                    # If not, it fails.
                    
                    # Safer fallback for O=0:
                    # Just return 2 paths: one long simple path found by DFS, and the rest?
                    # Actually, for O=0, the minimum is 2.
                    # If we can't find 2 simple paths, we might fail validity.
                    # Let's return the single edge and the rest as a "path" (even if not simple) and hope?
                    # No, must be valid.
                    
                    # Let's use a simple cycle decomposition.
                    # Find a cycle, remove it, repeat.
                    # Each cycle of length L gives 2 paths (1 path of L-1 edges, 1 path of 1 edge).
                    # This guarantees validity.
                    # Number of paths = 2 * (number of cycles).
                    # This might be > 2, but it's valid.
                    
                    all_edges_copy = set(edges)
                    paths = []
                    while all_edges_copy:
                        # Find a cycle
                        # Start from any vertex
                        start = next(iter(all_edges_copy))[0]
                        path = [start]
                        prev = None
                        curr = start
                        cycle_found = False
                        # DFS to find cycle
                        def find_cycle(curr, prev, path, edges_set):
                            for nb in adj[curr]:
                                edge = tuple(sorted((curr, nb)))
                                if edge in edges_set and nb != prev:
                                    if nb == start:
                                        return path + [nb], edge
                                    if nb not in path:
                                        res = find_cycle(nb, curr, path + [nb], edges_set)
                                        if res:
                                            return res
                            return None
                            
                        res = find_cycle(curr, prev, path, all_edges_copy)
                        if res:
                            cyc_path, last_edge = res
                            # cyc_path is vertices
                            # Remove edges in cyc_path
                            for i in range(len(cyc_path) - 1):
                                e = tuple(sorted((cyc_path[i], cyc_path[i+1])))
                                all_edges_copy.remove(e)
                            # cyc_path is a cycle.
                            # Convert to 2 paths:
                            # Path 1: cyc_path[0] to cyc_path[-2]
                            # Path 2: cyc_path[0] to cyc_path[-1] (the last edge)
                            # Wait, cyc_path[-1] is start.
                            # So Path 1: cyc_path[:-1]
                            # Path 2: [cyc_path[-2], cyc_path[-1]]? No.
                            # Cycle: v0-v1-...-vk-v0.
                            # Path 1: v0...vk. Edges (v0,v1)...(v{k-1},vk).
                            # Path 2: vk-v0. Edge (vk,v0).
                            # So:
                            paths.append(cyc_path[:-1])
                            paths.append([cyc_path[-2], cyc_path[-1]]) # Wait, cyc_path[-1] is v0. cyc_path[-2] is vk.
                            # Edge (vk, v0) is the last edge.
                            # So path is [vk, v0].
                            # My cyc_path is [v0, v1, ..., vk, v0].
                            # Path 1: [v0, ..., vk]
                            # Path 2: [vk, v0]
                            # Correct.
                        else:
                            # No cycle? Then it's a tree?
                            # But O=0 means no trees (unless empty).
                            # If no cycle found, break to avoid infinite loop.
                            break
                    
                    return paths

        # O > 0
        # Pair odd vertices
        # Greedy: sort by distance?
        # Just pair them in order for now, or use Hungarian.
        # Given small N, let's use a simple greedy:
        # Calculate all pair distances.
        # Sort pairs by distance.
        # Pick smallest, remove vertices, repeat.
        
        # Dijkstra from all odd vertices?
        # Or just BFS from each.
        
        def bfs_dist(src, adj):
            dist = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            return dist

        # Precompute distances from all odd vertices
        # N is small, O is small.
        
        # Pairing
        odd_list = list(odd_vertices)
        pairs = []
        # Simple greedy:
        # Find closest pair, add to pairs, remove from list.
        # Repeat.
        
        # To avoid O(N^3) if O is large, but O is small.
        
        while odd_list:
            # Find min distance pair
            min_d = float('inf')
            min_i, min_j = -1, -1
            for i in range(len(odd_list)):
                for j in range(i+1, len(odd_list)):
                    u, v = odd_list[i], odd_list[j]
                    # Need distance
                    # Calculate on the fly?
                    # Better to precompute dist matrix for odd vertices.
                    pass
            
            # Precompute dist matrix for current odd_list
            # This is inefficient if done in loop.
            # Let's just use a fixed pairing: (0,1), (2,3)...
            # This is suboptimal but fast and valid.
            # The score is total paths.
            # The number of paths is O/2 + remaining_components.
            # The pairing affects remaining_components.
            # Let's stick to a simple pairing for now.
            
            if len(odd_list) >= 2:
                pairs.append((odd_list[0], odd_list[1]))
                odd_list = odd_list[2:]
            else:
                break

        # Find paths for each pair
        used_edges = set()
        paths = []
        
        for u, v in pairs:
            # Find shortest path from u to v avoiding used_edges
            # BFS
            prev = {u: None}
            q = deque([u])
            found = False
            while q:
                curr = q.popleft()
                if curr == v:
                    found = True
                    break
                for nb in adj[curr]:
                    edge = tuple(sorted((curr, nb)))
                    if edge in used_edges:
                        continue
                    if nb not in prev:
                        prev[nb] = curr
                        q.append(nb)
                        
            if found:
                # Reconstruct path
                path = []
                curr = v
                while curr is not None:
                    path.append(curr)
                    curr = prev[curr]
                path.reverse()
                paths.append(path)
                # Mark edges as used
                for i in range(len(path) - 1):
                    used_edges.add(tuple(sorted((path[i], path[i+1]))))
            else:
                # Should not happen if graph is connected
                pass

        # Handle remaining edges
        remaining_edges = set(edges) - used_edges
        
        # Find connected components in remaining_edges
        # For each component, decompose into paths (cycles)
        
        # Build adj for remaining
        adj_rem = defaultdict(list)
        for u, v in remaining_edges:
            adj_rem[u].append(v)
            adj_rem[v].append(u)
            
        visited = set()
        for node in range(n):
            if node in adj_rem and node not in visited:
                # BFS to get component
                comp_nodes = set()
                q = deque([node])
                visited.add(node)
                while q:
                    curr = q.popleft()
                    comp_nodes.add(curr)
                    for nb in adj_rem[curr]:
                        if nb not in visited:
                            visited.add(nb)
                            q.append(nb)
                
                # Get edges in this component
                comp_edges = set()
                for u, v in remaining_edges:
                    if u in comp_nodes and v in comp_nodes:
                        comp_edges.add(tuple(sorted((u, v))))
                        
                if not comp_edges:
                    continue
                    
                # Decompose this Eulerian component into cycles
                # Use the cycle decomposition logic from before
                # We need adj for this component
                adj_comp = defaultdict(list)
                for u, v in comp_edges:
                    adj_comp[u].append(v)
                    adj_comp[v].append(u)
                    
                edges_set = set(comp_edges)
                while edges_set:
                    start = next(iter(edges_set))[0]
                    # Find cycle
                    # Simple DFS
                    def find_cycle_in_comp(curr, prev, path, edges_set, adj_comp):
                        for nb in adj_comp[curr]:
                            edge = tuple(sorted((curr, nb)))
                            if edge in edges_set and nb != prev:
                                if nb == start:
                                    return path + [nb]
                                if nb not in path:
                                    res = find_cycle_in_comp(nb, curr, path + [nb], edges_set, adj_comp)
                                    if res:
                                        return res
                        return None
                            
                    cyc = find_cycle_in_comp(start, None, [start], edges_set, adj_comp)
                    if cyc:
                        # Remove edges
                        for i in range(len(cyc) - 1):
                            e = tuple(sorted((cyc[i], cyc[i+1])))
                            edges_set.remove(e)
                        # Add 2 paths
                        paths.append(cyc[:-1])
                        paths.append([cyc[-2], cyc[-1]])
                    else:
                        # Fallback: add edges as single paths
                        for e in edges_set:
                            paths.append([e[0], e[1]])
                        break

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Return best effort or trivial
            break
        n, p, seed = spec
        edges = instance(n, p, seed)
        paths = solve_graph(n, edges)
        results.append(paths)
        
    # If time runs out, fill with trivial
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        n, p, seed = spec
        edges = instance(n, p, seed)
        results.append([[list(e)] for e in edges])

    return results

# EVOLVE-BLOCK-END
