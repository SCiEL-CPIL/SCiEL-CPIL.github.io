"""Rewritten program: decompose each instance graph into edge-disjoint cycles and edges, minimizing pieces."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque
import math

def construct():
    deadline = time.time() + 145  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def get_odd_vertices(edges, n):
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return [v for v in range(n) if deg[v] % 2 == 1]

    def shortest_path_dist(adj, n):
        """Compute all-pairs shortest path distances using BFS from each node in T."""
        # Since we only need distances between T nodes, and n is small, we can do BFS from each T node.
        # But T can be up to n. n=30 is small.
        dist = {}
        for src in adj.keys():
            # BFS
            q = deque([src])
            d = {src: 0}
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            dist[src] = d
        return dist

    def find_min_t_join(adj, n, odd_verts, edges):
        """Find the minimum T-join using Minimum Perfect Matching on the metric closure of T."""
        if not odd_verts:
            return set()
        
        # Map odd vertices to indices
        t_map = {v: i for i, v in enumerate(odd_verts)}
        m = len(odd_verts)
        
        # Compute distances between all pairs in T
        # Since n is small, we can compute shortest paths in the original graph.
        # We need the shortest path *edges*, not just distances.
        # Let's store the path itself.
        
        # First, compute shortest paths from each odd vertex to all others.
        paths = {} # (src, dst) -> list of edges
        for i, src in enumerate(odd_verts):
            # BFS to find shortest path to all other odd vertices
            prev = {src: None}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in prev:
                        prev[v] = u
                        q.append(v)
            for j, dst in enumerate(odd_verts):
                if i == j:
                    continue
                if dst in prev:
                    # Reconstruct path
                    path_edges = []
                    curr = dst
                    while curr != src:
                        p = prev[curr]
                        e = tuple(sorted((p, curr)))
                        path_edges.append(e)
                        curr = p
                    paths[(i, j)] = path_edges
                    paths[(j, i)] = path_edges # Symmetric
        
        # Now we have a complete graph on T with edge weights = len(path)
        # We need to find a Minimum Perfect Matching.
        # Since m is even (sum of degrees is even), and m <= 30.
        # We can use a simple DP for Matching? m=30 is too big for 2^m DP.
        # We can use a greedy heuristic or a simple augmenting path algorithm for matching?
        # Actually, for m <= 30, a simple randomized greedy with restarts might work, 
        # or we can use the fact that it's a metric TSP path? No, it's Matching.
        
        # Let's use a simple greedy:
        # Sort all pairs by distance.
        # Pick the smallest, add to matching, remove vertices.
        # This is not optimal, but good enough?
        # Better: Randomized Greedy with multiple restarts.
        
        best_matching = None
        best_cost = float('inf')
        
        for _ in range(100): # 100 restarts for matching
            if time.time() > deadline - 5:
                break
            used = [False] * m
            matching = []
            cost = 0
            # Shuffle pairs
            pairs = [(i, j) for i in range(m) for j in range(i+1, m)]
            random.shuffle(pairs)
            # Sort by distance
            pairs.sort(key=lambda x: len(paths[x]) if x in paths else float('inf'))
            
            for i, j in pairs:
                if not used[i] and not used[j]:
                    used[i] = True
                    used[j] = True
                    matching.append((i, j))
                    cost += len(paths[(i, j)])
                    
            if cost < best_cost:
                best_cost = cost
                best_matching = matching[:]
                
        # Reconstruct edges of the T-join
        join_edges = set()
        if best_matching:
            for i, j in best_matching:
                for e in paths[(i, j)]:
                    join_edges.add(e)
                    
        return join_edges

    def decompose_eulerian(edges, n, time_budget):
        """Decompose an Eulerian graph (all degrees even) into cycles.
        Uses a greedy approach to find cycles.
        To minimize the number of cycles, we try to find long cycles.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = build_adjacency(edge_set)
        pieces = []
        
        # We will iteratively find a cycle, remove it, and repeat.
        # To encourage long cycles, we can use a randomized DFS that tries to avoid closing the loop early.
        # Or simply run DFS and keep the longest cycle found in a set of attempts.
        
        while edge_set:
            if time.time() > deadline - 2:
                # Fallback: break remaining into edges? 
                # But it's Eulerian, so we can just find any cycle.
                # For safety, just add remaining edges as single edges if we run out of time?
                # No, that breaks validity if they don't form cycles.
                # But if we run out of time, we return what we have + remaining edges as single edges.
                # This might be invalid if the remaining edges don't form cycles?
                # The task says "A piece is a single edge [u, v] OR a cycle".
                # So remaining edges CAN be single edges.
                # So fallback is safe.
                for e in edge_set:
                    pieces.append(list(e))
                break
                
            # Find a cycle
            # Pick a random start edge to ensure variety
            start_edge = random.choice(list(edge_set))
            u, v = start_edge
            
            # Try to find a path from v to u in the graph excluding start_edge
            # This forms a cycle.
            # We want a "long" path.
            
            best_cycle = None
            best_len = 0
            
            # Try a few DFS runs to find a longer cycle
            for _ in range(20): # 20 attempts per cycle extraction
                if time.time() > deadline - 2:
                    break
                
                # DFS from v to u
                # We must not use start_edge in the path
                # We can use other edges multiple times? No, simple path.
                # So we need a simple path from v to u in G \ {start_edge}.
                
                visited = {v}
                path = [v]
                stack = [(v, iter(adj[v]))]
                
                # We need to handle the exclusion of start_edge carefully.
                # When exploring neighbors of x, if (x, y) is start_edge and y is the other end, skip?
                # Actually, start_edge is (u, v).
                # We are going from v to u.
                # So we should not use (v, u) or (u, v) as the first/last step?
                # Actually, the path is v -> ... -> u.
                # The edge (v, u) is the closing edge.
                # So we just need a path from v to u in the graph with start_edge removed.
                
                rem_adj = defaultdict(set)
                for x, nbrs in adj.items():
                    for y in nbrs:
                        if (x, y) != start_edge and (y, x) != start_edge:
                            rem_adj[x].add(y)
                
                # DFS
                found = False
                stack = [(v, iter(list(rem_adj[v])))]
                visited = {v}
                path = [v]
                
                while stack:
                    node, it = stack[-1]
                    advanced = False
                    for nb in it:
                        if nb == u:
                            # Found path
                            path.append(u)
                            found = True
                            break
                        if nb not in visited:
                            visited.add(nb)
                            path.append(nb)
                            stack.append((nb, iter(list(rem_adj[nb]))))
                            advanced = True
                            break
                    if found:
                        break
                    if not advanced:
                        stack.pop()
                        path.pop()
                        if not stack:
                            break
                            
                if found:
                    # Cycle is path + closing edge
                    # path is [v, ..., u]
                    cycle = path[:]
                    # Check length
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
            
            if best_cycle:
                # Add cycle to pieces
                pieces.append(best_cycle[:])
                # Remove edges from edge_set
                for i in range(len(best_cycle)):
                    x, y = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
                    e = tuple(sorted((x, y)))
                    edge_set.discard(e)
                    # Update adj
                    adj[x].discard(y)
                    adj[y].discard(x)
                    if not adj[x]:
                        del adj[x]
                    if not adj[y]:
                        del adj[y]
            else:
                # Should not happen if Eulerian, but fallback
                e = tuple(next(iter(edge_set)))
                edge_set.discard(e)
                pieces.append(list(e))
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
                
        return pieces

    def decompose(edges, n, time_budget):
        if not edges:
            return []

        edge_set = set(tuple(sorted(e)) for e in edges)
        adj = build_adjacency(edge_set)
        
        # 1. Find Odd Vertices
        odd_verts = get_odd_vertices(edge_set, n)
        
        # 2. Find Min T-Join
        if odd_verts:
            join_edges = find_min_t_join(adj, n, odd_verts, edge_set)
        else:
            join_edges = set()
            
        # 3. Separate Edges
        eulerian_edges = edge_set - join_edges
        join_edge_list = [list(e) for e in join_edges]
        
        # 4. Decompose Eulerian Part
        cycles = decompose_eulerian(eulerian_edges, n, time_budget)
        
        # 5. Combine
        # The T-join edges are single edges.
        # Wait, can the T-join edges form cycles among themselves?
        # We established that Min T-join is a forest of paths.
        # So they must be single edges.
        
        return cycles + join_edge_list

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() > deadline - 5:
            # Fallback: all edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n, deadline - time.time())
        results.append(pieces)

    return results
# EVOLVE-BLOCK-END
