import time
import random
import sys
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    graphs = [instance(*spec) for spec in specs]

    def max_flow(s, t, adj, cap, n):
        """Edmonds-Karp max flow. adj: list of sets, cap: dict of (u,v)->int"""
        flow = 0
        while True:
            # BFS to find augmenting path
            parent = [-1] * n
            parent[s] = s
            q = deque([s])
            found = False
            while q:
                u = q.popleft()
                for v in adj[u]:
                    c = cap.get((u, v), 0)
                    if c > 0 and parent[v] == -1:
                        parent[v] = u
                        if v == t:
                            found = True
                            break
                        q.append(v)
                if found:
                    break
            if not found:
                break
            # Find bottleneck
            path_flow = float('inf')
            v = t
            while v != s:
                u = parent[v]
                path_flow = min(path_flow, cap.get((u, v), 0))
                v = u
            # Update capacities
            v = t
            while v != s:
                u = parent[v]
                cap[(u, v)] = cap.get((u, v), 0) - path_flow
                cap[(v, u)] = cap.get((v, u), 0) + path_flow
                v = u
            flow += path_flow
        return flow

    def extract_vertex_disjoint_cycles(remaining_edges, time_limit):
        """
        Find a maximum set of vertex-disjoint cycles in the graph defined by remaining_edges.
        Returns list of cycles (each a list of vertices).
        """
        if not remaining_edges:
            return []

        adj = defaultdict(set)
        for u, v in remaining_edges:
            adj[u].add(v)
            adj[v].add(u)

        # Only consider vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            return []

        # Sort candidates by degree descending for better packing
        candidates.sort(key=lambda x: len(adj[x]), reverse=True)

        # We'll greedily pick cycles using a flow-based approach on the current graph.
        # Actually, for small graphs, let's use a simpler iterative approach:
        # Repeatedly find a cycle, remove its vertices, until no more cycles.
        # But to maximize count, we want vertex-disjoint cycles.
        # Strategy: iteratively find the shortest cycle (to leave room for more), 
        # or use a matching-based approach.
        # For these small graphs, let's do: find all cycles via DFS, then greedily 
        # select vertex-disjoint ones preferring shorter cycles first (more cycles total).
        
        # Alternative: use the fact that in a 2-core, we can decompose into cycles.
        # Let's do: find 2-core, then decompose.
        
        # Find 2-core
        deg = {v: len(adj[v]) for v in adj}
        q = deque([v for v in deg if deg[v] < 2])
        in_core = set(adj.keys())
        while q:
            v = q.popleft()
            if v not in in_core:
                continue
            in_core.discard(v)
            for nb in adj[v]:
                if nb in in_core:
                    deg[nb] -= 1
                    if deg[nb] < 2:
                        q.append(nb)
        
        if not in_core:
            return []
        
        # Build core subgraph
        core_adj = defaultdict(set)
        for v in in_core:
            for nb in adj[v]:
                if nb in in_core:
                    core_adj[v].add(nb)
        
        # Decompose 2-core into cycles
        # A 2-core can be decomposed into edge-disjoint cycles.
        # We want vertex-disjoint cycles for maximum count.
        # Greedy: repeatedly find a cycle in the remaining core, remove its vertices.
        
        remaining_vertices = set(in_core)
        remaining_edges_local = set()
        for v in remaining_vertices:
            for nb in core_adj[v]:
                if v < nb and nb in remaining_vertices:
                    remaining_edges_local.add((v, nb))
        
        cycles = []
        while remaining_edges_local:
            # Find a cycle in the remaining subgraph
            sub_adj = defaultdict(set)
            for u, v in remaining_edges_local:
                sub_adj[u].add(v)
                sub_adj[v].add(u)
            
            # Find shortest cycle via BFS from each vertex
            best_cycle = None
            best_len = float('inf')
            
            for start in list(sub_adj.keys()):
                if len(sub_adj[start]) < 2:
                    continue
                # BFS to find shortest cycle through start
                dist = {start: 0}
                parent = {start: None}
                q = deque([start])
                while q:
                    u = q.popleft()
                    for nb in sub_adj[u]:
                        if nb == start:
                            # Found a cycle
                            cycle = [start]
                            cur = u
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            if len(cycle) < best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                            break
                        elif nb not in dist:
                            dist[nb] = dist[u] + 1
                            parent[nb] = u
                            q.append(nb)
                        elif nb == parent.get(u, None) or dist[nb] >= dist[u]:
                            continue
                        else:
                            # Back edge found cycle
                            cycle = [start]
                            cur = u
                            while cur != start:
                                cycle.append(cur)
                                cur = parent[cur]
                            cycle.reverse()
                            # The cycle is start -> ... -> u -> nb -> ... -> start
                            # Actually this is getting complex. Let's use a simpler method.
                            pass
                if best_cycle is not None and len(best_cycle) <= 3:
                    break  # Found a triangle, can't do better
            
            if best_cycle is None:
                # Try DFS for any cycle
                for start in list(sub_adj.keys()):
                    if len(sub_adj[start]) < 2:
                        continue
                    visited = set()
                    stack = [(start, [start])]
                    found = False
                    while stack and not found:
                        cur, path = stack.pop()
                        visited.add(cur)
                        for nb in sub_adj[cur]:
                            if nb == start and len(path) >= 3:
                                best_cycle = path
                                found = True
                                break
                            if nb not in visited:
                                stack.append((nb, path + [nb]))
                    if found:
                        break
            
            if best_cycle is None:
                break
            
            cycles.append(best_cycle)
            
            # Remove vertices of this cycle
            for v in best_cycle:
                remaining_vertices.discard(v)
            
            # Rebuild remaining edges
            new_remaining = set()
            for u, v in remaining_edges_local:
                if u in remaining_vertices and v in remaining_vertices:
                    new_remaining.add((u, v))
            remaining_edges_local = new_remaining
        
        return cycles

    def decompose_graph(edges, time_limit):
        if not edges:
            return []

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        remaining_edges = set(edges)
        pieces = []

        # Phase 1: Extract vertex-disjoint cycles iteratively
        # We'll do this in rounds: each round, find max set of vertex-disjoint cycles
        while remaining_edges and time.time() < time_limit:
            cycles = extract_vertex_disjoint_cycles(remaining_edges, time_limit)
            if not cycles:
                break
            for cycle in cycles:
                pieces.append(cycle)
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
                u, v = cycle[-1], cycle[0]
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)

        # Phase 2: For remaining edges, try to find longer cycles (not necessarily vertex-disjoint)
        # Repeatedly find the longest cycle in the remaining graph
        while remaining_edges and time.time() < time_limit:
            # Build adjacency of remaining
            rem_adj = defaultdict(set)
            for u, v in remaining_edges:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Find longest cycle using DFS with pruning
            candidates = [v for v in rem_adj if len(rem_adj[v]) >= 2]
            if not candidates:
                break
            candidates.sort(key=lambda x: len(rem_adj[x]), reverse=True)
            
            best_cycle = None
            best_len = 0
            max_depth = min(25, len(rem_adj))
            
            for start in candidates[:20]:
                # DFS to find longest cycle
                stack = [(start, [start], frozenset([start]))]
                while stack:
                    cur, path, visited = stack.pop()
                    if len(path) >= 3:
                        if (cur, start) in remaining_edges or (start, cur) in remaining_edges:
                            if len(path) > best_len:
                                best_len = len(path)
                                best_cycle = list(path)
                    for nb in rem_adj[cur]:
                        if nb == start:
                            continue
                        if nb not in visited:
                            new_path = path + [nb]
                            if len(new_path) >= max_depth:
                                continue
                            if len(new_path) + 1 <= best_len:
                                continue
                            stack.append((nb, new_path, visited | {nb}))
            
            if best_cycle is None:
                break
            
            pieces.append(best_cycle)
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i+1]
                e = (u, v) if u < v else (v, u)
                remaining_edges.discard(e)
            u, v = best_cycle[-1], best_cycle[0]
            e = (u, v) if u < v else (v, u)
            remaining_edges.discard(e)

        # Phase 3: Remaining edges as single pieces
        for e in remaining_edges:
            pieces.append(list(e))

        return pieces

    results = []
    for i, edges in enumerate(graphs):
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        time_limit = time.time() + max(1.0, time_per_graph * 0.9)

        pieces = decompose_graph(edges, time_limit)
        results.append(pieces)

        if time.time() > deadline - 2.0:
            for j in range(i + 1, len(graphs)):
                results.append([list(e) for e in graphs[j]])
            break

    return results

if __name__ == "__main__":
    res = construct()
    for i, spec in enumerate([(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]):
        n, p, seed = spec
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        edge_set = set(edges)

        used_edges = set()
        valid = True
        for piece in res[i]:
            if len(piece) == 2:
                u, v = piece
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            elif len(piece) >= 3:
                if len(set(piece)) != len(piece):
                    valid = False
                    break
                for k in range(len(piece) - 1):
                    u, v = piece[k], piece[k+1]
                    e = tuple(sorted((u, v)))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break
                u, v = piece[-1], piece[0]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            else:
                valid = False
                break

        if valid and used_edges != edge_set:
            valid = False

        print(f"Graph {i}: Valid={valid}, Pieces={len(res[i])}")
