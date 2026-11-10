import sys
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_bridges(n, adj):
        # adj: list of sets of neighbors
        visited = [False] * n
        disc = [0] * n
        low = [0] * n
        parent = [-1] * n
        bridges = set()
        timer = 0

        def dfs(u):
            nonlocal timer
            visited[u] = True
            disc[u] = timer
            low[u] = timer
            timer += 1
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    dfs(v)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u]:
                        bridges.add((min(u, v), max(u, v)))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(n):
            if not visited[i]:
                dfs(i)
        return bridges

    def find_long_cycle(n, adj, current_edges):
        # adj: list of sets
        # current_edges: set of tuples (u, v) with u < v
        
        best_cycle = None
        best_len = 0
        
        # Sort vertices by degree descending to start from high-degree nodes
        # We only consider vertices that are part of current_edges
        active_vertices = set()
        for u, v in current_edges:
            active_vertices.add(u)
            active_vertices.add(v)
            
        # Get degrees in the current graph
        degs = {v: 0 for v in active_vertices}
        for u, v in current_edges:
            degs[u] += 1
            degs[v] += 1
            
        # Try starting from top 10 highest degree vertices
        start_candidates = sorted(active_vertices, key=lambda x: degs[x], reverse=True)[:10]
        
        for start in start_candidates:
            # DFS to find a cycle
            # path: list of vertices
            # visited_in_path: set of vertices in current path
            # We use recursion or stack. Stack is safer for deep paths, but N=30 is small.
            
            def dfs_cycle(u, path, visited_in_path):
                nonlocal best_cycle, best_len
                if time.time() > deadline:
                    return
                
                for v in sorted(adj[u], key=lambda x: degs.get(x, 0), reverse=True):
                    # Check if edge (u, v) is in current_edges
                    edge = (min(u, v), max(u, v))
                    if edge not in current_edges:
                        continue
                    
                    if v in visited_in_path:
                        # Found a cycle
                        # Extract cycle from path
                        if v == start:
                            # The cycle is the whole path
                            cycle = list(path)
                            # Ensure distinct vertices (path is simple by construction)
                            if len(cycle) >= 3:
                                if len(cycle) > best_len:
                                    best_len = len(cycle)
                                    best_cycle = cycle
                        else:
                            # v is in path but not start
                            # Find index of v in path
                            try:
                                idx = path.index(v)
                                cycle = path[idx:]
                                if len(cycle) >= 3:
                                    if len(cycle) > best_len:
                                        best_len = len(cycle)
                                        best_cycle = cycle
                            except ValueError:
                                pass
                    else:
                        # Visit v
                        visited_in_path.add(v)
                        path.append(v)
                        dfs_cycle(v, path, visited_in_path)
                        path.pop()
                        visited_in_path.remove(v)
                        
            visited_in_path = {start}
            path = [start]
            dfs_cycle(start, path, visited_in_path)
            
            if best_cycle and best_len >= 10: # Heuristic: if we found a reasonably long cycle, stop searching other starts
                break
                
        return best_cycle

    def decompose(n, edges):
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        current_edges = set(edges)
        pieces = []
        
        while current_edges:
            if time.time() > deadline:
                break # Timeout safeguard
            
            # 1. Find bridges in the current graph
            # Rebuild adj for current_edges? 
            # We can update adj incrementally or just rebuild. N is small, rebuild is fine.
            curr_adj = defaultdict(set)
            for u, v in current_edges:
                curr_adj[u].add(v)
                curr_adj[v].add(u)
                
            bridges = find_bridges(n, curr_adj)
            
            if bridges:
                # Remove all bridges
                for b in bridges:
                    pieces.append([b[0], b[1]])
                    current_edges.remove(b)
                # Loop back to re-evaluate bridges
                continue
            
            # 2. No bridges, so graph is 2-edge-connected (or single edge? No, single edge is a bridge)
            # Find a long cycle
            best_cycle = find_long_cycle(n, curr_adj, current_edges)
            
            if best_cycle:
                pieces.append(best_cycle)
                # Remove edges of this cycle from current_edges
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = (min(u, v), max(u, v))
                    if edge in current_edges:
                        current_edges.remove(edge)
            else:
                # Fallback: take any edge
                if current_edges:
                    e = next(iter(current_edges))
                    pieces.append([e[0], e[1]])
                    current_edges.remove(e)
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        res = decompose(n, edges)
        results.append(res)
        
    return results
