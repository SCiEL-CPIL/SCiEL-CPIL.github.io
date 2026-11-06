"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _hopcroft_karp(n, edges):
    """
    Finds maximum matching in the bipartite double cover of the graph.
    Left nodes: 0..n-1
    Right nodes: 0..n-1
    Edge (u, v) in original graph creates (L_u, R_v) and (L_v, R_u) in bipartite graph.
    Returns matching array of size n, where matching[i] = j means L_i is matched to R_j.
    """
    # Build adjacency list for the bipartite graph
    # adj[i] is list of right nodes j such that (i, j) is an edge in bipartite graph
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # pair_u[i] = j means left node i is matched to right node j
    # pair_v[j] = i means right node j is matched to left node i
    pair_u = [-1] * n
    pair_v = [-1] * n
    
    INF = float('inf')
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                # If right node v is matched to left node w, and we can improve w's distance
                w = pair_v[v]
                if w != -1 and dist[w] == INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
                elif w == -1:
                    found = True
        return found

    def dfs(u):
        for v in adj[u]:
            w = pair_v[v]
            if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False

    matching_size = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1
                    
    return pair_u

def _solve_graph(n, p, seed, time_budget=10.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    # Build edge set for quick lookup
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    
    # Find maximum matching in bipartite double cover
    matching = _hopcroft_karp(n, edges)
    
    # Reconstruct paths and cycles
    # succ[u] = v if matching[u] == v
    succ = [None] * n
    pred = [None] * n
    for u in range(n):
        v = matching[u]
        if v is not None:
            succ[u] = v
            pred[v] = u
            
    paths = []
    visited = [False] * n
    
    # Find paths starting from nodes with no predecessor
    for u in range(n):
        if pred[u] is None and not visited[u]:
            path = [u]
            visited[u] = True
            curr = u
            while succ[curr] is not None:
                curr = succ[curr]
                if visited[curr]:
                    # This would imply a cycle reachable from a path start, 
                    # which shouldn't happen in a functional graph derived from matching
                    break
                path.append(curr)
                visited[curr] = True
            if len(path) > 1:
                paths.append(path)
            else:
                # Single vertex, no edges? Should not happen if m>0 and connected?
                # Or it's an isolated vertex in the matching subgraph.
                pass
                
    # Find cycles
    for u in range(n):
        if not visited[u]:
            # u is part of a cycle
            cycle = [u]
            visited[u] = True
            curr = u
            while succ[curr] is not None and not visited[succ[curr]]:
                curr = succ[curr]
                cycle.append(curr)
                visited[curr] = True
            # If we broke because visited[succ[curr]] is True, it's a cycle back to start
            # Or if succ[curr] is None, it's a path that was missed? 
            # In a matching, if pred[u] is not None, u is in a cycle or a path.
            # We already processed paths starting from pred==None.
            # So any unvisited node must be in a cycle.
            if len(cycle) > 1:
                paths.append(cycle)
            else:
                # Self loop? Not possible in simple graph.
                pass

    # Verify validity
    covered_edges = set()
    valid = True
    for path in paths:
        if len(path) < 2:
            valid = False
            break
        if len(set(path)) != len(path):
            valid = False
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            e = (min(a, b), max(a, b))
            if e in covered_edges:
                valid = False
                break
            if e not in edge_set:
                valid = False
                break
            covered_edges.add(e)
        if not valid:
            break
    
    if not valid or len(covered_edges) != len(edge_set):
        # Fallback: greedy
        paths = _greedy_solve(n, edges, time_budget)
        
    return paths

def _greedy_solve(n, edges, time_budget=10.0):
    """Fallback greedy path decomposition."""
    adj = defaultdict(set)
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    rem_adj = defaultdict(set)
    for (u, v) in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    paths = []
    start_time = time.time()
    
    while remaining_edges:
        if time.time() - start_time > time_budget:
            for e in remaining_edges:
                paths.append([e[0], e[1]])
            break
            
        # Find a vertex with max degree to start
        start_v = -1
        max_deg = -1
        for v in range(n):
            if len(rem_adj[v]) > max_deg:
                max_deg = len(rem_adj[v])
                start_v = v
        
        if start_v == -1:
            break
            
        path = [start_v]
        visited_in_path = {start_v}
        current = start_v
        
        while True:
            next_v = -1
            max_next_deg = -1
            # Prefer unvisited neighbors
            for neighbor in rem_adj[current]:
                if neighbor not in visited_in_path:
                    if len(rem_adj[neighbor]) > max_next_deg:
                        max_next_deg = len(rem_adj[neighbor])
                        next_v = neighbor
            
            if next_v == -1:
                break
            
            path.append(next_v)
            visited_in_path.add(next_v)
            
            e = (min(current, next_v), max(current, next_v))
            remaining_edges.discard(e)
            rem_adj[current].discard(next_v)
            rem_adj[next_v].discard(current)
            
            current = next_v
            
            if not any(nb not in visited_in_path for nb in rem_adj[current]):
                break
                
        if len(path) > 1:
            paths.append(path)
            
    return paths

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 5
            time_budget = max(1.0, min(20.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
