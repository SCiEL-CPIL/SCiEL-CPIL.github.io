"""Decompose three fixed instance graphs into few cycles + single edges."""
import time
import sys
from collections import defaultdict, deque
from random import Random

_DEADLINE = 148.0
_START = None

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_flow_bipartite(n, edge_set, adj, deadline):
    """
    Find maximum edge-disjoint cycles via max flow on bipartite graph.
    Left: vertices 0..n-1, Right: vertices 0..n-1.
    Edge (u,v) in original -> edges (u_L, v_R) and (v_L, u_R) with cap 1.
    Source -> each u_L cap 1, each v_R -> sink cap 1.
    Max flow gives max 2-factor (collection of edge-disjoint cycles).
    Returns list of cycles (each cycle is list of vertices) and remaining edges.
    """
    # Build flow network
    # Nodes: source=2n, sink=2n+1, left u = u, right v = n+v
    S = 2 * n
    T = 2 * n + 1
    # Adjacency for flow: list of [to, cap, rev_index]
    flow_adj = [[] for _ in range(2 * n + 2)]
    
    def add_edge(u, v, cap):
        flow_adj[u].append([v, cap, len(flow_adj[v])])
        flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
    
    # Source to left nodes
    for u in range(n):
        add_edge(S, u, 1)
    # Right nodes to sink
    for v in range(n):
        add_edge(n + v, T, 1)
    # Original edges -> bipartite edges
    for (u, v) in edge_set:
        if u == v:
            continue
        add_edge(u, n + v, 1)
        add_edge(v, n + u, 1)
    
    # Edmonds-Karp (BFS-based max flow)
    def bfs_flow():
        # BFS to find augmenting path
        parent = [-1] * (2 * n + 2)
        parent_edge = [-1] * (2 * n + 2)  # index of edge in flow_adj[parent]
        queue = deque([S])
        parent[S] = S
        while queue:
            u = queue.popleft()
            if u == T:
                break
            for i, (v, cap, rev) in enumerate(flow_adj[u]):
                if cap > 0 and parent[v] == -1:
                    parent[v] = u
                    parent_edge[v] = i
                    queue.append(v)
        if parent[T] == -1:
            return 0
        # Find bottleneck
        flow = float('inf')
        v = T
        while v != S:
            u = parent[v]
            i = parent_edge[v]
            flow = min(flow, flow_adj[u][i][1])
            v = u
        # Augment
        v = T
        while v != S:
            u = parent[v]
            i = parent_edge[v]
            flow_adj[u][i][1] -= flow
            flow_adj[v][i][2] += flow  # rev index
            v = u
        return flow
    
    total_flow = 0
    while time.time() < deadline:
        f = bfs_flow()
        if f == 0:
            break
        total_flow += f
    
    # Extract cycles from flow
    # For each original edge (u,v), check if flow went u->n+v or v->n+u
    # Build mapping: for each u, which v it maps to (at most one since cap 1)
    mapping = {}
    for u in range(n):
        for (v, cap, rev) in flow_adj[u]:
            if n <= v < 2*n and cap == 0:  # edge was used (cap reduced from 1 to 0)
                orig_v = v - n
                # Check if original edge (u, orig_v) exists
                e = tuple(sorted((u, orig_v)))
                if e in edge_set:
                    mapping[u] = orig_v
                    break
    
    # Now follow the mapping to extract cycles
    visited = set()
    cycles = []
    for start in range(n):
        if start in visited or start not in mapping:
            continue
        # Follow the cycle
        cycle = [start]
        visited.add(start)
        cur = mapping[start]
        while cur != start:
            if cur in visited:
                break
            cycle.append(cur)
            visited.add(cur)
            if cur not in mapping:
                break
            cur = mapping[cur]
        if len(cycle) >= 3 and cur == start:
            cycles.append(cycle)
    
    # Determine which edges are used in cycles
    used_edges = set()
    for cycle in cycles:
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            used_edges.add(e)
    
    remaining = edge_set - used_edges
    return cycles, remaining

def _find_cycle_in_residual(adj, edge_set, deadline):
    """Find a single cycle in the residual graph using BFS-based approach."""
    if not edge_set:
        return None
    
    # Try to find a cycle by BFS from each vertex
    for start in sorted(edge_set.keys() if isinstance(edge_set, dict) else list(adj.keys())):
        if len(adj[start]) < 2:
            continue
        # BFS to find shortest cycle through start
        # Actually, just do a DFS with limited depth
        best = None
        def dfs(path, used_edges, cur, depth):
            nonlocal best
            if time.time() > deadline:
                return
            if len(path) >= 3 and (cur, start) in edge_set and tuple(sorted((cur, start))) not in used_edges:
                if best is None or len(path) > len(best):
                    best = list(path)
                return
            if depth > 20:  # limit depth
                return
            for nxt in adj[cur]:
                e = tuple(sorted((cur, nxt)))
                if e in edge_set and e not in used_edges and nxt not in path:
                    path.append(nxt)
                    used_edges.add(e)
                    dfs(path, used_edges, nxt, depth + 1)
                    path.pop()
                    used_edges.discard(e)
                    if best is not None and len(path) < len(best) - 1:
                        return
        
        dfs([start], set(), start, 0)
        if best is not None:
            return best
    return None

def _decompose_one(edges, deadline):
    """Decompose a single graph into cycles and edges."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    adj = _adjacency(edges)
    pieces = []
    
    if not edge_set:
        return pieces
    
    # Phase 1: Find maximum 2-factor via max flow
    cycles, remaining = _max_flow_bipartite(len(adj), edge_set, adj, deadline - 2.0)
    pieces.extend(cycles)
    
    # Update adj and edge_set with remaining
    edge_set = remaining
    adj = _adjacency(list(edge_set))
    
    # Phase 2: Greedily find more cycles in residual
    while edge_set and time.time() < deadline - 1.0:
        # Find vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            break
        
        cycle = _find_cycle_in_residual(adj, edge_set, deadline - 1.0)
        if cycle is None:
            break
        
        # Remove cycle edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            if e in edge_set:
                edge_set.remove(e)
                adj[u].discard(v)
                adj[v].discard(u)
        
        pieces.append(cycle)
    
    # Phase 3: Remaining edges as single-edge pieces
    for e in sorted(edge_set):
        pieces.append(list(e))
    
    return pieces

def construct():
    global _START
    _START = time.time()
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    all_pieces = []
    
    # Allocate time: give more time to larger graphs
    total_edges = []
    for spec in specs:
        edges = _instance(*spec)
        total_edges.append(len(edges))
    
    total = sum(total_edges)
    time_per_graph = [max(5.0, (_DEADLINE - 5.0) * (te / total)) for te in total_edges]
    
    for i, spec in enumerate(specs):
        if time.time() > _DEADLINE - 3.0:
            # Fallback
            edges = _instance(*spec)
            all_pieces.append([list(e) for e in edges])
            continue
        
        edges = _instance(*spec)
        deadline = _START + 3.0 + sum(time_per_graph[:i]) + time_per_graph[i]
        deadline = min(deadline, _DEADLINE - 2.0)
        
        pieces = _decompose_one(edges, deadline)
        all_pieces.append(pieces)
    
    return all_pieces
