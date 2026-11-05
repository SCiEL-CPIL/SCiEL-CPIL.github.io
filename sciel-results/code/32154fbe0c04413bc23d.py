import time
from collections import defaultdict
from random import Random
import sys

def instance_graph(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def find_longest_cycle(adj, n):
    """Find a long cycle in the graph using DFS with backtracking."""
    best_cycle = None
    best_len = 0
    
    # Try from each vertex as start
    for start in range(n):
        if not adj[start]:
            continue
        # DFS to find cycles through start
        visited = set()
        path = [start]
        visited.add(start)
        
        def dfs(node, path, visited):
            nonlocal best_cycle, best_len
            for neighbor in adj[node]:
                if neighbor == start and len(path) >= 3:
                    # Found a cycle
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle[:]
                elif neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        dfs(start, path, visited)
    
    return best_cycle

def decompose_greedy(edges, n):
    """Greedy cycle extraction: repeatedly find longest cycle."""
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    cycles = []
    
    while remaining:
        cycle = find_longest_cycle(adj, n)
        if cycle is None:
            break
        
        # Remove cycle edges
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge_key = tuple(sorted((u, v)))
            if edge_key in remaining:
                remaining.remove(edge_key)
                adj[u].discard(v)
                adj[v].discard(u)
                if not adj[u]:
                    del adj[u]
                if not adj[v]:
                    del adj[v]
        
        cycles.append(cycle)
    
    edge_pieces = [list(e) for e in remaining]
    return cycles, edge_pieces

def try_extend_cycle(cycle, adj, n):
    """Try to find a longer cycle that uses at least one edge from the given cycle."""
    # Get edges in the cycle
    cycle_edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add(tuple(sorted((u, v))))
    
    # Build a subgraph with only edges that are in adj
    # Try to find a cycle that uses at least one cycle edge and is longer
    best = None
    best_len = len(cycle)
    
    # For each edge in the cycle, try to find a longer cycle through that edge
    for edge in cycle_edges:
        u, v = edge
        # Find a path from u to v that doesn't use this edge, with length > 1
        # i.e., a path u -> ... -> v with at least 2 edges
        # Combined with edge (u,v), this forms a cycle
        
        # BFS/DFS from u to v avoiding the direct edge
        visited = set([u])
        parent = {u: None}
        queue = [u]
        found = False
        end_node = None
        
        while queue and not found:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if neighbor == v and node != u:
                    # Found a path from u to v
                    # Reconstruct path
                    path = [v]
                    curr = node
                    while curr is not None:
                        path.append(curr)
                        curr = parent[curr]
                    path.reverse()
                    # Cycle is path (which goes from u to v) plus edge v->u
                    # But we need the cycle in order
                    cycle_candidate = path
                    if len(cycle_candidate) > best_len:
                        best_len = len(cycle_candidate)
                        best = cycle_candidate[:]
                    found = True
                    break
                if neighbor not in visited and (node, neighbor) != edge and (neighbor, node) != edge:
                    visited.add(neighbor)
                    parent[neighbor] = node
                    queue.append(neighbor)
    
    return best

def local_search(cycles, edge_pieces, adj_original, n, time_limit=5):
    """Improve the decomposition by trying to extend cycles."""
    start_time = time.time()
    remaining = set(tuple(sorted(e)) for e in edge_pieces)
    adj = defaultdict(set)
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    # Rebuild adj including cycle edges for reference
    # Actually, we work with the current state
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        new_cycles = []
        for cycle in cycles:
            # Try to extend this cycle
            extended = try_extend_cycle(cycle, adj, n)
            if extended is not None and len(extended) > len(cycle):
                # Remove old cycle edges from adj, add new cycle edges
                # First, add old cycle edges back to remaining/adj
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge_key = tuple(sorted((u, v)))
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Remove new cycle edges
                for i in range(len(extended)):
                    u = extended[i]
                    v = extended[(i + 1) % len(extended)]
                    edge_key = tuple(sorted((u, v)))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    if not adj[u]:
                        del adj[u]
                    if not adj[v]:
                        del adj[v]
                
                new_cycles.append(extended)
                improved = True
            else:
                new_cycles.append(cycle)
        
        cycles = new_cycles
    
    # Rebuild edge pieces from adj
    edge_pieces = []
    for u in list(adj.keys()):
        for v in adj[u]:
            if u < v:
                edge_pieces.append([u, v])
    
    return cycles, edge_pieces

def decompose_one(edges, n, deadline):
    """Decompose one graph with time limit."""
    cycles, edge_pieces = decompose_greedy(edges, n)
    
    # Local search improvement
    adj_original = defaultdict(set)
    for u, v in edges:
        adj_original[u].add(v)
        adj_original[v].add(u)
    
    if (deadline - time.time()) > 2:
        cycles, edge_pieces = local_search(cycles, edge_pieces, adj_original, n, 
                                           time_limit=max(1, deadline - time.time() - 1))
    
    pieces = cycles + edge_pieces
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    for n, p, seed in specs:
        if time.time() > deadline:
            results.append([])
            continue
        
        edges = instance_graph(n, p, seed)
        pieces = decompose_one(edges, n, deadline)
        results.append(pieces)
    
    return results
