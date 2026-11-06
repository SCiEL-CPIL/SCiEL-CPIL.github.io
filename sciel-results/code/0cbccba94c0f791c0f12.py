"""Decompose graphs into cycles and edges to minimize piece count."""
import time
from random import Random
from collections import defaultdict

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(adj, n, max_len=None, deadline=None):
    """Find a long cycle in the graph using DFS with pruning."""
    if deadline is not None and time.time() > deadline:
        return None
    
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex
    for start in range(n):
        if start in adj and len(adj[start]) < 2:
            continue
        
        # DFS to find cycles starting and ending at start
        visited = {start: 0}  # vertex -> position in path
        path = [start]
        
        def dfs(v, depth):
            nonlocal best_cycle, best_len
            if deadline is not None and time.time() > deadline:
                return
            
            for w in sorted(adj.get(v, [])):
                if w == start and depth >= 3:
                    # Found a cycle
                    if depth > best_len:
                        best_len = depth
                        best_cycle = path[:]
                    return
                if w not in visited:
                    visited[w] = depth
                    path.append(w)
                    dfs(w, depth + 1)
                    path.pop()
                    del visited[w]
        
        dfs(start, 1)
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best_cycle

def _find_cycles_greedy(adj, n, deadline=None):
    """Find multiple cycles greedily, preferring longer ones."""
    cycles = []
    remaining_adj = {i: set(s) for i, s in adj.items()}
    
    max_iterations = 100
    for _ in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the longest cycle
        cycle = _find_longest_cycle(remaining_adj, n, deadline=deadline)
        if cycle is None:
            break
        
        cycles.append(cycle)
        
        # Remove cycle edges from remaining graph
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            if u in remaining_adj and v in remaining_adj[u]:
                remaining_adj[u].discard(v)
            if v in remaining_adj and u in remaining_adj[v]:
                remaining_adj[v].discard(u)
    
    return cycles

def _decompose(edges, n, deadline=None):
    """Decompose edges into cycles and single edges."""
    if not edges:
        return []
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find cycles greedily
    cycles = _find_cycles_greedy(adj, n, deadline=deadline)
    
    # Build piece list
    pieces = []
    used_edges = set()
    
    for cycle in cycles:
        cycle_edges = set()
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            cycle_edges.add(edge)
        
        # Verify all edges are still available
        valid = True
        for e in cycle_edges:
            if e in used_edges:
                valid = False
                break
        if valid:
            pieces.append(cycle[:])
            used_edges.update(cycle_edges)
    
    # Add remaining edges as single-edge pieces
    for u, v in edges:
        edge = tuple(sorted((u, v)))
        if edge not in used_edges:
            pieces.append([u, v])
            used_edges.add(edge)
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        if time.time() > deadline:
            # Return what we have so far
            break
        
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, deadline=deadline)
        results.append(pieces)
    
    # If we didn't process all graphs, add empty lists for remaining
    while len(results) < len(specs):
        results.append([])
    
    return results
