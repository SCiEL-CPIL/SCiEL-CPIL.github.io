"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_cycle_dfs(adj, n, start, rng, max_depth=50):
    """Find a cycle through start using randomized DFS with backtracking."""
    visited = {start}
    path = [start]
    stack = [(start, list(adj[start]))]
    
    while stack:
        node, neighbors = stack[-1]
        if not neighbors:
            stack.pop()
            path.pop()
            continue
        nxt = neighbors.pop()
        if nxt == start and len(path) >= 3:
            return list(path)
        if nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            stack.append((nxt, list(adj[nxt])))
            if len(path) > max_depth:
                # Too deep, backtrack
                stack.pop()
                path.pop()
    
    return None

def _find_longest_cycle(adj, n, max_walks=300):
    """Find a long cycle using multiple randomized DFS attempts."""
    best_cycle = None
    best_len = 0
    
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    rng = Random(42)
    
    # First pass: try each candidate vertex once
    for start in candidates:
        cycle = _find_cycle_dfs(adj, n, start, rng)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
    
    # Second pass: random restarts with different neighbor orderings
    for _ in range(max_walks - len(candidates)):
        if not candidates:
            break
        start = rng.choice(candidates)
        cycle = _find_cycle_dfs(adj, n, start, rng)
        if cycle and len(cycle) > best_len:
            best_len = len(cycle)
            best_cycle = cycle
    
    return best_cycle

def _decompose(edges, n):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    max_iterations = 300
    iters = 0
    no_progress = 0
    
    while remaining and iters < max_iterations:
        # Try to find a cycle
        cycle = _find_longest_cycle(adj, n, max_walks=200)
        
        if cycle is None or len(cycle) < 3:
            no_progress += 1
            if no_progress > 5:
                break
            iters += 1
            continue
        
        no_progress = 0
        
        # Validate the cycle: all edges must exist in remaining
        cycle_edges = set()
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                cycle_edges.add(edge)
            else:
                valid = False
                break
        
        if not valid:
            iters += 1
            continue
        
        # Remove cycle edges from remaining
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        pieces.append(cycle[:])
        
        iters += 1
    
    # Remaining edges become single-edge pieces
    for u, v in remaining:
        pieces.append([u, v])
    
    return pieces

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Return what we have so far, fill rest with singletons
            while len(results) < 3:
                s = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))[len(results)]
                results.append([list(e) for e in _instance(*s)])
            break
        
        edges = _instance(*spec)
        n = spec[0]
        pieces = _decompose(edges, n)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
