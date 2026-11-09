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

def _find_longest_cycle(adj, n, max_len=50):
    """Find a long cycle in the graph using randomized DFS walks."""
    best_cycle = None
    best_len = 0
    for _ in range(200):
        # Pick a random start vertex with degree >= 2
        candidates = [v for v in range(n) if len(adj[v]) >= 2]
        if not candidates:
            break
        start = candidates[_rng.randint(0, len(candidates) - 1)]
        # Randomized DFS to find a cycle
        visited = {start: -1}
        stack = [(start, iter(sorted(adj[start])))]
        path = [start]
        found = False
        while stack and not found:
            node, it = stack[-1]
            advanced = False
            for nxt in it:
                if nxt == start and len(path) >= 3:
                    found = True
                    break
                if nxt not in visited:
                    visited[nxt] = node
                    path.append(nxt)
                    stack.append((nxt, iter(sorted(adj[nxt]))))
                    advanced = True
                    break
            if not advanced and not found:
                stack.pop()
                path.pop()
        if found:
            cycle = list(path)
            if len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
    return best_cycle

def _decompose(edges, n):
    """Decompose edges into cycles and single edges, minimizing piece count."""
    global _rng
    _rng = Random(42)
    pieces = []
    remaining = set(tuple(sorted(e)) for e in edges)
    adj = {i: set() for i in range(n)}
    for u, v in remaining:
        adj[u].add(v)
        adj[v].add(u)
    
    # Iteratively find and extract longest cycles
    max_iterations = 1000
    iters = 0
    while remaining and iters < max_iterations:
        # Find longest cycle
        best = None
        for attempt in range(100):
            # Pick random start
            candidates = [v for v in range(n) if len(adj[v]) >= 2]
            if not candidates:
                break
            start = candidates[_rng.randint(0, len(candidates) - 1)]
            
            # Try to find a cycle through random walks
            visited = {start: -1}
            path = [start]
            stack = [(start, list(adj[start]))]
            found = False
            
            while stack and not found:
                node, neighbors = stack[-1]
                advanced = False
                # Shuffle neighbors for randomization
                _rng.shuffle(neighbors)
                for nxt in neighbors:
                    if nxt == start and len(path) >= 3:
                        found = True
                        break
                    if nxt not in visited:
                        visited[nxt] = node
                        path.append(nxt)
                        stack.append((nxt, list(adj[nxt])))
                        advanced = True
                        break
                if not advanced and not found:
                    stack.pop()
                    path.pop()
            
            if found:
                if best is None or len(path) > len(best):
                    best = list(path)
        
        if best is None or len(best) < 3:
            break
        
        # Extract this cycle
        cycle_edges = set()
        for i in range(len(best)):
            u = best[i]
            v = best[(i + 1) % len(best)]
            edge = tuple(sorted((u, v)))
            if edge in remaining:
                cycle_edges.add(edge)
            else:
                # Cycle not valid, skip
                best = None
                break
        
        if best is None:
            break
        
        # Remove cycle edges from remaining
        for edge in cycle_edges:
            remaining.discard(edge)
            u, v = edge
            adj[u].discard(v)
            adj[v].discard(u)
        
        pieces.append(best[:])
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
