"""Rewritten program: decompose into cycles + edges, minimize piece count."""
import time
import random as _random
from collections import deque

_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges, n):
    adj = {i: set() for i in range(n)}
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_longest_cycle(adj, edges_set, n, rng, max_iter=200000):
    """Find a long cycle using randomized DFS with backtracking limit."""
    best_cycle = None
    for _ in range(50):  # multiple attempts
        if _time_left() < 5:
            break
        start = rng.choice(list(range(n)))
        # Randomized DFS to find a cycle through start
        visited = set()
        path = [start]
        visited.add(start)
        
        def dfs(v, depth, max_depth=50):
            if depth > max_depth:
                return None
            if _time_left() < 5:
                return None
            # Check if we can close a cycle
            if len(path) >= 3 and start in adj[v] and (min(v, start), max(v, start)) in edges_set:
                return path[:]
            # Try neighbors
            neighbors = list(adj[v])
            rng.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited and nb != start:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(nb, depth + 1, max_depth)
                    if result:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None
        
        result = dfs(start, 0)
        if result and (not best_cycle or len(result) > len(best_cycle)):
            best_cycle = result
            if len(best_cycle) >= 15:
                break
    return best_cycle

def _find_cycle_greedy(adj, edges_set, n, rng):
    """Find a cycle using greedy approach: start from a vertex, walk randomly, try to close."""
    best_cycle = None
    for _ in range(30):
        if _time_left() < 5:
            break
        start = rng.choice(list(range(n)))
        if not adj[start]:
            continue
        path = [start]
        used_edges = set()
        current = start
        for _ in range(n + 5):
            neighbors = [nb for nb in adj[current] if (min(current, nb), max(current, nb)) not in used_edges]
            if not neighbors:
                break
            # Prefer closing the cycle if possible
            if len(path) >= 3 and start in neighbors:
                cycle = path[:]
                # Verify all edges exist
                valid = True
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    if (min(a, b), max(a, b)) not in edges_set:
                        valid = False
                        break
                if valid:
                    if not best_cycle or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                    break
            # Pick a random neighbor
            nb = rng.choice(neighbors)
            used_edges.add((min(current, nb), max(current, nb)))
            path.append(nb)
            current = nb
            if nb == start and len(path) >= 4:
                cycle = path[:-1]
                valid = True
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    if (min(a, b), max(a, b)) not in edges_set:
                        valid = False
                        break
                if valid:
                    if not best_cycle or len(cycle) > len(best_cycle):
                        best_cycle = cycle
                break
    return best_cycle

def _decompose_one(edges, n, rng, time_budget=25):
    """Decompose one graph into cycles + edges, minimize piece count."""
    start_time = time.time()
    best_pieces = None
    best_count = float('inf')
    
    for restart in range(50):
        if _time_left() < 10:
            break
        if time.time() - start_time > time_budget:
            break
        
        remaining = set(edges)
        adj = _build_adj(remaining, n)
        pieces = []
        
        while remaining:
            if _time_left() < 10:
                break
            # Find a cycle
            cycle = _find_longest_cycle(adj, remaining, n, rng)
            if cycle is None:
                cycle = _find_cycle_greedy(adj, remaining, n, rng)
            
            if cycle and len(cycle) >= 3:
                # Remove cycle edges
                for i in range(len(cycle)):
                    a, b = cycle[i], cycle[(i + 1) % len(cycle)]
                    e = (min(a, b), max(a, b))
                    if e in remaining:
                        remaining.discard(e)
                        adj[a].discard(b)
                        adj[b].discard(a)
                pieces.append(cycle[:])
            else:
                # No cycle found, take one edge
                e = next(iter(remaining))
                remaining.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append([u, v])
        
        # Count pieces
        count = len(pieces)
        if count < best_count:
            best_count = count
            best_pieces = [p[:] for p in pieces]
    
    if best_pieces is None:
        # Fallback: all edges
        best_pieces = [[u, v] for (u, v) in edges]
    
    return best_pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, spec in enumerate(specs):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        rng = _random.Random(seed * 1000 + i + 42)
        pieces = _decompose_one(edges, n, rng, time_budget=40)
        results.append(pieces)
    return results
