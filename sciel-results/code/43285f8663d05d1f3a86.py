"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

# Ensure we have enough recursion depth for DFS
sys.setrecursionlimit(10000)


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle(adj, n, max_iter=5000):
    """
    Find a long cycle using randomized DFS with backtracking.
    Returns a list of vertices in cycle order (no repeat of start at end), or None.
    """
    # Try multiple random starts
    for _ in range(50):
        start = random.randint(0, n - 1)
        if not adj[start]:
            continue
        # DFS with path tracking
        path = [start]
        visited = {start}
        parent = {start: -1}
        found = False
        
        def dfs(u):
            nonlocal found
            if found:
                return
            for v in sorted(adj[u], key=lambda x: random.random()):  # randomize order
                if v == start and len(path) >= 3:
                    # Found a cycle
                    found = True
                    return
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    path.append(v)
                    dfs(v)
                    if found:
                        return
                    path.pop()
                    visited.discard(v)
                    del parent[v]
        
        dfs(start)
        if found:
            return path
    return None


def _decompose_graph(edges, n, deadline):
    """
    Decompose the graph into cycles and single edges.
    Returns a list of pieces.
    """
    if not edges:
        return []
    
    # Build adjacency
    adj = _build_adj(edges, n)
    
    # Track remaining edges
    remaining = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Strategy: repeatedly find long cycles, then handle leftovers
    # We'll use a greedy approach with some randomness
    
    # First, try to find as many long cycles as possible
    max_cycles = 100  # Safety limit
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        # Find a long cycle in the remaining graph
        cycle = None
        for attempt in range(20):
            candidate = _find_long_cycle(adj, n)
            if candidate and len(candidate) >= 3:
                # Check if all edges in this cycle are in remaining
                valid = True
                for i in range(len(candidate)):
                    u = candidate[i]
                    v = candidate[(i + 1) % len(candidate)]
                    e = tuple(sorted((u, v)))
                    if e not in remaining:
                        valid = False
                        break
                if valid:
                    cycle = candidate
                    break
        
        if cycle:
            # Add the cycle as a piece
            pieces.append(list(cycle))
            # Remove the cycle's edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                # Update adjacency
                adj[u].discard(v)
                adj[v].discard(u)
        else:
            break
    
    # Handle remaining edges as single-edge pieces
    for e in sorted(remaining):
        if time.time() > deadline:
            break
        pieces.append(list(e))
    
    return pieces


def construct():
    """
    Decompose three fixed graphs into few cycles and edges.
    Returns three lists of pieces, one per fixed instance graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = _instance(n, p, seed)
        
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = _decompose_graph(edges, n, deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
