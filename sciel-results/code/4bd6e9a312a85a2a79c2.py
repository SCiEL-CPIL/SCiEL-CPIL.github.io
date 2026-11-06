"""Rewritten program: greedy long-cycle extraction via randomized DFS + backtracking."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _decompose(edges, deadline, rng):
    """Decompose edge set into cycles and single edges, minimizing piece count."""
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    remaining = set(edges)
    pieces = []

    # Precompute a heuristic order: prefer vertices with higher degree
    def degree(v):
        return len(adj[v]) & 1  # parity trick not needed; just use len
    # Actually just use len(adj[v]) but we need dynamic degree. Use a set of remaining edges.

    # We'll maintain remaining edges as a set of frozensets or tuples
    rem = set(edges)
    rem_adj = defaultdict(set)
    for u, v in rem:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    def try_extract_cycle(start, max_len, rng):
        """Try to find a cycle starting and ending at start using DFS with backtracking."""
        # Use iterative DFS with a stack
        # Path: list of vertices
        # We want to find a simple cycle of length >= 3
        
        # Heuristic: sort neighbors by degree (prefer higher degree to keep options open)
        def neighbor_order(v):
            return sorted(rem_adj[v], key=lambda x: -len(rem_adj[x]))
        
        # DFS with backtracking
        path = [start]
        visited = {start}
        
        def dfs(current, depth):
            if time.time() > deadline:
                return None
            if depth >= 3:
                if start in rem_adj[current]:
                    return path + [start]
            for nb in neighbor_order(current):
                if nb == start and depth >= 3:
                    return path + [start]
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    result = dfs(nb, depth + 1)
                    if result:
                        return result
                    path.pop()
                    visited.discard(nb)
            return None
        
        return dfs(start, 1)

    # Greedy: repeatedly extract the longest cycle we can find
    # Strategy: try from each vertex with highest degree, do randomized restarts
    max_iterations = 10000
    iteration = 0
    
    while rem and time.time() < deadline:
        iteration += 1
        if iteration > max_iterations:
            break
            
        # Pick a starting vertex: prefer high degree
        candidates = list(rem_adj.keys())
        if not candidates:
            break
        # Sort by degree descending, add some randomness
        candidates.sort(key=lambda v: -len(rem_adj[v]))
        
        best_cycle = None
        best_len = 0
        
        # Try a few random starting vertices (weighted toward high degree)
        num_tries = min(20, len(candidates))
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            # Weighted random selection
            r = rng.random() ** 2  # bias toward top
            idx = int(r * len(candidates))
            if idx >= len(candidates):
                idx = len(candidates) - 1
            start = candidates[idx]
            
            if not rem_adj[start]:
                continue
            
            cycle = try_extract_cycle(start, 100, rng)
            if cycle and len(cycle) - 1 > best_len:
                best_len = len(cycle) - 1
                best_cycle = cycle
        
        # If no cycle found, take a single edge
        if best_cycle and best_len >= 3:
            # Remove edges from cycle
            for i in range(len(best_cycle) - 1):
                u, v = best_cycle[i], best_cycle[i + 1]
                e = (u, v) if u < v else (v, u)
                rem.discard(e)
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                if not rem_adj[u]:
                    del rem_adj[u]
                if not rem_adj[v]:
                    del rem_adj[v]
            # Store cycle as list of vertices (closing edge implied)
            pieces.append(best_cycle[:-1])  # remove the repeated start at end
        else:
            # Take a single edge
            if not rem:
                break
            u, v = next(iter(rem))
            rem.discard((u, v))
            rem_adj[u].discard(v)
            rem_adj[v].discard(u)
            if not rem_adj[u]:
                del rem_adj[u]
            if not rem_adj[v]:
                del rem_adj[v]
            pieces.append([u, v])
    
    # If time ran out, add all remaining edges as single-edge pieces
    for u, v in rem:
        pieces.append([u, v])
    
    return pieces


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin: 150s limit, leave 10s buffer
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        rng = random.Random(42)
        pieces = _decompose(edges, deadline, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
