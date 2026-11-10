"""Decompose three fixed Erdos-Renyi-ish graphs into few cycles + edges.

Strategy:
  * Minimizing pieces == maximizing number of edge-disjoint cycles.
  * Use 2-core decomposition to identify the cycle-supporting subgraph.
  * Within the 2-core, use randomized greedy with multiple restarts and
    low-degree vertex bias to extract many edge-disjoint cycles.
  * Leftover edges (including those outside the 2-core) become single-edge pieces.
"""

import time
import random
from collections import deque

_DEADLINE = 140.0
_SAFE = 3.0
_START = time.monotonic()


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _two_core(adj):
    """Return the 2-core: repeatedly remove degree-1 vertices."""
    n = len(adj)
    deg = [len(adj[v]) for v in range(n)]
    in_core = [deg[v] >= 2 for v in range(n)]
    queue = deque([v for v in range(n) if deg[v] < 2])
    while queue:
        v = queue.popleft()
        if not in_core[v]:
            continue
        in_core[v] = False
        for w in adj[v]:
            if in_core[w]:
                deg[w] -= 1
                if deg[w] < 2:
                    queue.append(w)
    return in_core


def _find_cycle_greedy(n, adj, in_core, rng, max_steps=50000):
    """Find one cycle using greedy DFS within the 2-core."""
    # Start from a random vertex in the 2-core with degree >= 2
    candidates = [v for v in range(n) if in_core[v] and len(adj[v]) >= 2]
    if not candidates:
        return None
    
    # Bias toward lower-degree vertices
    weights = [1.0 / (len(adj[v]) + 1) for v in candidates]
    s = rng.choices(candidates, weights=weights)[0]
    
    # Greedy path extension
    path = [s]
    visited = {s}
    current = s
    
    for _ in range(max_steps):
        # Choose next vertex: prefer unvisited, then check if we can close
        neighbors = list(adj[current])
        rng.shuffle(neighbors)
        
        found_next = False
        for w in neighbors:
            if w == s and len(path) >= 3:
                # Found a cycle!
                return path[:]
            if w not in visited:
                visited.add(w)
                path.append(w)
                current = w
                found_next = True
                break
        
        if not found_next:
            # Dead end, backtrack
            if len(path) <= 1:
                break
            visited.discard(current)
            path.pop()
            current = path[-1]
    
    return None


def _decompose(n, edges):
    adj = _build_adjacency(n, edges)
    in_core = _two_core(adj)
    
    cycles = []
    t0 = time.monotonic()
    rng = random.Random(42)
    
    # Extract cycles from the 2-core with multiple restarts
    max_restarts = 500
    restart = 0
    while time.monotonic() - t0 < 105.0 and restart < max_restarts:
        restart += 1
        
        # Check if any cycle exists in current graph
        has_cycle = False
        for u in range(n):
            if in_core[u] and adj[u]:
                for v in adj[u]:
                    if in_core[v] and u < v and (adj[u] & adj[v]):
                        has_cycle = True
                        break
            if has_cycle:
                break
        if not has_cycle:
            break
        
        # Try to find a cycle
        cyc = _find_cycle_greedy(n, adj, in_core, rng, max_steps=20000)
        if not cyc or len(cyc) < 3:
            continue
        
        # Remove cycle edges from adj
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            adj[a].discard(b)
            adj[b].discard(a)
        cycles.append(cyc)
        
        # Update 2-core
        in_core = _two_core(adj)
        
        if time.monotonic() - _START > _DEADLINE - _SAFE:
            break
    
    # Collect all remaining edges as singles
    leftover = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                leftover.append((u, v))
    
    pieces = [list(c) for c in cycles]
    pieces.extend([list(e) for e in leftover])
    return pieces


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    result = []
    for spec in specs:
        n, p, seed = spec
        edges = _instance(n, p, seed)
        if time.monotonic() - _START > _DEADLINE - _SAFE:
            result.append([list(e) for e in edges])
            continue
        pieces = _decompose(n, edges)
        result.append(pieces)
    return result
