"""Decompose three fixed graphs into cycles + edges, minimizing piece count."""
import time
import random
import sys
from collections import defaultdict, deque

_DEADLINE = 120.0  # safety margin before 150s kill
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return _DEADLINE - _elapsed()

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle_fast(adj, start):
    """
    Finds a cycle in the graph starting from 'start' using DFS.
    Returns a list of vertices in the cycle (e.g., [v1, v2, ..., vk]) where (vk, v1) is an edge.
    Returns None if no cycle is found.
    """
    parent = {start: None}
    visited = {start}
    stack = [start]
    
    # We need to track the path to reconstruct the cycle
    # Standard DFS for cycle detection in undirected graph:
    # If we find an edge (u, v) where v is visited and v is not parent[u], we have a cycle.
    
    def dfs(node):
        for nb in adj.get(node, []):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = node
                cycle = dfs(nb)
                if cycle:
                    return cycle
            elif nb != parent.get(node, -1):
                # Found a back edge to an ancestor
                # Reconstruct path from node to nb
                cycle = [node]
                curr = node
                while curr != nb:
                    curr = parent[curr]
                    if curr is None:
                        return None # Should not happen if nb is ancestor
                    cycle.append(curr)
                # The cycle is nb -> ... -> node -> nb
                # The list `cycle` is [node, ..., nb]
                # We want the sequence of vertices.
                # The cycle edges are (cycle[0], cycle[1]), ..., (cycle[-1], cycle[0])?
                # No, the path is nb -> ... -> node. The edge is (node, nb).
                # So the cycle is nb, ..., node.
                # Let's return the list of vertices in order.
                # cycle list currently: [node, parent[node], ..., nb]
                # Reverse it to get nb, ..., node
                cycle.reverse()
                return cycle
        return None
    
    return dfs(start)

def _greedy_decompose(edges, strategy=0, rng=None):
    """
    Greedy cycle extraction using fast DFS.
    """
    if rng is None:
        rng = random.Random(0)
    remaining = set(edges)
    if not remaining:
        return []
    adj = _build_adj(remaining)
    pieces = []
    
    while remaining:
        if _remaining() < 1.0:
            break
        
        # Find a cycle in the current graph
        cycle = None
        # Try to find a cycle starting from high-degree vertices first to potentially get larger cycles
        # But since _find_cycle_fast is linear, we can just try a few starts.
        # To improve quality, we can try to find a cycle, and if it's small, try another start.
        # For now, just find *a* cycle.
        
        # Identify vertices in the 2-core to avoid wasting time on tree parts
        # Simple heuristic: try vertices with degree >= 2
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            # No cycles possible
            break
            
        # Shuffle candidates to vary results if rng is used, or sort by degree
        # Sorting by degree descending often finds cycles in dense parts
        candidates.sort(key=lambda v: -len(adj[v]))
        
        found_cycle = False
        for start in candidates[:10]: # Try top 10 highest degree vertices
            c = _find_cycle_fast(adj, start)
            if c and len(c) >= 3:
                cycle = c
                found_cycle = True
                break
        
        if not found_cycle or cycle is None or len(cycle) < 3:
            # No cycle found in the core? Or graph is a tree?
            # Take an edge
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
            continue
        
        # Verify the cycle edges are in remaining
        valid = True
        cycle_edges = set()
        for i in range(len(cycle)):
            a, b = cycle[i], cycle[(i + 1) % len(cycle)]
            ce = (min(a, b), max(a, b))
            if ce in remaining:
                cycle_edges.add(ce)
            else:
                valid = False
                break
        
        if not valid or len(cycle_edges) < 3:
             # Fallback
             e = next(iter(remaining))
             pieces.append([e[0], e[1]])
             remaining.discard(e)
             u, v = e
             adj[u].discard(v)
             adj[v].discard(u)
             if not adj[u]:
                 del adj[u]
             if not adj[v]:
                 del adj[v]
             continue
        
        pieces.append(list(cycle))
        for ce in cycle_edges:
            remaining.discard(ce)
            u, v = ce
            adj[u].discard(v)
            adj[v].discard(u)
            if not adj[u]:
                del adj[u]
            if not adj[v]:
                del adj[v]
    
    # Remaining edges as individual pieces
    for e in sorted(remaining):
        pieces.append([e[0], e[1]])
    
    return pieces

def _validate(pieces, edges):
    """Check that pieces are edge-disjoint and cover all edges."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    used = set()
    for p in pieces:
        if len(p) == 2:
            e = tuple(sorted(p))
            if e in used:
                return False
            used.add(e)
        elif len(p) >= 3:
            # Check all vertices distinct
            if len(set(p)) != len(p):
                return False
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                e = tuple(sorted((a, b)))
                if e in used:
                    return False
                used.add(e)
        else:
            return False
    return used == edge_set

def _local_improve(pieces, edges, time_budget=2.0):
    """Try to merge two edge-pieces into a cycle if possible, or extend cycles."""
    # This is a simple improvement: look for two edges that form a path of length 2
    # where the middle vertex connects to both endpoints, forming a triangle
    # For now, just return as-is; improvement is complex
    return pieces

def construct():
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    rng = random.Random(42)
    
    for idx, spec in enumerate(specs):
        if _remaining() < 5.0:
            # Not enough time; just do edges
            edges = _instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = _instance(*spec)
        best_pieces = None
        best_count = len(edges)  # baseline: all edges
        
        # Try multiple strategies and random seeds
        for strategy in [0, 1, 2]:
            for attempt in range(3):
                if _remaining() < 3.0:
                    break
                r = random.Random(idx * 100 + strategy * 10 + attempt)
                pieces = _greedy_decompose(edges, strategy=strategy, rng=r)
                if not _validate(pieces, edges):
                    continue
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
            if _remaining() < 3.0:
                break
        
        if best_pieces is None:
            best_pieces = [[u, v] for u, v in edges]
        
        results.append(best_pieces)
    
    return results
