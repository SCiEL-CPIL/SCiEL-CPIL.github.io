"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai style)."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
import random

_DEADLINE = 140.0  # safety margin under 150s wall-clock kill

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

def _find_cycles_eulerian(adj, edges_set):
    """
    Find cycles using an Eulerian-like approach on the remaining graph.
    Repeatedly find a cycle by doing a DFS walk that tries to return to start.
    """
    remaining = set(edges_set)
    cycles = []
    
    if not remaining:
        return cycles
    
    # Build adjacency
    rem_adj = defaultdict(list)
    for u, v in remaining:
        rem_adj[u].append(v)
        rem_adj[v].append(u)
    
    # Sort neighbors for determinism
    for v in rem_adj:
        rem_adj[v].sort()
    
    while remaining:
        # Find a vertex with degree >= 2 (part of a cycle)
        start = None
        for v in sorted(rem_adj.keys()):
            if len(rem_adj[v]) >= 2:
                start = v
                break
        
        if start is None:
            break
        
        # Try to find a cycle starting from 'start'
        cycle = _find_cycle_from(start, rem_adj, remaining)
        
        if cycle is None:
            # No cycle found from this vertex; try another
            # Remove start from consideration by finding another vertex
            # Actually, if no cycle can be found, we're stuck
            break
        
        # Extract the cycle
        cycles.append(cycle)
        
        # Remove edges of the cycle from remaining
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = tuple(sorted((u, v)))
            remaining.discard(e)
            # Remove from adjacency
            if v in rem_adj[u]:
                rem_adj[u].remove(v)
            if u in rem_adj[v]:
                rem_adj[v].remove(u)
    
    return cycles

def _find_cycle_from(start, rem_adj, remaining):
    """
    Find a cycle starting and ending at 'start' using DFS.
    Returns the cycle as a list of vertices, or None.
    """
    visited = {start}
    path = [start]
    
    # DFS
    stack = [(start, iter(rem_adj[start]))]
    
    while stack:
        node, neighbors = stack[-1]
        advanced = False
        
        for nb in neighbors:
            edge = tuple(sorted((node, nb)))
            if edge not in remaining:
                continue
            
            if nb == start and len(path) >= 3:
                # Found a cycle
                return path[:]
            elif nb not in visited:
                visited.add(nb)
                path.append(nb)
                stack.append((nb, iter(rem_adj[nb])))
                advanced = True
                break
        
        if not advanced:
            # Backtrack
            if len(path) > 0:
                visited.discard(path[-1])
                path.pop()
            stack.pop()
    
    return None

def _decompose_one(edges, deadline):
    """
    Decompose a single graph into cycles and single edges.
    Strategy: 
    1. Repeatedly find cycles (preferring longer ones via randomized restarts).
    2. Remaining edges become single-edge pieces.
    """
    if not edges:
        return []
    
    edges_set = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(edges)
    
    pieces = []
    remaining = set(edges_set)
    
    # Phase 1: Extract cycles greedily
    # Use randomized restarts to find longer cycles
    rng = random.Random(12345)
    
    max_iterations = 500
    iteration = 0
    
    while remaining and iteration < max_iterations and time.time() < deadline:
        iteration += 1
        
        # Rebuild adjacency for remaining edges
        rem_adj = defaultdict(list)
        for u, v in remaining:
            rem_adj[u].append(v)
            rem_adj[v].append(u)
        
        # Find a cycle - try multiple strategies
        best_cycle = None
        best_len = 0
        
        # Strategy 1: Try from each vertex with degree >= 2
        candidate_starts = [v for v in rem_adj if len(rem_adj[v]) >= 2]
        
        if not candidate_starts:
            break
        
        # Randomize the order of starts
        rng.shuffle(candidate_starts)
        
        # Try at most 20 different starts
        tried = 0
        for start in candidate_starts:
            if tried >= 20 or time.time() > deadline:
                break
            tried += 1
            
            # Find a cycle from this start
            cycle = _find_cycle_from(start, rem_adj, remaining)
            
            if cycle and len(cycle) > best_len:
                best_len = len(cycle)
                best_cycle = cycle
        
        if best_cycle is None:
            break
        
        # Verify cycle edges are all in remaining
        valid = True
        cycle_edges = []
        for i in range(len(best_cycle)):
            u, v = best_cycle[i], best_cycle[(i + 1) % len(best_cycle)]
            e = tuple(sorted((u, v)))
            if e not in remaining:
                valid = False
                break
            cycle_edges.append(e)
        
        if not valid or len(best_cycle) < 3:
            break
        
        # Remove cycle edges from remaining
        for e in cycle_edges:
            remaining.discard(e)
        pieces.append(best_cycle)
    
    # Phase 2: Remaining edges as single-edge pieces
    for u, v in sorted(remaining):
        pieces.append([u, v])
    
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose_one(edges, _DEADLINE)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
