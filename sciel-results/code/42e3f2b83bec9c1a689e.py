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


def _find_long_cycle(adj, n, beam_width=100):
    """
    Find a long cycle using beam search.
    Returns a list of vertices in cycle order (no repeat of start at end), or None.
    """
    best_cycle = None
    best_len = 0
    
    # Initialize beams: each is a tuple (path_list, visited_set)
    beams = []
    for start in range(n):
        if adj[start]:
            beams.append(([start], {start}))
            
    if not beams:
        return None
        
    # Limit depth to n to avoid infinite loops (though visited set prevents repeats)
    for _ in range(n):
        if not beams:
            break
            
        next_beams = []
        
        for path, visited in beams:
            last = path[-1]
            for v in adj[last]:
                if v == path[0] and len(path) >= 3:
                    # Found a cycle
                    cycle_len = len(path)
                    if cycle_len > best_len:
                        best_len = cycle_len
                        best_cycle = list(path)
                elif v not in visited:
                    new_path = path + [v]
                    new_visited = visited | {v}
                    next_beams.append((new_path, new_visited))
                    
        if not next_beams:
            break
            
        # Prune: keep top beam_width beams by length
        # Sorting O(B log B) where B is number of beams
        next_beams.sort(key=lambda x: len(x[0]), reverse=True)
        beams = next_beams[:beam_width]
        
    return best_cycle


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
        
        # Find a long cycle in the remaining graph using Beam Search
        cycle = _find_long_cycle(adj, n, beam_width=100)
        if cycle:
             # Verify validity (edges in remaining)
             valid = True
             for i in range(len(cycle)):
                 u = cycle[i]
                 v = cycle[(i + 1) % len(cycle)]
                 e = tuple(sorted((u, v)))
                 if e not in remaining:
                     valid = False
                     break
             if not valid:
                 cycle = None
        
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
