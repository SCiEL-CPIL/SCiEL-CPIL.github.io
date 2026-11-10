"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, forbidden_edges):
    """Find a Hamiltonian cycle using DFS with pruning. Returns list of vertices or None."""
    n = P['n']
    start = 0
    
    # forbidden_edges is a set of frozensets
    def dfs(path, visited, used_edges):
        if len(path) == n:
            # Check if last vertex connects to start
            last = path[-1]
            if start in adj[last]:
                edge = frozenset((last, start))
                if edge not in used_edges and edge not in forbidden_edges:
                    return list(path)
            return None
        
        current = path[-1]
        for neighbor in sorted(adj[current]):
            if neighbor not in visited:
                edge = frozenset((current, neighbor))
                if edge not in used_edges and edge not in forbidden_edges:
                    visited.add(neighbor)
                    used_edges.add(edge)
                    result = dfs(path + [neighbor], visited, used_edges)
                    if result is not None:
                        return result
                    used_edges.remove(edge)
                    visited.remove(neighbor)
        return None
    
    visited = {start}
    used_edges = set()
    return dfs([start], visited, used_edges)

def _find_two_hamiltonian_cycles():
    """Find two edge-disjoint Hamiltonian cycles."""
    adj = _build_adjacency()
    
    # Find first Hamiltonian cycle
    cycle1 = _find_hamiltonian_cycle(adj, set())
    if cycle1 is None:
        return None
    
    # Collect edges used in cycle1
    edges1 = set()
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        edges1.add(frozenset((u, v)))
    
    # Find second Hamiltonian cycle avoiding edges1
    cycle2 = _find_hamiltonian_cycle(adj, edges1)
    if cycle2 is None:
        return None
    
    return [cycle1, cycle2]

def construct():
    result = _find_two_hamiltonian_cycles()
    if result is None:
        # Fallback: return a single Hamiltonian cycle if we can't find two
        adj = _build_adjacency()
        cycle = _find_hamiltonian_cycle(adj, set())
        if cycle is not None:
            return [cycle]
        return []
    
    # Convert to proper format: list of lists of ints
    cycles = []
    for cycle in result:
        cycles.append([int(v) for v in cycle])
    
    return cycles
# EVOLVE-BLOCK-END
