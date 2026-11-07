"""Optimal packing of edge-disjoint Hamiltonian cycles on C_21(1,2,5)."""
import time
import math

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def _cycle_edges(cycle):
    """Get the set of edges in a cycle."""
    edges = set()
    n = len(cycle)
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges


def _verify_cycle(cycle, n, edge_set):
    """Verify cycle is valid: all distinct vertices, all edges in host graph."""
    if len(cycle) != n:
        return False
    if len(set(cycle)) != n:
        return False
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        e = (min(u, v), max(u, v))
        if e not in edge_set:
            return False
    return True


def construct():
    """Return the maximum set of edge-disjoint Hamiltonian cycles.
    
    The host graph is the circulant C_21(1,2,5) with 63 edges.
    Each Hamiltonian cycle on 21 vertices uses exactly 21 edges.
    63 / 21 = 3, so at most 3 edge-disjoint Hamiltonian cycles exist.
    Steps 1, 2, 5 (all coprime to 21) yield 3 valid edge-disjoint cycles,
    achieving the theoretical maximum.
    """
    n = P['n']
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
    
    cycles = []
    used_edges = set()
    
    # Pure step cycles: step s gives cycle [0, s, 2s, ..., 20s] mod 21
    # Valid steps must have distance in {1, 2, 5} mod 21
    # Steps coprime to 21 with |s mod 21| in {1,2,5}: {1, 2, 5, 16, 19, 20}
    # But steps s and 21-s use the same edge set (just reversed)
    # So distinct edge sets come from steps {1, 2, 5}
    for step in [1, 2, 5]:
        cycle = [(i * step) % n for i in range(n)]
        if not _verify_cycle(cycle, n, host_edges):
            continue
        ce = _cycle_edges(cycle)
        if ce & used_edges:
            continue
        used_edges |= ce
        cycles.append(cycle)
    
    # Verify we have 3 cycles (theoretical maximum)
    # 63 total edges, 21 per cycle, max 3 cycles
    # No further cycles are possible
    
    # Ensure proper types
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
    
    if not result:
        result = [[int(i) for i in range(n)]]
    
    return result
