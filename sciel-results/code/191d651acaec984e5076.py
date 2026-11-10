"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    n = P['n']
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, start=0, deadline=None):
    """Find a Hamiltonian cycle using DFS with backtracking."""
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    def dfs(current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            # Check if last vertex connects back to start
            if start in adj[path[-1]]:
                return list(path)
            return None
        for nxt in sorted(adj[current]):
            if not visited[nxt]:
                visited[nxt] = True
                path.append(nxt)
                result = dfs(nxt)
                if result is not None:
                    return result
                path.pop()
                visited[nxt] = False
        return None
    
    return dfs(start)

def _find_hamiltonian_cycle_with_used_edges(adj, n, used_edges, start=0, deadline=None):
    """Find a Hamiltonian cycle avoiding used edges."""
    visited = [False] * n
    path = [start]
    visited[start] = True
    
    def dfs(current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[path[-1]] and (min(start, path[-1]), max(start, path[-1])) not in used_edges:
                return list(path)
            return None
        for nxt in sorted(adj[current]):
            if not visited[nxt]:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in used_edges:
                    visited[nxt] = True
                    path.append(nxt)
                    result = dfs(nxt)
                    if result is not None:
                        return result
                    path.pop()
                    visited[nxt] = False
        return None
    
    return dfs(start)

def _extract_cycles_from_coloring(coloring, adj, n):
    """Given an edge coloring (dict edge->color), extract cycles for each color."""
    cycles = []
    for color in sorted(set(coloring.values())):
        # Build subgraph with edges of this color
        color_edges = [e for e, c in coloring.items() if c == color]
        color_adj = defaultdict(list)
        for u, v in color_edges:
            color_adj[u].append(v)
            color_adj[v].append(u)
        
        # Each vertex should have degree 2 in this subgraph
        # Find the cycle by following edges
        visited_edges = set()
        cycle = []
        # Start from vertex 0 (or any vertex with edges)
        start = 0
        current = start
        prev = None
        while True:
            cycle.append(current)
            # Find next unvisited edge from current
            next_vertex = None
            for nxt in color_adj[current]:
                edge = (min(current, nxt), max(current, nxt))
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    next_vertex = nxt
                    break
            if next_vertex is None:
                break
            prev, current = current, next_vertex
            if current == start and len(cycle) > 1:
                break
        
        if len(cycle) == n:
            cycles.append(cycle)
    
    return cycles

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    n = P['n']
    adj = _build_adjacency()
    
    # Try to find multiple edge-disjoint Hamiltonian cycles
    # The graph is 5-regular, so we can potentially have at most 2 edge-disjoint Hamiltonian cycles
    # (since each uses n=21 edges, and total edges = 5*21/2 = 52.5 -> 52 edges, so max 2 full cycles)
    
    best_cycles = []
    
    # Strategy: Try to find 2 edge-disjoint Hamiltonian cycles
    # First, find one Hamiltonian cycle
    cycle1 = _find_hamiltonian_cycle(adj, n, start=0, deadline=deadline)
    if cycle1 is None:
        # Fallback: return a simple cycle
        return [list(range(n))]
    
    used_edges = set()
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i + 1) % len(cycle1)]
        used_edges.add((min(u, v), max(u, v)))
    
    best_cycles.append(cycle1)
    
    # Try to find a second Hamiltonian cycle avoiding used edges
    if time.time() < deadline - 5:
        cycle2 = _find_hamiltonian_cycle_with_used_edges(adj, n, used_edges, start=0, deadline=deadline)
        if cycle2 is not None:
            best_cycles.append(cycle2)
    
    # If we couldn't find 2 Hamiltonian cycles, try other strategies
    if len(best_cycles) < 2:
        # Try to find any cycle (not necessarily Hamiltonian) that's edge-disjoint
        # For now, just return what we have
        pass
    
    # Ensure all cycles are lists of Python ints
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    if not result:
        result = [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
