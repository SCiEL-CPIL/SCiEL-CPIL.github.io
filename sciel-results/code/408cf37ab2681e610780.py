"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import random
import sys

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(adj, n, forbidden_edges, start, deadline, rng):
    """Find a Hamiltonian cycle starting at `start` avoiding forbidden edges."""
    path = [start]
    used_edges = set()
    visited = [False] * n
    visited[start] = True
    
    # Precompute adjacency lists as sorted lists for faster iteration
    adj_lists = [sorted(adj[i]) for i in range(n)]
    
    def dfs(path, visited, used_edges):
        if time.time() > deadline:
            return None
        if len(path) == n:
            last = path[-1]
            first = path[0]
            edge = (min(last, first), max(last, first))
            if first in adj[last] and edge not in forbidden_edges and edge not in used_edges:
                return list(path)
            return None
        
        curr = path[-1]
        # Gather unvisited neighbors not using forbidden/used edges
        candidates = []
        for v in adj_lists[curr]:
            if not visited[v]:
                edge = (min(curr, v), max(curr, v))
                if edge not in forbidden_edges and edge not in used_edges:
                    candidates.append(v)
        
        if not candidates:
            return None
        
        # Warnsdorff's heuristic: sort by number of available neighbors in remaining graph
        def warnsdorff_score(v):
            count = 0
            for u in adj_lists[v]:
                if not visited[u] and u != curr:
                    edge = (min(v, u), max(v, u))
                    if edge not in forbidden_edges and edge not in used_edges:
                        count += 1
            return count
        
        candidates.sort(key=warnsdorff_score)
        
        for v in candidates:
            edge = (min(curr, v), max(curr, v))
            if edge in forbidden_edges or edge in used_edges:
                continue
            path.append(v)
            visited[v] = True
            used_edges.add(edge)
            result = dfs(path, visited, used_edges)
            if result is not None:
                return result
            # Backtrack
            path.pop()
            visited[v] = False
            used_edges.discard(edge)
        return None
    
    return dfs(path, visited, used_edges)

def construct():
    t0 = time.time()
    deadline = t0 + 80  # Safety margin from 90s limit
    n = P['n']
    adj = _build_adj()
    
    best_cycles = []
    max_cycles = 3  # Theoretical max: floor(75/21) = 3
    
    # Try multiple starting vertices with different random seeds
    for attempt in range(20):
        if time.time() > deadline - 2:
            break
        
        if len(best_cycles) >= max_cycles:
            break
        
        seed = 42 + attempt * 137
        rng = random.Random(seed)
        start = rng.randint(0, n - 1)
        
        # Build forbidden edge set from already-found cycles
        forbidden = set()
        for cyc in best_cycles:
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                forbidden.add((min(u, v), max(u, v)))
        
        cycle = _find_ham_cycle(adj, n, forbidden, start, deadline, rng)
        if cycle is not None:
            # Validate the cycle
            valid = True
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if v not in adj[u]:
                    valid = False
                    break
            if valid and len(set(cycle)) == n:
                best_cycles.append(cycle)
    
    # Fallback if no cycles found
    if not best_cycles:
        for start in range(n):
            cycle = _find_ham_cycle(adj, n, set(), start, deadline, random.Random(0))
            if cycle is not None:
                return [cycle]
        return [list(range(n))]
    
    return best_cycles
# EVOLVE-BLOCK-END
