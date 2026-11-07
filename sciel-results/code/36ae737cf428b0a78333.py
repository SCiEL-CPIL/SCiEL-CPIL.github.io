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
    visited = {start}
    
    def dfs(path, visited, used_edges):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if closing edge exists and is not forbidden
            last = path[-1]
            first = path[0]
            edge = (min(last, first), max(last, first))
            if first in adj[last] and edge not in forbidden_edges and edge not in used_edges:
                return list(path)
            return None
        
        curr = path[-1]
        neighbors = [v for v in adj[curr] if v not in visited]
        
        # Warnsdorff's heuristic: sort by degree of available neighbors
        def warnsdorff_score(v):
            count = 0
            for u in adj[v]:
                if u not in visited and u != curr:
                    edge = (min(v, u), max(v, u))
                    if edge not in forbidden_edges and edge not in used_edges:
                        count += 1
            return count
        
        neighbors.sort(key=warnsdorff_score)
        
        for v in neighbors:
            edge = (min(curr, v), max(curr, v))
            if edge in forbidden_edges or edge in used_edges:
                continue
            new_path = path + [v]
            new_visited = visited | {v}
            new_used = used_edges | {edge}
            result = dfs(new_path, new_visited, new_used)
            if result is not None:
                return result
        return None
    
    return dfs(path, visited, used_edges)

def _validate_cycles(cycles, n, adj):
    """Validate that all cycles are valid and edge-disjoint."""
    if not cycles:
        return False
    all_edges = set()
    for cyc in cycles:
        if len(cyc) != n:
            return False
        if len(set(cyc)) != n:
            return False
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            edge = (min(u, v), max(u, v))
            if edge in all_edges:
                return False
            if v not in adj[u]:
                return False
            all_edges.add(edge)
    return True

def construct():
    deadline = time.time() + 85  # Safety margin from 90s limit
    n = P['n']
    adj = _build_adj()
    rng = random.Random(42)
    
    best_cycles = []
    
    # Try multiple random restarts with different starting vertices
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline - 5:
            break
        
        forbidden = set()
        for cyc in best_cycles:
            for i in range(len(cyc)):
                u = cyc[i]
                v = cyc[(i + 1) % len(cyc)]
                forbidden.add((min(u, v), max(u, v)))
        
        cycle = _find_ham_cycle(adj, n, forbidden, start, deadline, rng)
        if cycle is not None:
            best_cycles.append(cycle)
            if len(best_cycles) >= 10:
                break
    
    # Return best result found
    if not best_cycles:
        # Fallback: try to return at least one valid cycle
        for start in range(n):
            cycle = _find_ham_cycle(adj, n, set(), start, deadline, rng)
            if cycle is not None:
                return [cycle]
        return [list(range(n))]
    
    return best_cycles
# EVOLVE-BLOCK-END
