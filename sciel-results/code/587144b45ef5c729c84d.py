"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj

def _find_ham_cycle(adj, used_edges, start=0, deadline=None):
    """Find a Hamiltonian cycle using DFS with Warnsdorff heuristic."""
    n = P['n']
    vertices = list(range(n))
    
    # Check if start has enough unused degree
    unused_neighbors = [v for v in adj.get(start, set()) if (min(start,v), max(start,v)) not in used_edges]
    if len(unused_neighbors) < 2:
        return None
    
    def dfs(path, used_local, current):
        if deadline and time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if (min(current, start), max(current, start)) not in used_local and (current, start) != (start, current):
                if start in adj.get(current, set()):
                    return path[:]
            return None
        
        # Warnsdorff: sort neighbors by number of unused forward options
        candidates = []
        for nb in adj.get(current, set()):
            if nb not in path and (min(current, nb), max(current, nb)) not in used_local:
                # Count unused neighbors of nb (excluding current and already visited)
                forward = 0
                for nn in adj.get(nb, set()):
                    if nn not in path and nn != current and (min(nb, nn), max(nb, nn)) not in used_local:
                        forward += 1
                candidates.append((forward, nb))
        
        # Sort by fewest forward options first (Warnsdorff)
        candidates.sort(key=lambda x: x[0])
        
        for _, nb in candidates:
            new_used = used_local | {(min(current, nb), max(current, nb))}
            result = dfs(path + [nb], new_used, nb)
            if result is not None:
                return result
        
        return None
    
    return dfs([start], set(), start)

def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    adj = _build_adj(P['edges'])
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try multiple starting points and random orderings
        starts = list(range(P['n']))
        random.shuffle(starts)
        
        for start in starts:
            if time.time() > deadline - 5:
                break
            
            # Try finding a Hamiltonian cycle from this start
            cycle = _find_ham_cycle(adj, used_edges, start=start, deadline=deadline - 5)
            
            if cycle is not None:
                # Verify it's valid
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = (min(u, v), max(u, v))
                    if edge not in adj.get(u, set()) and edge not in adj.get(v, set()):
                        valid = False
                        break
                    if edge in used_edges:
                        valid = False
                        break
                    if len(set(cycle)) != len(cycle):
                        valid = False
                        break
                
                if valid and len(cycle) == P['n']:
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        used_edges.add((min(u, v), max(u, v)))
                    cycles.append(cycle)
                    found = True
                    break
        
        if not found:
            break
    
    # Ensure at least one cycle (fallback)
    if not cycles:
        cycles = [list(range(P['n']))]
    
    return cycles
# EVOLVE-BLOCK-END
