"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import time
import random
from collections import defaultdict

def _build_adjacency():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle_fast(adj, used_edges, start, deadline):
    """Find a Hamiltonian cycle starting/ending at `start` avoiding used_edges."""
    n = P['n']
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    # Precompute available neighbors for each vertex (excluding used edges)
    avail = {}
    for v in range(n):
        avail[v] = [u for u in adj[v] if (v, u) not in used_edges and (u, v) not in used_edges]
    
    # Sort neighbors by remaining degree for better pruning (Warnsdorff's rule heuristic)
    for v in range(n):
        avail[v].sort(key=lambda x: len(avail[x]))
        
    # Quick check: if start doesn't have enough neighbors to close, fail early
    if len(avail[start]) < 2:
        return None

    def backtrack():
        if time.time() > deadline:
            return False
        if len(path) == n:
            last = path[-1]
            # Check if we can close the cycle back to start
            if (last, start) not in used_edges and (start, last) not in used_edges:
                if start in adj[last]:
                    return True
            return False
        
        current = path[-1]
        for nb in avail[current]:
            # Cannot go back to start unless it's the last step
            if nb == start and len(path) < n:
                continue
            if visited[nb]:
                continue
            
            path.append(nb)
            visited[nb] = True
            if backtrack():
                return True
            path.pop()
            visited[nb] = False
        
        return False
    
    if backtrack():
        return path
    return None

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    adj = _build_adjacency()
    n = P['n']
    
    best_cycles = []
    
    # Try multiple strategies to find the 3 disjoint cycles
    # Strategy 1: Natural order 0..20
    # Strategy 2: Reversed order 20..0
    # Strategy 3: Random shuffles
    
    strategies = []
    strategies.append(list(range(n)))
    strategies.append(list(range(n-1, -1, -1)))
    
    for seed in range(10):
        random.seed(seed)
        shuffled = list(range(n))
        random.shuffle(shuffled)
        strategies.append(shuffled)

    for start_order in strategies:
        if time.time() > deadline:
            break
        if len(best_cycles) >= 3: # Max possible is 3
            break
            
        cycles = []
        used_edges = set()
        
        for start in start_order:
            if time.time() > deadline:
                break
            if len(cycles) >= 3:
                break
                
            cycle = _find_ham_cycle_fast(adj, used_edges, start, deadline)
            if cycle is not None:
                # Verify and add edges to used set
                valid = True
                cycle_edges = set()
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if (u, v) in used_edges or (v, u) in used_edges:
                        valid = False
                        break
                    cycle_edges.add((u, v))
                    cycle_edges.add((v, u))
                
                if valid:
                    cycles.append(cycle)
                    used_edges.update(cycle_edges)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 3:
                break

    # Ensure types are correct
    result = [list(c) for c in best_cycles]
    for c in result:
        for v in c:
            if not isinstance(v, int) or isinstance(v, bool):
                # Fallback to a known valid structure if types are wrong (shouldn't happen)
                return [list(range(n))]

    if not result:
        # If we failed to find any, return a dummy valid cycle (0,1,2... is not a cycle, need a real one)
        # But the task says "nonempty list". If we find 0, we are invalid? 
        # "construct() -> nonempty list". 
        # If we can't find any, we should return something valid if possible, or let it fail.
        # However, the graph is dense and 3 cycles exist. This is just a safety net.
        # A simple cycle: 0-1-2-...-20-0 is not necessarily a cycle.
        # Let's just return an empty list? No, "nonempty".
        # If we truly fail, the score is 0.
        return []

    return result
# EVOLVE-BLOCK-END
