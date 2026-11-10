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

def _find_ham_cycle(adj, used_edges, start):
    """Find a Hamiltonian cycle starting/ending at `start` avoiding used_edges."""
    n = P['n']
    path = [start]
    visited = {start}
    
    def backtrack():
        if len(path) == n:
            # Check if last vertex connects back to start
            last = path[-1]
            if (last, start) not in used_edges and (start, last) not in used_edges:
                if start in adj[last]:
                    return True
            return False
        
        current = path[-1]
        # Order neighbors by degree (fewest options first) for better pruning
        neighbors = sorted(adj[current], key=lambda x: len([y for y in adj[x] if (x, y) not in used_edges and (y, x) not in used_edges]))
        
        for nb in neighbors:
            if nb in visited and nb != start:
                continue
            if nb == start and len(path) < n:
                continue
            # Check edge availability
            if (current, nb) in used_edges or (nb, current) in used_edges:
                continue
            if nb not in adj[current]:
                continue
            
            path.append(nb)
            visited.add(nb)
            if backtrack():
                return True
            path.pop()
            visited.discard(nb)
        
        return False
    
    if backtrack():
        return path
    return None

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    adj = _build_adjacency()
    n = P['n']
    cycles = []
    used_edges = set()
    
    # Try multiple random orderings of starting vertices
    random.seed(42)
    start_order = list(range(n))
    random.shuffle(start_order)
    
    for start in start_order:
        if time.time() > deadline:
            break
        if len(cycles) >= 10:
            break
        
        cycle = _find_ham_cycle(adj, used_edges, start)
        if cycle is None:
            continue
        
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
    
    # If we found fewer than expected, try with different random seeds
    if len(cycles) < 3:
        for seed in range(1, 20):
            if time.time() > deadline:
                break
            if len(cycles) >= 10:
                break
            
            # Reset and retry with new random ordering
            cycles = []
            used_edges = set()
            random.seed(seed)
            start_order = list(range(n))
            random.shuffle(start_order)
            
            for start in start_order:
                if time.time() > deadline:
                    break
                if len(cycles) >= 10:
                    break
                
                cycle = _find_ham_cycle(adj, used_edges, start)
                if cycle is None:
                    continue
                
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
            
            if len(cycles) >= 3:
                break
    
    # Ensure types are correct
    result = [list(c) for c in cycles]
    for c in result:
        for v in c:
            if not isinstance(v, int) or isinstance(v, bool):
                return [list(range(n))]
    
    if not result:
        return [list(range(n))]
    
    return result
# EVOLVE-BLOCK-END
