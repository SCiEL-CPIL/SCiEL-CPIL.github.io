"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

# Precompute adjacency
N = P['n']
ADJ = defaultdict(set)
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

# Edge set for quick lookup
EDGE_SET = set()
for u, v in P['edges']:
    if u > v:
        u, v = v, u
    EDGE_SET.add((u, v))

def _deadline_check(deadline):
    return time.time() < deadline

def _find_hamiltonian_cycle(used_edges, deadline):
    """Find a Hamiltonian cycle using DFS with backtracking, avoiding used edges."""
    n = N
    # Try to find a Hamiltonian cycle in the remaining graph
    # Use a heuristic: start from vertex 0
    
    # Build remaining adjacency
    rem_adj = defaultdict(set)
    for u in range(n):
        for v in ADJ[u]:
            e = (min(u, v), max(u, v))
            if e not in used_edges:
                rem_adj[u].add(v)
    
    # Check if all vertices have degree >= 2 in remaining graph
    for u in range(n):
        if len(rem_adj[u]) < 2:
            return None
    
    # DFS to find Hamiltonian cycle
    def dfs(path, visited):
        if not _deadline_check(deadline):
            return None
        if len(path) == n:
            # Check if last vertex connects to first
            last = path[-1]
            first = path[0]
            e = (min(last, first), max(last, first))
            if e not in used_edges:
                return list(path)
            return None
        
        cur = path[-1]
        # Try neighbors in some order
        neighbors = list(rem_adj[cur])
        random.shuffle(neighbors)
        
        for nxt in neighbors:
            if nxt not in visited:
                new_path = path + [nxt]
                new_visited = visited | {nxt}
                result = dfs(new_path, new_visited)
                if result is not None:
                    return result
        return None
    
    result = dfs([0], {0})
    return result

def _canonical_cycle(cycle):
    """Return a canonical representation of a cycle for comparison."""
    if not cycle:
        return ()
    # Rotate so smallest vertex is first
    min_idx = cycle.index(min(cycle))
    rotated = cycle[min_idx:] + cycle[:min_idx]
    return tuple(rotated)

def construct():
    """Find maximum number of edge-disjoint Hamiltonian cycles."""
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s margin
    
    # Strategy: iteratively find Hamiltonian cycles, removing their edges
    used_edges = set()
    cycles = []
    
    # Multiple attempts with different random seeds
    best_cycles = []
    
    for attempt in range(100):
        if not _deadline_check(deadline):
            break
        
        random.seed(attempt * 7919 + 13)
        used_edges = set()
        cycles = []
        
        for _ in range(10):  # Max 10 cycles
            if not _deadline_check(deadline):
                break
            
            cycle = _find_hamiltonian_cycle(used_edges, deadline)
            if cycle is None:
                break
            
            # Add edges to used set
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                if e in used_edges:
                    break
                used_edges.add(e)
            else:
                cycles.append(list(cycle))
        
        if len(cycles) > len(best_cycles):
            best_cycles = list(cycles)
            if len(best_cycles) >= 3:  # Theoretical max for 6-regular
                break
    
    # Verify and return
    if not best_cycles:
        # Fallback: return a single valid cycle if we can find one
        random.seed(42)
        used_edges = set()
        cycle = _find_hamiltonian_cycle(used_edges, deadline)
        if cycle:
            best_cycles = [list(cycle)]
        else:
            # Last resort: return empty (shouldn't happen)
            best_cycles = [list(range(N))]
    
    # Ensure types are correct (Python int, list)
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
    
    return result

# EVOLVE-BLOCK-END
