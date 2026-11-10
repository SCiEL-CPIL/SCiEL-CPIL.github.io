"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(forbidden=None):
    """Build adjacency list, optionally excluding forbidden edges."""
    n = P['n']
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        if forbidden is not None:
            key = (min(u, v), max(u, v))
            if key in forbidden:
                continue
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_hamiltonian_cycle(adj, start, deadline, rng):
    """Find a Hamiltonian cycle starting at 'start' using DFS with backtracking."""
    n = P['n']
    # Shuffle neighbors for diversity
    for v in adj:
        rng.shuffle(adj[v])
    
    path = [start]
    visited = [False] * n
    visited[start] = True
    
    def dfs(u):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[u]:
                return list(path)
            return None
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                path.append(v)
                result = dfs(v)
                if result is not None:
                    return result
                path.pop()
                visited[v] = False
        return None
    
    return dfs(start)

def _find_two_hamiltonian_cycles(deadline, rng):
    """Find two edge-disjoint Hamiltonian cycles."""
    n = P['n']
    best_result = None
    
    # Try different starting vertices and orderings
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        
        # Find first Hamiltonian cycle
        adj1 = _build_adjacency(forbidden=None)
        cycle1 = _find_hamiltonian_cycle(adj1, start, deadline, rng)
        if cycle1 is None:
            continue
        
        # Extract edges of cycle1
        forbidden = set()
        for i in range(len(cycle1)):
            u = cycle1[i]
            v = cycle1[(i + 1) % len(cycle1)]
            forbidden.add((min(u, v), max(u, v)))
        
        # Find second Hamiltonian cycle avoiding forbidden edges
        adj2 = _build_adjacency(forbidden=forbidden)
        cycle2 = _find_hamiltonian_cycle(adj2, start, deadline, rng)
        if cycle2 is not None:
            return [cycle1, cycle2]
        
        # Try different start for second cycle
        for start2 in starts:
            if time.time() > deadline:
                break
            adj2 = _build_adjacency(forbidden=forbidden)
            cycle2 = _find_hamiltonian_cycle(adj2, start2, deadline, rng)
            if cycle2 is not None:
                return [cycle1, cycle2]
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # Deterministic first attempt
    rng = random.Random(42)
    
    result = _find_two_hamiltonian_cycles(deadline, rng)
    if result is not None and len(result) >= 2:
        # Validate and return
        return [list(c) for c in result[:2]]
    
    # Fallback: try with different seeds
    for seed in [0, 1, 2, 3, 7, 13, 42, 99, 123, 456, 789, 1000]:
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        result = _find_two_hamiltonian_cycles(deadline, rng)
        if result is not None and len(result) >= 2:
            return [list(c) for c in result[:2]]
    
    # Ultimate fallback: return a single valid Hamiltonian cycle
    adj = _build_adjacency(forbidden=None)
    rng = random.Random(0)
    for start in range(P['n']):
        if time.time() > deadline:
            break
        cycle = _find_hamiltonian_cycle(adj, start, deadline, rng)
        if cycle is not None:
            return [list(cycle)]
    
    # Last resort: return trivial cycle if any exists (shouldn't happen)
    return [list(range(P['n']))]
# EVOLVE-BLOCK-END
