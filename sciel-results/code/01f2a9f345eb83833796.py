"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, start=0, deadline=None):
    """Find a Hamiltonian cycle in the given adjacency using DFS with backtracking."""
    
    def dfs(path, visited, current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if current in adj.get(start, set()):
                return path
            return None
        for neighbor in adj.get(current, set()):
            if neighbor not in visited:
                # Pruning: if this is the last vertex to visit, check if it connects to start
                if len(path) == n - 1:
                    if neighbor in adj.get(start, set()):
                        return path + [neighbor]
                else:
                    result = dfs(path + [neighbor], visited | {neighbor}, neighbor)
                    if result is not None:
                        return result
        return None
    
    # Try from start vertex
    return dfs([start], {start}, start)

def _find_hamiltonian_cycle_random(adj, n, start=0, deadline=None, rng=None):
    """Find a Hamiltonian cycle with randomized neighbor ordering."""
    if rng is None:
        rng = random.Random()
    
    def dfs(path, visited, current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if current in adj.get(start, set()):
                return path
            return None
        neighbors = list(adj.get(current, set()))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                if len(path) == n - 1:
                    if neighbor in adj.get(start, set()):
                        return path + [neighbor]
                else:
                    result = dfs(path + [neighbor], visited | {neighbor}, neighbor)
                    if result is not None:
                        return result
        return None
    
    return dfs([start], {start}, start)

def _remove_cycle_edges(edges, cycle, n):
    """Remove edges used by the cycle from the edge list."""
    cycle_edges = set()
    for i in range(len(cycle)):
        u = cycle[i]
        v = cycle[(i + 1) % len(cycle)]
        cycle_edges.add((min(u, v), max(u, v)))
    
    remaining = []
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key not in cycle_edges:
            remaining.append([u, v])
    return remaining

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    edges = [list(e) for e in P['edges']]
    
    # First cycle: the natural ordering
    first_cycle = list(range(n))
    # Verify it's valid
    adj0 = _build_adjacency(edges, n)
    valid = True
    for i in range(n):
        u = first_cycle[i]
        v = first_cycle[(i + 1) % n]
        if v not in adj0.get(u, set()):
            valid = False
            break
    if not valid:
        # Fallback: try to find one
        first_cycle = _find_hamiltonian_cycle(adj0, n, 0, deadline)
        if first_cycle is None:
            return []
    
    cycles = [first_cycle]
    
    # Remove first cycle's edges
    remaining_edges = _remove_cycle_edges(edges, first_cycle, n)
    
    # Find second cycle
    adj1 = _build_adjacency(remaining_edges, n)
    second_cycle = None
    
    # Try deterministic first
    if time.time() < deadline:
        second_cycle = _find_hamiltonian_cycle(adj1, n, 0, deadline)
    
    # If not found, try randomized restarts
    if second_cycle is None:
        rng = random.Random(42)
        max_restarts = 100
        for attempt in range(max_restarts):
            if time.time() > deadline - 5:
                break
            second_cycle = _find_hamiltonian_cycle_random(adj1, n, 0, deadline, rng)
            if second_cycle is not None:
                break
    
    if second_cycle is not None:
        cycles.append(second_cycle)
        remaining_edges = _remove_cycle_edges(remaining_edges, second_cycle, n)
        
        # Find third cycle
        adj2 = _build_adjacency(remaining_edges, n)
        third_cycle = None
        
        if time.time() < deadline:
            third_cycle = _find_hamiltonian_cycle(adj2, n, 0, deadline)
        
        if third_cycle is None:
            rng2 = random.Random(123)
            max_restarts2 = 100
            for attempt in range(max_restarts2):
                if time.time() > deadline - 2:
                    break
                third_cycle = _find_hamiltonian_cycle_random(adj2, n, 0, deadline, rng2)
                if third_cycle is not None:
                    break
        
        if third_cycle is not None:
            cycles.append(third_cycle)
    
    # Convert to proper types
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
