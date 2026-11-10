import time
import sys
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

n = P['n']
adj = {i: [] for i in range(n)}
all_edges = set()
for u, v in P['edges']:
    adj[u].append(v)
    adj[v].append(u)
    if u < v:
        all_edges.add((u, v))
    else:
        all_edges.add((v, u))

def get_edge(u, v):
    if u < v:
        return (u, v)
    else:
        return (v, u)

def find_hamiltonian_cycle(available_edges, time_limit=5.0, start_time=None):
    """
    Finds a Hamiltonian cycle using backtracking with degree-based heuristics.
    """
    if start_time is None:
        start_time = time.time()
    
    # Build local adjacency from available edges
    local_adj = {i: [] for i in range(n)}
    for u, v in available_edges:
        local_adj[u].append(v)
        local_adj[v].append(u)
    
    # Precompute degrees for heuristic ordering
    deg = {i: len(local_adj[i]) for i in range(n)}
    
    path = []
    visited = [False] * n
    
    def dfs(current, depth):
        if time.time() - start_time > time_limit:
            return None
        
        if depth == n:
            edge = get_edge(current, path[0])
            if edge in available_edges:
                return list(path)
            return None
        
        # Sort neighbors by degree (prefer lower degree first - more constrained)
        neighbors = sorted(local_adj[current], key=lambda x: (deg[x], x))
        
        for neighbor in neighbors:
            if not visited[neighbor]:
                edge = get_edge(current, neighbor)
                if edge in available_edges:
                    visited[neighbor] = True
                    path.append(neighbor)
                    
                    result = dfs(neighbor, depth + 1)
                    if result is not None:
                        return result
                    
                    path.pop()
                    visited[neighbor] = False
        
        return None
    
    # Try starting from each vertex, preferring lower degree
    start_vertices = sorted(range(n), key=lambda x: (deg[x], x))
    
    for start in start_vertices:
        if time.time() - start_time > time_limit:
            break
        path = [start]
        visited = [False] * n
        visited[start] = True
        
        result = dfs(start, 1)
        if result is not None:
            return result
    
    return None

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    best_cycles = []
    
    # Strategy 1: Greedy with degree-based heuristics
    current_edges = set(all_edges)
    cycles_found = []
    
    for i in range(3):
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1.0:
            break
        
        # Allocate time proportional to remaining cycles
        time_limit = min(remaining_time - 0.5, 15.0)
        
        cycle = find_hamiltonian_cycle(current_edges, time_limit=time_limit, start_time=time.time())
        
        if cycle is None:
            break
        
        cycles_found.append(cycle)
        
        # Remove edges from current_edges
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j + 1) % len(cycle)]
            current_edges.discard(get_edge(u, v))
    
    if len(cycles_found) == 3:
        return [list(c) for c in cycles_found]
    
    # Strategy 2: If greedy failed, try with different random seeds
    for seed in [0, 1, 2, 3, 4]:
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        random.seed(seed)
        current_edges = set(all_edges)
        cycles_found = []
        
        for i in range(3):
            sub_remaining = remaining_time - (time.time() - start_time)
            if sub_remaining < 0.5:
                break
            
            # Shuffle adjacency lists for diversity
            for v in range(n):
                random.shuffle(adj[v])
            
            time_limit = min(sub_remaining - 0.1, 10.0)
            cycle = find_hamiltonian_cycle(current_edges, time_limit=time_limit, start_time=time.time())
            
            if cycle is None:
                break
            
            cycles_found.append(cycle)
            
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_edges.discard(get_edge(u, v))
        
        if len(cycles_found) > len(best_cycles):
            best_cycles = cycles_found
        
        if len(best_cycles) == 3:
            return [list(c) for c in best_cycles]
    
    # Strategy 3: More aggressive random restarts
    for seed in range(5, 20):
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1.0:
            break
        
        random.seed(seed)
        current_edges = set(all_edges)
        cycles_found = []
        
        for i in range(3):
            sub_remaining = remaining_time - (time.time() - start_time)
            if sub_remaining < 0.3:
                break
            
            for v in range(n):
                random.shuffle(adj[v])
            
            time_limit = min(sub_remaining - 0.1, 5.0)
            cycle = find_hamiltonian_cycle(current_edges, time_limit=time_limit, start_time=time.time())
            
            if cycle is None:
                break
            
            cycles_found.append(cycle)
            
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_edges.discard(get_edge(u, v))
        
        if len(cycles_found) > len(best_cycles):
            best_cycles = cycles_found
        
        if len(best_cycles) == 3:
            return [list(c) for c in best_cycles]
    
    return [list(c) for c in best_cycles]
# EVOLVE-BLOCK-END
