"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 70  # safety margin before 90s kill
    
    n = P['n']
    all_edges = set()
    for e in P['edges']:
        all_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    # Build adjacency
    adj = {v: set() for v in range(n)}
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_hamiltonian_cycle(available_edges, start_time, rng):
        """Find a Hamiltonian cycle using randomized greedy with backtracking."""
        if time.time() - start_time > deadline - 2:
            return None
        
        # Check degree constraints
        deg = {v: 0 for v in range(n)}
        for u, v in available_edges:
            deg[u] += 1
            deg[v] += 1
        
        for v in range(n):
            if deg[v] < 2:
                return None
        
        # Build adjacency from available edges
        adj_avail = {v: set() for v in range(n)}
        for u, v in available_edges:
            adj_avail[u].add(v)
            adj_avail[v].add(u)
        
        # Try multiple random starting vertices
        starts = list(range(n))
        rng.shuffle(starts)
        
        for start in starts:
            if time.time() - start_time > deadline - 2:
                return None
            
            # Randomized DFS with Warnsdorff-like heuristic
            for attempt in range(20):
                if time.time() - start_time > deadline - 2:
                    break
                
                path = [start]
                visited = {start}
                
                def backtrack():
                    if time.time() - start_time > deadline - 2:
                        return False
                    
                    if len(path) == n:
                        last = path[-1]
                        if start in adj_avail[last]:
                            return True
                        return False
                    
                    current = path[-1]
                    
                    # Get unvisited neighbors
                    unvisited_neighbors = [x for x in adj_avail[current] if x not in visited]
                    
                    if not unvisited_neighbors:
                        return False
                    
                    # Order by degree (prefer lower degree first - Warnsdorff)
                    # Add some randomness
                    if attempt < 10:
                        unvisited_neighbors.sort(key=lambda x: len(adj_avail[x]))
                    else:
                        rng.shuffle(unvisited_neighbors)
                    
                    for nb in unvisited_neighbors:
                        # Pruning: check if nb would be isolated
                        if nb != start:
                            # nb needs at least one unvisited neighbor (besides current)
                            # unless it's the last vertex
                            if len(path) < n - 1:
                                nb_unvisited = [x for x in adj_avail[nb] if x not in visited and x != current]
                                if not nb_unvisited:
                                    continue
                        
                        path.append(nb)
                        visited.add(nb)
                        
                        if backtrack():
                            return True
                        
                        path.pop()
                        visited.remove(nb)
                    
                    return False
                
                if backtrack():
                    return path[:]
        
        return None
    
    # Strategy: use randomized restarts to find edge-disjoint Hamiltonian cycles
    best_cycles = []
    best_count = 0
    
    # Try many different random seeds
    for seed in range(200):
        if time.time() - start_time > deadline - 5:
            break
        
        rng = random.Random(seed)
        cycles = []
        available = set(all_edges)
        
        while time.time() - start_time < deadline - 5:
            if len(available) < n:
                break
            
            cycle = find_hamiltonian_cycle(available, start_time, rng)
            if cycle is None:
                break
            
            # Remove edges of this cycle from available
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in available:
                    available.remove(edge)
            
            cycles.append(cycle)
        
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles[:]
        
        # If we found 3 cycles (theoretical max), stop
        if best_count >= 3:
            break
    
    # Ensure we return at least 1 cycle
    if not best_cycles:
        # Fallback: find any Hamiltonian cycle
        rng = random.Random(0)
        cycle = find_hamiltonian_cycle(all_edges, start_time, rng)
        if cycle:
            best_cycles = [cycle]
        else:
            # Last resort: find any 3-cycle
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v and w in adj[u]:
                                best_cycles = [[u, v, w]]
                                break
                        if best_cycles:
                            break
                if best_cycles:
                    break
            if not best_cycles:
                best_cycles = [[0, 1, 2]]
    
    # Convert to list of lists of ints
    result = []
    for c in best_cycles:
        result.append([int(v) for v in c])
    
    return result
# EVOLVE-BLOCK-END
