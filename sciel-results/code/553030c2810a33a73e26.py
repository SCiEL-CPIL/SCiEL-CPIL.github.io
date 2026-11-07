"""Find edge-disjoint Hamiltonian cycles on the 21-vertex 6-regular host graph."""
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges, n):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, start=0, rng=None, deadline=None):
    """Find a Hamiltonian cycle starting at `start` using backtracking + Warnsdorff heuristic."""
    if deadline is not None and time.time() > deadline:
        return None
    
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    def dfs(current):
        if deadline is not None and time.time() > deadline:
            return None
        if len(path) == n:
            if start in adj[current]:
                return list(path)
            return None
        
        # Get unvisited neighbors
        neighbors = [v for v in adj[current] if not visited[v]]
        if not neighbors:
            return None
        
        # Warnsdorff: sort by number of onward unvisited neighbors (ascending)
        if rng is not None:
            # Add slight randomization to avoid getting stuck in bad orderings
            scored = []
            for v in neighbors:
                onward = sum(1 for w in adj[v] if not visited[w] and w != start)
                # Add small random perturbation
                scored.append((onward + rng.random() * 0.5, v))
            scored.sort()
            neighbors = [v for _, v in scored]
        else:
            neighbors.sort(key=lambda v: sum(1 for w in adj[v] if not visited[w] and w != start))
        
        for v in neighbors:
            visited[v] = True
            path.append(v)
            result = dfs(v)
            if result is not None:
                return result
            path.pop()
            visited[v] = False
        
        return None
    
    return dfs(start)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    n = P['n']
    edges = P['edges']
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # Build adjacency
    adj = _build_adjacency(edges, n)
    
    # Try to find 3 edge-disjoint Hamiltonian cycles
    # Strategy: iteratively find a Hamiltonian cycle, remove its edges, repeat
    
    best_cycles = []
    
    # Use a fixed seed for reproducibility, but try multiple seeds if first fails
    for seed in [42, 123, 999, 7, 314, 271, 137, 555, 888, 101, 202, 303]:
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        cycles = []
        current_adj = {v: set(neighbors) for v, neighbors in adj.items()}
        remaining_edges = set(edge_set)
        
        for cycle_idx in range(3):
            if time.time() > deadline:
                break
            
            # Try multiple starting vertices
            found = False
            starts = list(range(n))
            rng.shuffle(starts)
            
            for start in starts:
                if time.time() > deadline:
                    break
                
                # Find Hamiltonian cycle in current_adj
                cycle = _find_hamiltonian_cycle(current_adj, n, start=start, rng=rng, deadline=deadline)
                
                if cycle is not None:
                    # Verify the cycle uses only remaining edges
                    valid = True
                    cycle_edges = set()
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e not in remaining_edges:
                            valid = False
                            break
                        cycle_edges.add(e)
                    
                    if valid and len(cycle) == n and len(set(cycle)) == n:
                        cycles.append(cycle)
                        # Remove these edges
                        for e in cycle_edges:
                            remaining_edges.discard(e)
                            u, v = e
                            current_adj[u].discard(v)
                            current_adj[v].discard(u)
                        found = True
                        break
            
            if not found:
                break
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
        
        if len(best_cycles) == 3:
            break
    
    # Return the best set of cycles found
    if not best_cycles:
        # Fallback: return a single cycle
        adj_full = _build_adjacency(edges, n)
        cycle = _find_hamiltonian_cycle(adj_full, n, start=0, rng=random.Random(0), deadline=deadline)
        if cycle is not None:
            return [cycle]
        return [list(range(n))]
    
    return best_cycles
