import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency lists and sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
            
    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def find_hamiltonian_cycle(used_edges, deadline, rng=None):
        """Find a Hamiltonian cycle avoiding used_edges using backtracking with Warnsdorff heuristic."""
        path = [0]
        visited = {0}
        
        def dfs(current, remaining_count):
            if time.time() > deadline:
                return None
            
            if remaining_count == 0:
                ek = edge_key(current, 0)
                if ek not in used_edges and 0 in adj[current]:
                    return list(path)
                return None
            
            # Get unvisited neighbors with available edges
            candidates = []
            for nb in adj[current]:
                if nb not in visited:
                    ek = edge_key(current, nb)
                    if ek not in used_edges:
                        # Warnsdorff: count forward degree (unvisited neighbors with available edges)
                        fwd_deg = 0
                        for n2 in adj[nb]:
                            if n2 not in visited:
                                if edge_key(nb, n2) not in used_edges:
                                    fwd_deg += 1
                        candidates.append((fwd_deg, nb))
            
            if not candidates:
                return None
                
            # Sort by forward degree (fewest options first)
            if rng:
                candidates = [(deg, rng.random(), nb) for deg, nb in candidates]
                candidates.sort(key=lambda x: (x[0], x[1]))
                iter_candidates = [nb for _, _, nb in candidates]
            else:
                candidates.sort(key=lambda x: x[0])
                iter_candidates = [nb for _, nb in candidates]
            
            for nb in iter_candidates:
                path.append(nb)
                visited.add(nb)
                
                result = dfs(nb, remaining_count - 1)
                if result is not None:
                    return result
                    
                path.pop()
                visited.remove(nb)
                
            return None
        
        return dfs(0, n - 1)

    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    used_edges = set()
    cycles = []
    
    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    for attempt in range(3):
        if time.time() > deadline:
            break
            
        # First try deterministic
        cycle = find_hamiltonian_cycle(used_edges, deadline, rng=None)
        
        if not cycle:
            # Try with different random seeds
            found = False
            for seed in range(20):
                if time.time() > deadline:
                    break
                rng = random.Random(seed)
                cycle = find_hamiltonian_cycle(used_edges, deadline, rng=rng)
                if cycle:
                    found = True
                    break
            
            if not found:
                break

        if cycle:
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                used_edges.add(edge_key(u, v))
        else:
            break

    # Ensure result types are correct
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
        
    return result
# EVOLVE-BLOCK-END
