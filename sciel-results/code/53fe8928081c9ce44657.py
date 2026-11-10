"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(edges):
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycles(adj, used_edges, n, max_cycles=10, deadline=None, rng=None):
    """
    Find edge-disjoint Hamiltonian cycles using randomized DFS with restarts.
    Returns list of cycles found.
    """
    if rng is None:
        rng = random.Random()
    
    cycles = []
    all_vertices = list(range(n))
    
    # We'll try to find up to max_cycles edge-disjoint Hamiltonian cycles
    for cycle_idx in range(max_cycles):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try multiple random restarts for this cycle
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if deadline is not None and time.time() > deadline:
                break
            
            # Randomly shuffle the order in which we try to build the cycle
            shuffled_vertices = list(all_vertices)
            rng.shuffle(shuffled_vertices)
            
            # Pick a random start vertex from the shuffled list
            start = shuffled_vertices[0]
            
            # DFS to find a Hamiltonian cycle starting and ending at start
            # that avoids used_edges
            def dfs(path, visited, current):
                if len(path) == n:
                    # Check if we can close the cycle
                    if (current, start) not in used_edges and (start, current) not in used_edges:
                        if start in adj[current]:
                            return path
                    return None
                
                # Get neighbors of current, sorted randomly for diversity
                neighbors = list(adj[current])
                rng.shuffle(neighbors)
                
                for next_v in neighbors:
                    if next_v in visited:
                        continue
                    # Check if edge (current, next_v) is available
                    if (current, next_v) in used_edges or (next_v, current) in used_edges:
                        continue
                    if next_v not in adj[current]:
                        continue
                    
                    new_path = path + [next_v]
                    new_visited = visited | {next_v}
                    result = dfs(new_path, new_visited, next_v)
                    if result is not None:
                        return result
                return None
            
            result = dfs([start], {start}, start)
            if result is not None:
                cycles.append(result)
                # Mark edges as used
                for i in range(len(result)):
                    u = result[i]
                    v = result[(i + 1) % len(result)]
                    used_edges.add((u, v))
                    used_edges.add((v, u))
                found = True
                break
        
        if not found:
            break
    
    return cycles

def construct():
    deadline = time.time() + 55  # Safety margin before 60s limit
    n = P['n']
    adj = _build_adjacency(P['edges'])
    used_edges = set()
    
    # Try to find edge-disjoint Hamiltonian cycles
    # Use multiple random seeds for diversity
    best_cycles = []
    
    for seed in range(20):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        used_edges_copy = set()
        cycles = _find_hamiltonian_cycles(adj, used_edges_copy, n, max_cycles=10, deadline=deadline, rng=rng)
        
        if len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) >= 10:
                break
    
    # Convert to proper format: list of lists of ints
    result = []
    for cycle in best_cycles:
        # Ensure all vertices are Python ints
        cycle_ints = [int(v) for v in cycle]
        result.append(cycle_ints)
    
    # If we couldn't find any cycles, return a single valid cycle as fallback
    if not result:
        # Try to find at least one Hamiltonian cycle with a simple approach
        rng = random.Random(42)
        used_edges_copy = set()
        cycles = _find_hamiltonian_cycles(adj, used_edges_copy, n, max_cycles=1, deadline=deadline, rng=rng)
        for cycle in cycles:
            result.append([int(v) for v in cycle])
    
    # Final safety: if still empty, use a known cycle
    if not result:
        # Simple cycle: 0-1-2-3-4-5-6-7-8-9-10-11-12-13-14-15-16-17-18-19-20-0
        # Check if this is valid
        cycle = list(range(n))
        valid = True
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]
            if v not in adj[u]:
                valid = False
                break
        if valid:
            result = [cycle]
        else:
            # Find any cycle of length >= 3
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w != u and w != v and w in adj[u]:
                                result = [[u, v, w]]
                                break
                        if result:
                            break
                if result:
                    break
    
    return result
# EVOLVE-BLOCK-END
