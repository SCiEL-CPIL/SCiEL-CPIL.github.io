"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency():
    adj = defaultdict(set)
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle_randomized(adj, forbidden_edges, start=0, max_depth=10000):
    """
    Find a Hamiltonian cycle using randomized DFS with Warnsdorff's heuristic.
    Returns list of vertices or None.
    """
    n = P['n']
    visited = {start}
    path = [start]
    used_edges = set()
    
    # Pre-compute initial available neighbors for start
    # We will use a recursive approach with randomization
    
    def dfs(current, depth):
        if depth >= max_depth:
            return None
            
        if len(path) == n:
            last = path[-1]
            edge = frozenset((last, start))
            if start in adj[last] and edge not in used_edges and edge not in forbidden_edges:
                return list(path)
            return None
        
        current = path[-1]
        # Get valid neighbors
        candidates = []
        for neighbor in adj[current]:
            if neighbor not in visited:
                edge = frozenset((current, neighbor))
                if edge not in used_edges and edge not in forbidden_edges:
                    # Heuristic: count available moves for neighbor (degree in remaining graph)
                    # This is expensive, so we approximate or just randomize.
                    # Let's just randomize for now, but sort by degree in full graph as a tie-breaker?
                    # Actually, simple randomization with restarts is often effective.
                    candidates.append(neighbor)
        
        if not candidates:
            return None
            
        # Randomize order
        random.shuffle(candidates)
        
        for neighbor in candidates:
            edge = frozenset((current, neighbor))
            visited.add(neighbor)
            used_edges.add(edge)
            path.append(neighbor)
            
            result = dfs(neighbor, depth + 1)
            if result is not None:
                return result
                
            path.pop()
            used_edges.remove(edge)
            visited.remove(neighbor)
            
        return None

    return dfs(start, 0)

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    # Try to find up to 3 cycles
    # We use a randomized approach with restarts for each cycle
    # Time limit: 60s. Let's reserve 55s for search.
    start_time = time.time()
    max_time = 55.0
    
    cycles_found = []
    forbidden_edges = set()
    
    # We want to find as many as possible.
    # Strategy: Find one cycle. Add its edges to forbidden. Find next.
    # If we fail to find the next, we might need to backtrack the previous choices?
    # That's complex. Instead, we can just try many random instances of finding a set of k cycles.
    # Or, simpler: Greedy randomized.
    # 1. Find a Hamiltonian cycle C1 (randomized).
    # 2. Find a Hamiltonian cycle C2 disjoint from C1 (randomized).
    # 3. Find a Hamiltonian cycle C3 disjoint from C1, C2 (randomized).
    # If step 2 or 3 fails, we can restart the whole process with different random seeds for C1.
    
    best_cycles = []
    
    # Try a few different random seeds/approaches
    for attempt in range(50):
        if time.time() - start_time > max_time:
            break
            
        current_forbidden = set()
        current_cycles = []
        
        success = True
        # Try to find 3 cycles
        for i in range(3):
            # Randomize the search by shuffling the adjacency list or just relying on random.shuffle in dfs
            # To make each attempt different, we can slightly perturb the graph view or just rely on internal randomness
            # Since _find_cycle_randomized uses random.shuffle, it is non-deterministic.
            
            # Add a small delay or just run it.
            cycle = _find_cycle_randomized(adj, current_forbidden)
            
            if cycle is None:
                success = False
                break
            
            # Add edges to forbidden
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                current_forbidden.add(frozenset((u, v)))
                
            current_cycles.append(cycle)
            
        if success and len(current_cycles) > len(best_cycles):
            best_cycles = current_cycles
            # If we found 3, that's the theoretical max, so we can stop early
            if len(best_cycles) == 3:
                break
                
    # Fallback if we didn't find 3, try to find 2, then 1
    if len(best_cycles) < 2:
        # The previous loop might have failed to find 2 in all 50 attempts? Unlikely.
        # But if best_cycles is empty, we should try a deterministic fallback or just return whatever we have.
        # If best_cycles has 1, we return 1.
        pass

    # Ensure types are correct
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
        
    return result

# EVOLVE-BLOCK-END
