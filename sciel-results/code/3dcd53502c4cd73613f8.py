"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_adjacency(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, vertices, used_edges, deadline, rng):
    """Find a Hamiltonian cycle on the given vertex set using DFS with backtracking."""
    n = len(vertices)
    if n < 3:
        return None
    
    # Start from the smallest vertex for canonicalization
    start = min(vertices)
    vertex_set = set(vertices)
    
    # Precompute available neighbors for each vertex
    # A neighbor is available if the edge is not in used_edges and the neighbor is in vertex_set
    def available_neighbors(v, path_set):
        neighbors = []
        for nb in adj[v]:
            if nb not in vertex_set:
                continue
            if nb in path_set:
                # Only allow closing the cycle if we're at the end and nb is start
                if len(path_set) == n and nb == start:
                    if _edge_key(v, nb) not in used_edges:
                        neighbors.append(nb)
                continue
            if _edge_key(v, nb) not in used_edges:
                neighbors.append(nb)
        return neighbors
    
    def dfs(path, path_set):
        if time.time() > deadline:
            return None
        v = path[-1]
        if len(path) == n:
            # Check if we can close the cycle
            if _edge_key(v, start) not in used_edges and start in vertex_set:
                return path[:]
            return None
        
        neighbors = available_neighbors(v, path_set)
        # Shuffle for randomized search
        rng.shuffle(neighbors)
        
        for nb in neighbors:
            new_path = path + [nb]
            new_path_set = path_set | {nb}
            result = dfs(new_path, new_path_set)
            if result is not None:
                return result
        return None
    
    initial_path = [start]
    initial_set = {start}
    result = dfs(initial_path, initial_set)
    return result

def _find_hamiltonian_cycle_fast(adj, vertices, used_edges, deadline, rng):
    """Faster Hamiltonian cycle finder using iterative DFS with heuristics."""
    n = len(vertices)
    if n < 3:
        return None
    
    start = min(vertices)
    vertex_set = set(vertices)
    
    # Use Warnsdorff-like heuristic: prefer vertices with fewer available moves
    def get_score(v, path_set, remaining):
        count = 0
        for nb in adj[v]:
            if nb in remaining and nb != start and _edge_key(v, nb) not in used_edges:
                count += 1
        if len(remaining) == 1 and start in remaining and _edge_key(v, start) not in used_edges:
            count += 1
        return count
    
    # Iterative DFS
    stack = [(start, {start}, [start])]
    visited_paths = set()
    
    while stack:
        if time.time() > deadline:
            return None
        
        path, path_set, remaining = stack[-1]
        v = path[-1]
        
        if not remaining:
            # Check if we can close
            if _edge_key(v, start) not in used_edges:
                return path
            stack.pop()
            continue
        
        # Get available neighbors in remaining
        available = []
        for nb in remaining:
            if nb == start and len(remaining) > 1:
                continue
            if _edge_key(v, nb) not in used_edges:
                available.append(nb)
        
        if not available:
            stack.pop()
            continue
        
        # Sort by heuristic: prefer nodes with fewer future options
        available.sort(key=lambda nb: get_score(nb, path_set, remaining - {nb}))
        rng.shuffle(available)  # Add some randomness
        
        for nb in available:
            new_remaining = remaining - {nb}
            new_path = path + [nb]
            new_path_set = path_set | {nb}
            stack.append((new_path, new_path_set, new_remaining))
            break  # Only push one, will backtrack if it fails
    
    return None

def construct():
    deadline = time.time() + 55.0  # 55 second deadline for safety
    rng = random.Random(42)
    
    edges = P['edges']
    n = P['n']
    all_vertices = set(range(n))
    adj = _build_adjacency(edges)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    # Strategy: Iteratively find edge-disjoint Hamiltonian cycles
    # Each cycle must use ALL 21 vertices
    
    used_edges = set()
    cycles = []
    
    # Try to find as many edge-disjoint Hamiltonian cycles as possible
    # Maximum possible: 70 edges / 21 edges per cycle = 3.33, so at most 3
    
    for attempt in range(10):  # Try up to 10 times with different randomizations
        if time.time() > deadline - 5.0:
            break
        
        # Reset for each attempt with different seed
        rng_attempt = random.Random(attempt * 1000 + 42)
        
        # Try to find cycles one by one
        temp_used = set(used_edges)
        temp_cycles = list(cycles)
        
        success = True
        for i in range(7):  # Try to find up to 7 more cycles (but limited by edges)
            if time.time() > deadline - 5.0:
                success = False
                break
            
            # Check if we have enough edges left
            remaining_edges = len(edge_set) - len(temp_used)
            if remaining_edges < n:
                break
            
            # Find a Hamiltonian cycle using only unused edges
            cycle = _find_hamiltonian_cycle(adj, list(all_vertices), temp_used, deadline - 5.0, rng_attempt)
            
            if cycle is None:
                # Try with the fast version
                cycle = _find_hamiltonian_cycle_fast(adj, list(all_vertices), temp_used, deadline - 5.0, rng_attempt)
            
            if cycle is None:
                if i == 0 and not temp_cycles:
                    # Can't even find one cycle, break
                    break
                success = False
                break
            
            # Add edges of this cycle to used set
            for i_idx in range(len(cycle)):
                u = cycle[i_idx]
                v = cycle[(i_idx + 1) % len(cycle)]
                ek = _edge_key(u, v)
                if ek in temp_used:
                    # Shouldn't happen, but just in case
                    success = False
                    break
                temp_used.add(ek)
            
            if not success:
                break
            
            temp_cycles.append(cycle)
        
        if len(temp_cycles) > len(cycles):
            cycles = temp_cycles
            used_edges = temp_used
    
    # If we couldn't find any cycles, return a simple one
    if not cycles:
        # Find any Hamiltonian cycle
        cycle = _find_hamiltonian_cycle(adj, list(all_vertices), set(), deadline - 1.0, rng)
        if cycle is not None:
            cycles = [cycle]
        else:
            # Fallback: return a trivial cycle if possible (shouldn't happen for this graph)
            cycles = [list(range(n))]
    
    # Convert to proper format: list of lists of int
    result = []
    for cycle in cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
