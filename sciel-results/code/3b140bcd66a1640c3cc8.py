"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_connected(vertices, adj, used_edges):
    """Check if the graph induced by vertices with edges not in used_edges is connected."""
    if not vertices:
        return True
    
    vset = set(vertices)
    start = next(iter(vset))
    visited = {start}
    stack = [start]
    
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in vset and v not in visited:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    visited.add(v)
                    stack.append(v)
    
    return len(visited) == len(vset)


def _find_ham_cycle(vertices, adj, used_edges, deadline, rng):
    """Find a Hamiltonian cycle on the given vertex set using iterative DFS with backtracking."""
    vertices = list(vertices)
    n = len(vertices)
    if n < 3:
        return None
    
    vset = set(vertices)
    start = vertices[0]
    
    # Precompute adjacency as frozensets for faster lookup
    adj_frozen = {u: frozenset(v) for u, v in adj.items()}
    
    # Use the precomputed global edge set
    all_graph_edges = _ALL_EDGES
    
    # Count available edges (not in used_edges)
    available_edges = all_graph_edges - used_edges
    
    # Check if enough edges remain for a Hamiltonian cycle
    if len(available_edges) < n:
        return None
    
    # Precompute neighbor lists sorted by degree for initial ordering
    def _get_neighbors_sorted(current, visited, avail_edges):
        """Get unvisited neighbors sorted by remaining degree (Warnsdorff's heuristic)."""
        neighbors = []
        for nb in adj_frozen[current]:
            if nb in vset and nb not in visited:
                e = (min(current, nb), max(current, nb))
                if e in avail_edges:
                    # Compute degree in remaining graph
                    deg = 0
                    for y in adj_frozen[nb]:
                        if y in vset and y not in visited and y != current:
                            e2 = (min(nb, y), max(nb, y))
                            if e2 in avail_edges:
                                deg += 1
                    neighbors.append((deg, rng.random(), nb))
        neighbors.sort()
        return [nb for _, _, nb in neighbors]
    
    # Iterative DFS
    visited = {start}
    path = [start]
    stack = [(start, 0, _get_neighbors_sorted(start, visited, available_edges))]
    
    while stack:
        if time.time() > deadline:
            return None
        
        current, idx, neighbors = stack[-1]
        
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj_frozen[current]:
                e = (min(current, start), max(current, start))
                if e in available_edges:
                    return path
            # Can't close, backtrack
            stack.pop()
            if stack:
                path.pop()
                visited.discard(current)
                prev = stack[-1][0]
                e = (min(prev, current), max(prev, current))
                available_edges.add(e)
            continue
        
        if idx < len(neighbors):
            # Try next neighbor
            nb = neighbors[idx]
            stack[-1] = (current, idx + 1, neighbors)
            
            e = (min(current, nb), max(current, nb))
            if e in available_edges:
                available_edges.discard(e)
                visited.add(nb)
                path.append(nb)
                new_neighbors = _get_neighbors_sorted(nb, visited, available_edges)
                stack.append((nb, 0, new_neighbors))
        else:
            # No more neighbors, backtrack
            stack.pop()
            if stack:
                path.pop()
                visited.discard(current)
                prev = stack[-1][0]
                e = (min(prev, current), max(prev, current))
                available_edges.add(e)
    
    return None

# Precompute all graph edges once
_ALL_EDGES = set()
for _u, _v in P['edges']:
    _ALL_EDGES.add((min(_u, _v), max(_u, _v)))


def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 85 seconds to be safe (90s limit, leave margin)
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    all_vertices = list(range(n))
    max_possible_cycles = 3  # Theoretical max: 70 edges / 21 per cycle = 3.33, so max 3
    
    best_cycles = []
    best_cycle_count = 0
    
    # Try multiple random restarts to ensure we find the maximum set
    for attempt in range(100):
        if time.time() > total_deadline:
            break
        
        # Check if we already found the max
        if best_cycle_count == max_possible_cycles:
            break
        
        used_edges = set()
        cycles = []
        
        for i in range(max_possible_cycles):
            if time.time() > total_deadline - 1:
                break
                
            # Check connectivity before expensive search
            if not _is_connected(all_vertices, adj, used_edges):
                break
                
            # Use a unique RNG for each attempt to vary the search
            rng = random.Random(attempt * 7919 + i * 104729 + 42)
            
            # Allocate time: more time for earlier cycles (they're easier), less for later
            remaining_time = total_deadline - time.time()
            cycles_remaining = max_possible_cycles - i
            # Give each cycle a proportional share with some buffer
            cycle_deadline = time.time() + max(2, remaining_time / cycles_remaining * 0.7)
            
            cycle = _find_ham_cycle(all_vertices, adj, used_edges, cycle_deadline, rng)
            
            if cycle is None:
                break
            
            # Add edges of this cycle to used_edges
            for j in range(len(cycle)):
                u = cycle[j]
                v = cycle[(j + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                used_edges.add(e)
            
            cycles.append(cycle)
        
        # Update best if we found more
        if len(cycles) > best_cycle_count:
            best_cycles = cycles
            best_cycle_count = len(cycles)
            if best_cycle_count == max_possible_cycles:
                break  # Found the theoretical max, stop searching

    # Convert to proper types
    result = [list(c) for c in best_cycles]
    
    if not result:
        # Fallback: try to find at least one cycle with a simple approach
        used_edges_fb = set()
        for i in range(20):
            rng_fb = random.Random(12345 + i)
            c = _find_ham_cycle(all_vertices, adj, used_edges_fb, time.time() + 5, rng_fb)
            if c is not None:
                result = [list(c)]
                break
        if not result:
            result = [list(range(n))]

    return result
# EVOLVE-BLOCK-END
