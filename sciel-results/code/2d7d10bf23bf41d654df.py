import time
import itertools
from collections import deque

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Constraints:
    - Min degree >= 3
    - Girth >= 4 (Triangle-free)
    - Contains a cycle
    - Minimize number of distinct simple cycle lengths.
    
    Strategy: Try to find a graph with 1 distinct cycle length (all cycles same length).
    If not found within time budget, fall back to 2 distinct cycle lengths (two K_{3,3}).
    
    For 1 distinct cycle length, we need all simple cycles to have the same length.
    Since min degree >= 3 and triangle-free, shortest cycle is at least 4.
    A graph where ALL cycles have length 4 would be ideal.
    """
    start_time = time.time()
    deadline = start_time + 60.0  # Must finish within 60s per task, use 60s as safety
    
    # First, let's try to find a graph with only 4-cycles (1 distinct cycle length)
    # This requires a bipartite graph where every cycle has length 4.
    # Such graphs are known as "4-cycle-free" in the complement sense, but actually
    # we want ONLY 4-cycles.
    
    # A complete bipartite graph K_{m,n} has cycles of lengths 4, 6, 8, ...
    # So K_{m,n} won't work for a single cycle length unless it has no cycles > 4.
    
    # What about a graph that's a "linear" arrangement of 4-cycles?
    # Actually, let me think about this differently.
    
    # Key insight: If we can find a graph where the girth equals the maximum cycle length,
    # then all cycles have the same length. But in practice, this is very restrictive.
    
    # Let's try a different approach: use a search to find graphs with minimal distinct cycle lengths.
    
    # Start with the known good solution: two K_{3,3} components (2 distinct lengths: 4, 6)
    fallback_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            fallback_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            fallback_edges.append([u, v])
    
    def get_distinct_cycle_lengths(edges):
        """Get the set of distinct cycle lengths in the graph."""
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        for i in range(n):
            if len(adj[i]) < 3:
                return None  # Invalid
        
        # Check triangle-free
        for i in range(n):
            for j in adj[i]:
                for k in adj[i]:
                    if j < k and k in adj[j]:
                        return None  # Has triangle
        
        # Enumerate all simple cycles
        cycle_lengths = set()
        
        def find_cycles(start, current, visited, path):
            if len(path) > 12:
                return
            for neighbor in adj[current]:
                if neighbor == start and len(path) >= 4:
                    cycle_lengths.add(len(path))
                elif neighbor not in visited and neighbor > start:
                    # Only visit nodes with index > start to avoid duplicate cycles
                    visited.add(neighbor)
                    path.append(neighbor)
                    find_cycles(start, neighbor, visited, path)
                    path.pop()
                    visited.remove(neighbor)
        
        for start in range(n):
            find_cycles(start, start, {start}, [start])
        
        if not cycle_lengths:
            return None  # No cycles
        
        return len(cycle_lengths)
    
    # Try to improve by searching for graphs with fewer distinct cycle lengths
    # We'll do a local search starting from the fallback solution
    
    best_distinct = get_distinct_cycle_lengths(fallback_edges)
    best_edges = fallback_edges
    
    if best_distinct is None or best_distinct < 1:
        # Shouldn't happen, but be safe
        pass
    
    # Try edge swaps to see if we can reduce distinct cycle lengths
    edge_set = set(tuple(sorted(e)) for e in fallback_edges)
    
    # Generate candidate edges (all possible edges in triangle-free graph on 12 vertices)
    # Actually, let's just do random perturbations
    
    import random
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline - 5.0:  # Leave 5s safety margin
        iterations += 1
        if iterations % 1000 == 0 and time.time() > deadline - 10.0:
            break
        
        # Create a perturbation of the current best solution
        candidate_edges = list(best_edges)
        
        # Randomly modify 1-3 edges
        num_mods = random.randint(1, 3)
        for _ in range(num_mods):
            action = random.choice(['remove', 'add', 'swap'])
            
            if action == 'remove' and candidate_edges:
                idx = random.randint(0, len(candidate_edges) - 1)
                candidate_edges.pop(idx)
            
            elif action == 'add':
                # Add a random edge
                u, v = random.sample(range(12), 2)
                if u > v:
                    u, v = v, u
                if [u, v] not in candidate_edges and (u, v) not in edge_set:
                    candidate_edges.append([u, v])
            
            elif action == 'swap' and candidate_edges:
                idx = random.randint(0, len(candidate_edges) - 1)
                u, v = random.sample(range(12), 2)
                if u > v:
                    u, v = v, u
                if (u, v) not in edge_set:
                    candidate_edges[idx] = [u, v]
        
        # Validate the candidate
        distinct = get_distinct_cycle_lengths(candidate_edges)
        if distinct is not None and distinct < (best_distinct if best_distinct is not None else 100):
            best_distinct = distinct
            best_edges = candidate_edges
            
            if best_distinct == 1:
                break  # Can't do better than 1
    
    # Return the best solution found
    final_edges = []
    for u, v in best_edges:
        if u > v:
            u, v = v, u
        final_edges.append([int(u), int(v)])
    
    return final_edges
