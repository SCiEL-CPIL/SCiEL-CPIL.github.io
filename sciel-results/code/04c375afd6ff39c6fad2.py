import time
import random
from collections import deque

def construct():
    """
    Constructs an undirected simple graph on vertices 0..11.
    Constraints:
    - Min degree >= 3
    - Girth >= 4 (Triangle-free)
    - Contains a cycle
    - Minimize number of distinct simple cycle lengths.
    
    Strategy: 
    1. Start with known good solution: two K_{3,3} (value 2).
    2. Search for a graph with value 1 (all cycles same length).
       Focus on bipartite graphs to guarantee triangle-free.
       If a bipartite graph with only 4-cycles (or only 6-cycles) is found, value is 1.
    3. If not found, return the best solution (value 2).
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Fallback: Two disjoint K_{3,3}
    # Vertices 0,1,2 connected to 3,4,5
    # Vertices 6,7,8 connected to 9,10,11
    fallback_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            fallback_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            fallback_edges.append([u, v])
    
    def get_distinct_cycle_lengths(edges):
        """
        Returns the number of distinct simple cycle lengths.
        Returns None if constraints are violated (min deg < 3, triangle, no cycle).
        """
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        for i in range(n):
            if len(adj[i]) < 3:
                return None
        
        # Check triangle-free
        # For each edge (u,v), check if they share a neighbor
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    if adj[u] & adj[v]:
                        return None
        
        # Enumerate all simple cycles
        # To avoid duplicates, we only start cycles from the smallest vertex in the cycle.
        cycle_lengths = set()
        
        def find_cycles(start, current, visited, path_len):
            for neighbor in adj[current]:
                if neighbor == start and path_len >= 4:
                    cycle_lengths.add(path_len)
                elif neighbor > start and neighbor not in visited:
                    visited.add(neighbor)
                    find_cycles(start, neighbor, visited, path_len + 1)
                    visited.remove(neighbor)
        
        for start in range(n):
            find_cycles(start, start, {start}, 1)
            
        if not cycle_lengths:
            return None
            
        return len(cycle_lengths)

    # Initial best
    best_distinct = get_distinct_cycle_lengths(fallback_edges)
    best_edges = fallback_edges
    
    if best_distinct is None:
        best_distinct = 100 # Should not happen
    
    # Search for better solutions
    # We focus on bipartite graphs (Left: 0-5, Right: 6-11) to ensure triangle-free.
    # This reduces the search space and validation cost.
    
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a random bipartite graph
        # Left nodes: 0-5, Right nodes: 6-11
        # We want min degree 3.
        # Total edges must be at least 3*6 = 18.
        # Max edges 6*6 = 36.
        
        # Randomly select edges
        edges = []
        deg_left = [0] * 6
        deg_right = [0] * 6
        
        # Shuffle all possible edges
        possible_edges = [(i, j) for i in range(6) for j in range(6)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if deg_left[u] < 3 and deg_right[v] < 3:
                edges.append([u, v + 6])
                deg_left[u] += 1
                deg_right[v] += 1
            elif deg_left[u] < 3:
                # Try to add edge even if right is full? No, right degree would exceed?
                # We just want min degree 3. Exceeding is fine.
                # But we need to ensure we CAN reach degree 3 for all.
                # Greedy might fail.
                pass
        
        # Check if all degrees are at least 3
        if min(deg_left) < 3 or min(deg_right) < 3:
            continue
            
        # Convert to full vertex indices
        # u is 0-5, v is 0-5 (mapped to 6-11)
        # The edges list already has [u, v+6]
        
        # Validate
        distinct = get_distinct_cycle_lengths(edges)
        
        if distinct is not None and distinct < best_distinct:
            best_distinct = distinct
            best_edges = edges
            if best_distinct == 1:
                break
        
    # If we didn't find 1, we might have found a better 2? No, 2 is 2.
    # But maybe the random search found a valid graph with 2 that is "cleaner"?
    # The score depends only on the value.
    
    # Final check and format
    final_edges = []
    for u, v in best_edges:
        if u > v:
            u, v = v, u
        final_edges.append([int(u), int(v)])
        
    return final_edges
