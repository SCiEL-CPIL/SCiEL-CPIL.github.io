"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Strategy: Try to find a 12-vertex, min-degree-3, triangle-free graph
    # with the minimum number of distinct cycle lengths.
    # 
    # Known result: Two disjoint K_{3,3} give cycle lengths {4, 6} -> 2 distinct lengths.
    # 
    # Can we do better? Let's try to find a graph with only 4-cycles or only 6-cycles.
    #
    # Actually, let's try a different construction:
    # The 3-dimensional hypercube (cube graph) has 8 vertices, degree 3, 
    # and contains 4-cycles and 6-cycles.
    #
    # Let me try: Can we make a graph where all cycles are exactly 4?
    # A "crown" graph or "Möbius ladder" might help.
    #
    # Möbius ladder M_6: 12 vertices, 3-regular, but it has triangles? No, it's triangle-free.
    # Actually, the Möbius ladder on 12 vertices (M_6) is 3-regular and triangle-free.
    # Let me check its cycle structure.
    #
    # Alternatively, let me just do a random search for good graphs.
    
    n = 12
    
    def count_distinct_cycle_lengths(adj):
        """Count distinct simple cycle lengths in an undirected graph."""
        # Use DFS to find all simple cycles
        # For efficiency, we'll use a bounded approach
        cycle_lengths = set()
        
        # For each pair of vertices, try to find cycles through them
        # This is expensive, so we'll use a smarter approach
        
        # Actually, for 12 vertices, we can enumerate more carefully
        # Use Johnson's algorithm or a simpler DFS-based approach
        
        # Simpler: for each edge, find cycles containing that edge
        # But this might miss some. Let's use a standard approach.
        
        # For small graphs, we can use the following:
        # For each starting vertex, do DFS to find all simple cycles back to start
        # But this can be exponential. For 12 vertices, it should be manageable.
        
        # Let's use a different approach: enumerate all subsets and check
        # Actually, that's too expensive.
        
        # Use a practical approach: limit cycle length to reasonable values
        # In a 3-regular graph on 12 vertices, max cycle length is 12.
        
        max_len = n
        
        # For each vertex as start, find cycles
        for start in range(n):
            # DFS to find all simple cycles starting and ending at start
            # with at least 3 edges (cycle length >= 4 since triangle-free)
            stack = [(start, [start], {start}, 1)]
            while stack:
                node, path, visited, length = stack.pop()
                if length > max_len:
                    continue
                for neighbor in adj[node]:
                    if neighbor == start and length >= 4:
                        cycle_lengths.add(length)
                    elif neighbor not in visited and neighbor > start:
                        # Only explore neighbors with higher index to avoid duplicates
                        # This is a heuristic to reduce work
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}, length + 1))
        
        return len(cycle_lengths)
    
    def is_valid_graph(adj):
        """Check if graph is valid: min degree >= 3, triangle-free, has a cycle."""
        n = len(adj)
        # Check min degree
        for i in range(n):
            if len(adj[i]) < 3:
                return False
        # Check triangle-free
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[i]:
                        if k > j and k in adj[j]:
                            return False
        # Check has a cycle (for 12 vertices with min degree 3, always has a cycle)
        return True
    
    # Start with the known good solution: two K_{3,3}
    best_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            best_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            best_edges.append([u, v])
    
    best_score = 2  # We know this gives 2 distinct cycle lengths
    
    # Try to improve with random search
    random.seed(42)
    attempts = 0
    max_attempts = 5000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate a random 3-regular graph on 12 vertices
        # Use a simple approach: random pairing of stubs
        stubs = []
        for i in range(n):
            stubs.extend([i] * 3)
        
        random.shuffle(stubs)
        edges_set = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (min(u,v), max(u,v)) in edges_set:
                valid = False
                break
            edges_set.add((min(u,v), max(u,v)))
        
        if not valid or len(edges_set) != 18:  # 12*3/2 = 18 edges for 3-regular
            continue
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Convert to lists for consistency
        adj = [list(s) for s in adj]
        
        if not is_valid_graph(adj):
            continue
        
        # Count distinct cycle lengths
        score = count_distinct_cycle_lengths(adj)
        
        if score < best_score:
            best_score = score
            best_edges = [[u, v] for u, v in sorted(edges_set)]
            if best_score == 1:
                break
    
    # If we found a better solution, use it; otherwise use the known good one
    if best_score < 2:
        result = []
        for e in best_edges:
            result.append([int(e[0]), int(e[1])])
        return result
    else:
        # Return the known good solution: two K_{3,3}
        result = []
        for u in [0, 1, 2]:
            for v in [3, 4, 5]:
                result.append([int(u), int(v)])
        for u in [6, 7, 8]:
            for v in [9, 10, 11]:
                result.append([int(u), int(v)])
        return result

# EVOLVE-BLOCK-END
