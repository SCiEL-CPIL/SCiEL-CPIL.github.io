def construct():
    import time
    import random
    import sys
    from itertools import combinations
    
    deadline = time.time() + 80  # safety margin
    
    def _get_distinct_cycle_lengths(edges):
        """Find all distinct simple cycle lengths in the graph."""
        adj = {i: set() for i in range(12)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # DFS to find simple cycles
        # Use a systematic approach: for each edge, find cycles containing it
        # But this can be expensive. Let's use a simpler approach for small graphs.
        
        # Find all simple cycles using Johnson's algorithm (simplified)
        # For 12 vertices, this should be manageable
        
        def find_cycles(adj):
            cycles = set()
            n = len(adj)
            # For each starting vertex, do DFS to find cycles
            # To avoid duplicates, only consider cycles where the start vertex is the smallest
            
            for start in range(n):
                # DFS from start, only going through vertices >= start
                # except we can come back to start
                stack = [(start, [start], frozenset({start}))]
                while stack:
                    node, path, visited = stack.pop()
                    for neighbor in adj[node]:
                        if neighbor == start and len(path) >= 3:
                            # Found a cycle
                            cycle_len = len(path)
                            cycles.add(cycle_len)
                        elif neighbor > start and neighbor not in visited:
                            new_path = path + [neighbor]
                            new_visited = visited | {neighbor}
                            stack.append((neighbor, new_path, new_visited))
            return cycles
        
        return find_cycles(adj)
    
    def _is_valid(edges):
        """Check if edges form a valid graph."""
        if not edges:
            return False
        adj = {i: set() for i in range(12)}
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree >= 3
        for i in range(12):
            if len(adj[i]) < 3:
                return False
        
        # Check triangle-free
        for u in range(12):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has at least one cycle
        # In a graph with min degree 3 on 12 vertices, there must be cycles
        # (a tree on 12 vertices has at most 11 edges, but min degree 3 requires >= 18 edges)
        return True
    
    def _count_distinct_lengths(edges):
        lengths = _get_distinct_cycle_lengths(edges)
        return len(lengths) if lengths else 0
    
    # Start with the known best: two K_{3,3} components
    best_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            best_edges.append([u, v])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            best_edges.append([u, v])
    
    best_score = _count_distinct_lengths(best_edges)  # Should be 2
    
    # Try to find a graph with 1 distinct cycle length
    # Strategy: try various 3-regular or 4-regular bipartite graphs
    
    # Approach 1: Try all 3-regular bipartite graphs on 6+6 vertices
    # There are not too many. Let's enumerate them.
    
    # A 3-regular bipartite graph with parts L={0..5}, R={6..11}
    # Each vertex in L connects to 3 vertices in R, and vice versa.
    
    # We can represent the graph by the adjacency from L to R.
    # Each vertex in L has a 3-subset of R.
    
    # Let's do a randomized search
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Generate a random 3-regular bipartite graph
        # L = {0,1,2,3,4,5}, R = {6,7,8,9,10,11}
        # Each L vertex picks 3 R vertices, ensuring each R vertex gets exactly 3 connections
        
        # Use a randomized matching approach
        # Start with a random 3-regular bipartite graph
        # One way: take 3 perfect matchings
        
        # Generate 3 random perfect matchings
        matchings = []
        valid = True
        for _ in range(3):
            r_perm = list(range(6))
            random.shuffle(r_perm)
            matching = [(i, 6 + r_perm[i]) for i in range(6)]
            matchings.append(matching)
        
        # Check that no edge is repeated across matchings
        edge_set = set()
        edges = []
        valid = True
        for matching in matchings:
            for u, v in matching:
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append([u, v])
            if not valid:
                break
        
        if not valid:
            continue
        
        if not _is_valid(edges):
            continue
        
        score = _count_distinct_lengths(edges)
        if score < best_score:
            best_score = score
            best_edges = edges[:]
            if best_score == 1:
                break
        
        # Also try modifying the current best
        if attempts % 50 == 0:
            # Try swapping some edges
            pass
    
    # If we found a 1-cycle-length graph, return it
    if best_score == 1:
        # Convert to proper format
        result = []
        for u, v in best_edges:
            result.append([int(u), int(v)])
        return result
    
    # Otherwise return the best we found (which should be the K_{3,3}+K_{3,3} with score 2)
    result = []
    for u, v in best_edges:
        result.append([int(u), int(v)])
    return result
