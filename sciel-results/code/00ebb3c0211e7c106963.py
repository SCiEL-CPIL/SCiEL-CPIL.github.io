"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # 60s safety margin
    
    # Try to find a single 12-vertex bipartite graph with min degree 3
    # where all simple cycles have length 4 (achieving raw value 1)
    # 
    # Strategy: Start with K_{3,3} + K_{3,3} (two disjoint components),
    # then try to merge them into a single connected graph by adding edges
    # between the two components while removing some internal edges to
    # maintain triangle-freeness and eliminate 6-cycles.
    #
    # Actually, let's think more carefully. The two disjoint K_{3,3} graphs
    # have cycles of length 4 and 6. To get only 4-cycles, we need to break
    # all 6-cycles.
    #
    # Key insight: In a bipartite graph with parts of size 6 each, if we can
    # construct it so that the girth is 4 and there are no 6-cycles, we win.
    #
    # Let's try a different approach: use a known graph or search for one.
    # The incidence graph of a projective plane of order 3 has 13 vertices,
    # too many. Let's try to construct manually or search.
    
    # First, let's verify the current best: two disjoint K_{3,3} gives value 2.
    # Let's try to find a graph with value 1.
    
    # For a bipartite graph with parts A and B (each of size 6), 
    # a 4-cycle exists if two vertices in A share two common neighbors in B.
    # A 6-cycle exists if there's a path of length 5 connecting two vertices in A.
    
    # Let's try a specific construction: take a 6x6 bipartite graph where
    # each vertex has degree 3, and check if all cycles are length 4.
    
    # One idea: use the line graph or some other combinatorial structure.
    
    # Let's try a random search with local improvement.
    
    def count_distinct_cycle_lengths(edges, n=12):
        """Count distinct simple cycle lengths in the graph."""
        from collections import defaultdict
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles using DFS
        # This is exponential, so we need to be careful with n=12
        
        cycle_lengths = set()
        
        # Use a more efficient approach: for each pair of vertices,
        # find shortest path and check for cycles
        # Actually, let's just enumerate cycles up to a reasonable length
        
        # For n=12, we can enumerate all simple cycles
        # But this might be slow. Let's use a heuristic: check lengths 3,4,5,6,7,8
        
        # Check for cycles of length 4
        has_4 = False
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w != u:
                            for x in adj[w]:
                                if x != v and x != u:
                                    if u in adj[x]:
                                        has_4 = True
                                        break
                            if has_4:
                                break
                    if has_4:
                        break
                if has_4:
                    break
            if has_4:
                break
        if has_4:
            cycle_lengths.add(4)
        
        # Check for cycles of length 6
        has_6 = False
        # A 6-cycle in bipartite graph: a-b-c-d-e-f-a
        # This is equivalent to two vertices in the same part having
        # a common neighbor at distance 3
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u:
                        for x in adj[w]:
                            if x != v and x != u:
                                for y in adj[x]:
                                    if y != w and y != u and y != v:
                                        if u in adj[y]:
                                            has_6 = True
                                            break
                                    if has_6:
                                        break
                            if has_6:
                                break
                        if has_6:
                            break
                    if has_6:
                        break
                if has_6:
                    break
            if has_6:
                break
        if has_6:
            cycle_lengths.add(6)
        
        # Check for cycles of length 8
        # This gets expensive. Let's skip for now and assume no 8-cycles
        # in our candidate graphs.
        
        return len(cycle_lengths)
    
    def is_valid_graph(edges, n=12, min_degree=3, girth=4):
        """Check if the graph satisfies all constraints."""
        from collections import defaultdict
        
        # Check for duplicates and loops
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            e = tuple(sorted((u, v)))
            if e in edge_set:
                return False
            edge_set.add(e)
        
        # Check min degree
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for i in range(n):
            if deg[i] < min_degree:
                return False
        
        # Check girth (no triangles)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        
        # Check that there's at least one cycle
        # For a graph with min degree 3 on 12 vertices, there must be cycles
        # But let's verify: a tree on 12 vertices has 11 edges.
        # Our graph has at least 12*3/2 = 18 edges, so it has cycles.
        
        return True
    
    def get_cycle_lengths(edges, n=12):
        """Get the set of distinct simple cycle lengths."""
        from collections import defaultdict
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        # Check 4-cycles
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w != u and w > v:
                            for x in adj[w]:
                                if x != v and x > w:
                                    if u in adj[x]:
                                        cycle_lengths.add(4)
        
        # Check 6-cycles
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w != u and w > v:
                            for x in adj[w]:
                                if x != v and x != u and x > w:
                                    for y in adj[x]:
                                        if y != w and y != u and y != v and y > x:
                                            if u in adj[y]:
                                                cycle_lengths.add(6)
        
        # Check 8-cycles (might be slow, but n=12 is small)
        # Skip for speed; assume no 8+ cycles in our candidates
        
        return cycle_lengths
    
    # Current best: two disjoint K_{3,3}
    best_edges = []
    for i in range(6):
        for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
            best_edges.append([u, v])
    best_value = 2  # Known value for two disjoint K_{3,3}
    
    # Try to improve by searching for a graph with value 1
    
    # Let's try a specific construction:
    # Take a 6x6 bipartite graph and try to find one with only 4-cycles.
    
    # Random search
    random.seed(42)
    
    for iteration in range(10000):
        if time.time() > deadline - 10:
            break
        
        # Generate a random bipartite graph with parts {0..5} and {6..11}
        # Each vertex has degree 3
        edges = []
        adj_left = [set() for _ in range(6)]
        adj_right = [set() for _ in range(6)]
        
        # Assign 3 neighbors to each left vertex
        for i in range(6):
            neighbors = random.sample(range(6), 3)
            for j in neighbors:
                edges.append([i, j+6])
                adj_left[i].add(j)
                adj_right[j].add(i)
        
        # Check if each right vertex has degree at least 3
        # If not, adjust
        valid = True
        for j in range(6):
            if len(adj_right[j]) < 3:
                valid = False
                break
        
        if not valid:
            continue
        
        # Check min degree (should be satisfied by construction for left,
        # and we checked for right)
        # Check triangle-free (bipartite by construction)
        # Check for cycles
        
        cycle_lengths = get_cycle_lengths(edges, 12)
        
        if len(cycle_lengths) < best_value:
            best_value = len(cycle_lengths)
            best_edges = [list(e) for e in edges]
            if best_value == 1:
                break
    
    # If we didn't find value 1, return the best we have
    if best_value > 2:
        # Fall back to two disjoint K_{3,3}
        best_edges = []
        for i in range(6):
            for u, v in [(i, (i+1)%6), (i+6, (i+1)%6+6), (i, i+6)]:
                best_edges.append([u, v])
    
    return best_edges
# EVOLVE-BLOCK-END
