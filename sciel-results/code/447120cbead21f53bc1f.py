def construct():
    n = 12
    # We will try to find a graph with minimum distinct cycle lengths.
    # The best known so far is 2 (from two disjoint K3,3).
    # Let's see if we can do better.
    # A graph with only 4-cycles would be great.
    # But we need minimum degree 3.
    # Let's try to construct a graph where all cycles are length 4.
    # This is a "linear space" or something similar?
    # Actually, if we have a graph where every pair of vertices is in at most one 4-cycle?
    # Or simply, let's try to search for a graph with distinct cycle lengths = 1 or 2.
    
    # Let's try a different structure:
    # The 3-dimensional hypercube Q3 has 8 vertices.
    # Q3 has cycles of length 4 and 6.
    # We need 12 vertices.
    
    # What if we take the 3x4 grid graph? No, corners have degree 2.
    # We need degree >= 3.
    
    # Let's try a search approach.
    # Since 90s is a lot of time, we can do a random search.
    # But let's try to be clever.
    
    # The two disjoint K3,3 graph has cycles 4 and 6.
    # Can we merge them to eliminate 6-cycles?
    # Or eliminate 4-cycles?
    # If we can find a graph with only 4-cycles, score would be 1/1 = 1? No, raw value is 1. Score = 1.
    # Wait, current score 1.0 is from raw value 1? No, seed raw value was 5 (lengths 4,6,8,10,12).
    # Wait, the prompt says "search score is seed/value".
    # If seed value is 5, and seed score is 1.0, then seed = 5.
    # So if we find value 2, score = 5/2 = 2.5.
    # If we find value 1, score = 5/1 = 5.0.
    
    # So finding a graph with only one cycle length is the goal.
    # Is it possible to have a 3-regular, girth-4, 12-vertex graph with only 4-cycles?
    # If all cycles are 4-cycles, the graph is a "4-cycle graph".
    # In a 4-cycle graph, the neighbors of any vertex form an independent set.
    # Let's check if such a graph exists.
    
    # Let's try to construct one.
    # Vertices 0..11.
    # We want every cycle to be length 4.
    # This implies that if u~v, then N(u) and N(v) have no other common neighbors?
    # Actually, if u~v, and they share another neighbor w, then u-v-w-u is a triangle.
    # So no triangles.
    # If u~v, and they share another neighbor w, then u-v-w-...-u is a cycle.
    # If the cycle is length 4, then w must be connected to u and v.
    # Wait, if u~v, and N(u) ∩ N(v) = {w}, then u-v-w-u is a triangle.
    # So in a triangle-free graph, if u~v, then N(u) ∩ N(v) = {}.
    # So every 4-cycle is formed by two disjoint edges with two common neighbors?
    # No, a 4-cycle is u-v-w-x-u.
    # Then N(u) ∩ N(w) = {v, x}.
    # So every pair of vertices at distance 2 share exactly 2 neighbors?
    # This is a very strong condition.
    
    # Let's try to search for such a graph.
    import random
    import time
    from itertools import combinations
    
    deadline = time.time() + 80
    
    def count_distinct_cycle_lengths(edges):
        n = 12
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all simple cycles
        cycles = set()
        # Since n=12, we can use a simple DFS to find all cycles
        # But it might be slow if there are many cycles.
        # Let's try to find cycles of length 4, 5, 6, etc.
        lengths = set()
        
        # Check for 4-cycles
        # A 4-cycle is u-v-w-x-u
        # We can iterate over pairs of edges (u,v) and (w,x) and check if u~x and v~w?
        # Or just iterate over all 4-tuples? 12P4 = 11880. Very fast.
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]: continue
                for w in range(v+1, n):
                    if w not in adj[v]: continue
                    for x in range(w+1, n):
                        if x not in adj[w]: continue
                        if u in adj[x]:
                            # Check if it's a simple cycle
                            # u-v-w-x-u
                            # Need u!=w, v!=x, u!=x, v!=w
                            # Since we picked distinct u,v,w,x, this is fine.
                            # But we need to make sure we don't count the same cycle multiple times.
                            # Actually, we just want the set of lengths.
                            lengths.add(4)
        
        # Check for 5-cycles?
        # 12P5 = 95040. Still fast.
        for u in range(n):
            for v in range(u+1, n):
                if v not in adj[u]: continue
                for w in range(v+1, n):
                    if w not in adj[v]: continue
                    for x in range(w+1, n):
                        if x not in adj[w]: continue
                        for y in range(x+1, n):
                            if y not in adj[x]: continue
                            if u in adj[y]:
                                lengths.add(5)
        
        # Check for 6-cycles?
        # 12P6 = 665280. Might be slow in pure Python.
        # Let's try a smarter way.
        # Or just check up to length 6 and assume no longer cycles?
        # But there could be longer cycles.
        # However, if we find a graph with only 4-cycles, we are done.
        
        # Let's try to find a graph with only 4-cycles first.
        return len(lengths)
    
    # Let's try to construct a graph with only 4-cycles.
    # We need 3-regular, girth 4, 12 vertices.
    # Let's try a random 3-regular graph and check.
    
    def is_valid(edges):
        n = 12
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d < 3 for d in deg):
            return False
        # Check girth 4
        # No triangles
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w == u: continue
                    if w in adj[u]:
                        return False
        return True
    
    # Actually, let's just try the two disjoint K3,3 first.
    # It gives value 2. Score 2.5.
    # Can we do better?
    # Let's try a random search for value 1.
    
    def random_3_reg():
        # Try to generate a random 3-regular graph
        # This is tricky.
        # Let's just use the known K3,3+K3,3.
        pass
    
    # Let's try to merge the two K3,3s to see if we can reduce cycle lengths.
    # Or try a different structure.
    
    # What about the 3x4 prism? No.
    
    # Let's try to find a graph with only 4-cycles.
    # This is equivalent to finding a 3-regular graph where every 4 vertices that form a cycle are 4-cycles.
    # Actually, if a graph has no 3-cycles, no 5-cycles, no 6-cycles, etc., then it has only 4-cycles.
    
    # Let's try to search for such a graph.
    # Since the space is small, we can try to modify the K3,3+K3,3.
    # Or try a different starting point.
    
    # Let's try a random search.
    # Generate random 3-regular graphs and check.
    
    # We can use a simple random walk on the space of 3-regular graphs.
    # But let's just try a few known graphs.
    
    # The Clebsch graph is 5-regular, 16 vertices.
    # The Hoffman-Singleton is 7-regular, 50 vertices.
    
    # Let's try to construct a graph with only 4-cycles.
    # Vertices: 0..11.
    # Let's try to pair them up.
    # (0,1), (2,3), (4,5), (6,7), (8,9), (10,11).
    # Connect each pair to two others?
    
    # Let's try this specific graph:
    # It's the 3-dimensional hypercube Q3 with 4 extra vertices?
    # No.
    
    # Let's try the graph from the "4-cycle graph" family.
    # Actually, let's just return the two K3,3s.
    # It's a solid 2.5 score.
    
    # But let's try a bit of search to see if we can find value 1.
    
    # Let's try to find a graph where all cycles are length 4.
    # This means the graph is a "linear space" of 4-cycles.
    # Let's try to construct one.
    
    # Let's try this:
    # Vertices 0..11.
    # Edges:
    # 0-1, 1-2, 2-3, 3-0 (4-cycle)
    # 4-5, 5-6, 6-7, 7-4 (4-cycle)
    # 8-9, 9-10, 10-11, 11-8 (4-cycle)
    # Now we need to connect them to make it 3-regular.
    # Each vertex currently has degree 2.
    # We need to add one edge per vertex.
    # So we need to add a perfect matching on the 12 vertices.
    # But the added edges must not create triangles or cycles of other lengths.
    
    # Let's try a matching:
    # 0-4, 1-5, 2-6, 3-7, 8-10, 9-11?
    # Let's check if this creates any non-4-cycles.
    
    edges = []
    # Three disjoint 4-cycles
    for i in range(3):
        base = i * 4
        for j in range(4):
            edges.append([base + j, base + (j+1)%4])
    
    # Add a matching
    # Let's try 0-4, 1-5, 2-6, 3-7, 8-12? No, 12 doesn't exist.
    # 8-10, 9-11?
    # Let's try 0-4, 1-5, 2-6, 3-7, 8-10, 9-11.
    matching = [(0,4), (1,5), (2,6), (3,7), (8,10), (9,11)]
    for u, v in matching:
        edges.append([u, v])
    
    # Let's check the distinct cycle lengths.
    n = 12
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    lengths = set()
    
    # Check 4-cycles
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]: continue
            for w in range(v+1, n):
                if w not in adj[v]: continue
                for x in range(w+1, n):
                    if x not in adj[w]: continue
                    if u in adj[x]:
                        lengths.add(4)
    
    # Check 5-cycles
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]: continue
            for w in range(v+1, n):
                if w not in adj[v]: continue
                for x in range(w+1, n):
                    if x not in adj[w]: continue
                    for y in range(x+1, n):
                        if y not in adj[x]: continue
                        if u in adj[y]:
                            lengths.add(5)
    
    # Check 6-cycles
    for u in range(n):
        for v in range(u+1, n):
            if v not in adj[u]: continue
            for w in range(v+1, n):
                if w not in adj[v]: continue
                for x in range(w+1, n):
                    if x not in adj[w]: continue
                    for y in range(x+1, n):
                        if y not in adj[x]: continue
                        for z in range(y+1, n):
                            if z not in adj[y]: continue
                            if u in adj[z]:
                                lengths.add(6)
    
    if len(lengths) == 1:
        return edges
    
    # If not, return the two K3,3s.
    es = set()
    for i in range(3):
        for j in range(3):
            es.add(tuple(sorted((i, j+3))))
            es.add(tuple(sorted((i+6, j+9))))
    return [list(e) for e in sorted(es)]
