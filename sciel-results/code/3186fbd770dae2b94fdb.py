import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try to find the 320-vertex cage using voltage graph search
    # Base: C4, Group: Z2 x Z2 x Z2 x Z2 x Z5
    # Elements: (a,b,c,d,e)
    # We represent elements as integers 0..79
    # 0..15: Z2^4, 16..79: ? No, map tuple to int.
    # index = a*16 + b*8 + c*4 + d*2 + e
    # e is 0..4.
    # This mapping is not a group homomorphism if we just add integers.
    # We must use tuple arithmetic.
    
    # For speed, precompute addition and negation for all 80 elements.
    # Element: (a,b,c,d,e)
    # Add: (a1^a2, b1^b2, c1^c2, d1^d2, (e1+e2)%5)
    # Neg: (a,b,c,d, -e%5)
    
    elems = []
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    for e in range(5):
                        elems.append((a,b,c,d,e))
                        
    # Map tuple to index
    idx_map = {t:i for i,t in enumerate(elems)}
    
    # Precompute add table: add_table[i][j] = idx of (elems[i] + elems[j])
    add_table = [[0]*80 for _ in range(80)]
    neg_table = [0]*80
    for i in range(80):
        a1,b1,c1,d1,e1 = elems[i]
        neg_table[i] = idx_map[(a1,b1,c1,d1, (-e1)%5)]
        for j in range(80):
            a2,b2,c2,d2,e2 = elems[j]
            res = (a1^a2, b1^b2, c1^c2, d1^d2, (e1+e2)%5)
            add_table[i][j] = idx_map[res]
            
    # Base graph C4: vertices 0,1,2,3. Edges: 0-1, 1-2, 2-3, 3-0.
    # Voltages: v01, v12, v23, v30.
    # Lifted graph: 320 vertices.
    # Vertex (u, g) -> index = u*80 + g.
    
    def check_girth(voltages, max_depth=4):
        # voltages: [v01, v12, v23, v30]
        # Build adjacency list
        adj = [[] for _ in range(320)]
        # Edge 0-1
        v01 = voltages[0]
        v10 = neg_table[v01]
        # Edge 1-2
        v12 = voltages[1]
        v21 = neg_table[v12]
        # Edge 2-3
        v23 = voltages[2]
        v32 = neg_table[v23]
        # Edge 3-0
        v30 = voltages[3]
        v03 = neg_table[v30]
        
        for g in range(80):
            # 0-1
            u0 = 0*80 + g
            u1 = 1*80 + add_table[g][v01]
            adj[u0].append(u1)
            adj[u1].append(u0)
            
            # 1-2
            u1 = 1*80 + g
            u2 = 2*80 + add_table[g][v12]
            adj[u1].append(u2)
            adj[u2].append(u1)
            
            # 2-3
            u2 = 2*80 + g
            u3 = 3*80 + add_table[g][v23]
            adj[u2].append(u3)
            adj[u3].append(u2)
            
            # 3-0
            u3 = 3*80 + g
            u0 = 0*80 + add_table[g][v30]
            adj[u3].append(u0)
            adj[u0].append(u3)
            
        # Check girth
        # BFS from each vertex
        for start in range(320):
            visited = [-1] * 320
            visited[start] = 0
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist >= max_depth:
                    continue
                for nei in adj[node]:
                    if visited[nei] == -1:
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1))
                    else:
                        # Cycle found?
                        # If visited[nei] <= dist, it's a back edge or cross edge.
                        # For girth check, we just need to ensure no cycle < 10.
                        # If we find a cycle of length L, we fail.
                        # Simple check: if we encounter a visited node at distance d2,
                        # the cycle length is dist + 1 + d2? No, this is tricky.
                        # Better: just run BFS to depth 4. If we don't find any cycle, girth >= 10.
                        # If we find a neighbor that is already visited at a smaller distance, 
                        # it might form a cycle.
                        # Actually, for girth >= 10, the BFS tree must be a tree up to depth 4.
                        # So if any node has two parents in the BFS tree, it's a cycle.
                        # Or if a node is visited twice.
                        # If we use standard BFS, each node is visited once.
                        # If a neighbor is already visited, it means there is a cycle.
                        # The length of the cycle is dist + 1 + dist[neighbor]?
                        # Not exactly.
                        # To be safe, if we find ANY edge (u,v) such that dist[u] + dist[v] + 1 < 10 
                        # and they are not parent-child, it's a short cycle.
                        # Simpler: Just check if the graph contains a cycle of length 3,4,5,6,7,8,9.
                        # For 320 vertices, we can just check if the BFS tree has any "back edges" 
                        # that close a cycle of length < 10.
                        pass
                        
        # This check is complex. Let's use a simpler heuristic: 
        # If the graph is a tree up to depth 4 from every vertex, girth >= 10.
        # We can check this by counting edges in the BFS tree.
        # Total edges in 4-regular graph on 320 vertices: 320*4/2 = 640.
        # In a tree of depth 4 from a root: 1 + 4 + 16 + 64 + 256 = 341 nodes.
        # But we have 320 nodes. So the BFS tree will cover all nodes.
        # If the BFS tree has 319 edges, it's a tree.
        # If it has more, there are cycles.
        # But we need to check cycle LENGTH.
        
        # Let's just return True for now and assume the search will find good voltages.
        # Or use a library if available? No.
        
        # Let's refine the girth check:
        # For each start, BFS.
        # If we find a neighbor 'nei' of 'node' such that visited[nei] != -1 and 
        # visited[nei] != dist-1 (parent), then we have a cycle.
        # The cycle length is dist + 1 + visited[nei]? No.
        # It is dist + 1 + (dist[nei] - 1)? No.
        # The path is start -> ... -> node -> nei -> ... -> start.
        # Length = dist[node] + 1 + dist[nei].
        # If this length < 10, fail.
        
        for start in range(320):
            visited = [-1] * 320
            visited[start] = 0
            queue = [(start, 0)]
            while queue:
                node, dist = queue.pop(0)
                if dist >= 4:
                    continue
                for nei in adj[node]:
                    if visited[nei] == -1:
                        visited[nei] = dist + 1
                        queue.append((nei, dist + 1))
                    else:
                        # Check cycle length
                        # If nei is parent, skip.
                        # We don't store parent. But we can check dist[nei] == dist-1.
                        if dist > 0 and visited[nei] == dist - 1:
                            continue
                        # Cycle length
                        # The path from start to nei is dist[nei].
                        # The path from start to node is dist.
                        # The edge node-nei closes the cycle.
                        # Length = dist + 1 + dist[nei]
                        # Wait, this assumes the paths are disjoint.
                        # In BFS, the paths are shortest.
                        # If dist + 1 + dist[nei] < 10, we have a short cycle.
                        if dist + 1 + visited[nei] < 10:
                            return False
        return True

    # Search for voltages
    best_voltages = None
    for restart in range(100):
        if time.time() > deadline:
            break
        voltages = [random.randint(0, 79) for _ in range(4)]
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(4):
                for new_v in range(80):
                    if voltages[i] == new_v:
                        continue
                    old_v = voltages[i]
                    voltages[i] = new_v
                    if check_girth(voltages):
                        best_voltages = voltages.copy()
                        improved = True
                        break
                    voltages[i] = old_v
                if improved:
                    break
        if best_voltages:
            break
            
    if best_voltages:
        # Construct the graph
        v01, v12, v23, v30 = best_voltages
        v10 = neg_table[v01]
        v21 = neg_table[v12]
        v32 = neg_table[v23]
        v03 = neg_table[v30]
        
        edges = []
        for g in range(80):
            # 0-1
            u0 = 0*80 + g
            u1 = 1*80 + add_table[g][v01]
            edges.append((u0, u1))
            
            # 1-2
            u1 = 1*80 + g
            u2 = 2*80 + add_table[g][v12]
            edges.append((u1, u2))
            
            # 2-3
            u2 = 2*80 + g
            u3 = 3*80 + add_table[g][v23]
            edges.append((u2, u3))
            
            # 3-0
            u3 = 3*80 + g
            u0 = 0*80 + add_table[g][v30]
            edges.append((u3, u0))
            
        return 320, edges
    else:
        return luw_construct(k, g)
