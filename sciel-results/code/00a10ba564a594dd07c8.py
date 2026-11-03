import time
import sys
from collections import deque
import random

def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; keep margin before 150s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Moore bound for 3-regular, girth 16
    # n >= 1 + k*(1 + (k-1) + ... + (k-1)^(g/2-2)) for even girth
    # For k=3, g=16: n >= 1 + 3*(1+2+4+8+16+32+64) = 1 + 3*127 = 382
    # Actually for even girth g=2r: n >= 2*(k-1)^r for k-regular
    # r = g/2 = 8, so n >= 2*2^8 = 512

    def moore_bound(k, g):
        if g % 2 == 0:
            r = g // 2
            return 2 * (k - 1) ** r
        else:
            r = g // 2
            return 1 + k * ((k - 1) ** r - 1) // (k - 2) if k > 2 else g + 1

    mb = moore_bound(k, g)

    # Greedy construction: build vertex-by-vertex, maintaining non-backtracking walks
    # For girth g, we need no cycles shorter than g.
    # A cycle of length < g means there's a closed non-backtracking walk of length < g.
    # We maintain for each vertex the set of vertices reachable by non-backtracking walks of length up to g//2 - 1.

    def greedy_construct(k, g, max_time=100.0):
        """
        Greedy algorithm to construct a k-regular graph with girth at least g.
        We build the graph by adding vertices one at a time, ensuring that
        no short cycles are formed.
        """
        t_start = time.time()
        max_walk_len = (g - 2) // 2  # We track walks up to this length
        
        # adj[v] = list of neighbors of v
        adj = []
        # nb_walks[v][l] = set of (vertex, prev_vertex) pairs reachable from v by non-backtracking walk of length l
        # Actually, let's track: for each vertex v and each length l from 1 to max_walk_len,
        # the set of vertices u such that there's a non-backtracking walk of length l from v to u.
        # But we also need to avoid creating cycles, so we need to track the "direction" to avoid backtracking.
        
        # Better approach: maintain for each vertex v a dictionary mapping (prev, length) -> set of current vertices
        # Or simpler: use BFS-like approach to check girth after each edge addition.
        
        # Let's use a different strategy: 
        # 1. Start with a star-like structure
        # 2. Add vertices and edges greedily
        # 3. After each addition, verify no short cycles exist using BFS
        
        # Actually, let's try a simpler approach: 
        # Build the graph level by level (like a tree) and then close it up carefully.
        
        # For a 3-regular graph with girth 16, we can try to construct it as follows:
        # - Start with a root vertex
        # - Level 0: 1 vertex
        # - Level 1: 3 vertices (neighbors of root)
        # - Level 2: 6 vertices (each level-1 vertex has 2 new neighbors)
        # - ...
        # - Level 7: 3*2^6 = 192 vertices
        # - Total vertices up to level 7: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382
        # - But we need 3-regular, so we need to add more vertices and close the graph.
        
        # Let's try a different approach: use the fact that for girth g, the graph must be a tree up to depth (g-1)/2 from any vertex.
        # So we can try to construct a "cage" by ensuring that the BFS tree from each vertex has no cycles within depth g//2.
        
        # Practical approach: 
        # 1. Start with a small 3-regular graph (e.g., a cycle with chords, or a known small cage)
        # 2. Try to expand it while maintaining girth
        
        # For now, let's try to construct a graph with ~512-600 vertices using a greedy approach.
        
        # Initialize with a root and its neighbors
        n = 0
        adj = []
        
        def add_vertex():
            nonlocal n
            adj.append([])
            n += 1
            return n - 1
        
        def add_edge(u, v):
            if u == v:
                return False
            if v in adj[u]:
                return False
            adj[u].append(v)
            adj[v].append(u)
            return True
        
        # Check if adding edge (u, v) would create a cycle shorter than g
        def would_create_short_cycle(u, v, g):
            # BFS from u to v, avoiding the edge (u,v) itself
            # If we find a path of length < g-1 from u to v (not using edge u-v), then adding u-v creates a cycle of length < g
            if u == v:
                return True
            if v in adj[u]:
                return True  # Already an edge
            
            # BFS from u, not going through v initially
            dist = {u: 0}
            parent = {u: -1}
            queue = deque([u])
            while queue:
                curr = queue.popleft()
                if dist[curr] >= g - 2:  # We only care about paths of length < g-1
                    break
                for nb in adj[curr]:
                    if nb == v and dist[curr] == 0:
                        continue  # Don't use the edge we're considering
                    if nb not in dist:
                        dist[nb] = dist[curr] + 1
                        parent[nb] = curr
                        if nb == v:
                            # Found a path from u to v of length dist[v]
                            # Adding edge u-v creates a cycle of length dist[v] + 1
                            return dist[v] + 1 < g
                        queue.append(nb)
            return False
        
        # Start with a small seed graph
        # For girth 16, we need a fairly large graph. Let's start with a 12-cycle and add chords carefully.
        
        # Actually, let's try a completely different approach: 
        # Use a known construction or a heuristic to generate a candidate graph, then verify.
        
        # Let's try to construct a graph with exactly the Moore bound or close to it.
        # For k=3, g=16, Moore bound is 512.
        
        # Try to construct a 512-vertex 3-regular graph with girth 16.
        # One approach: use a circulant or quasi-circulant graph.
        
        # For now, let's fall back to the LUW construction but with a better rule selection.
        # The current LUW code produces 4374 vertices for d=7. Let's see if we can do better.
        
        # Actually, let's just return the LUW result for now, but optimize the verification.
        return None  # Signal to use LUW
        
    # Try greedy first
    greedy_result = greedy_construct(k, g)
    if greedy_result is not None:
        return greedy_result
    
    # Fall back to LUW construction
    q = k
    
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q

    def build_rule(d):
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        return rule

    def build_graph(d, rule):
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)

        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return n, edges, adj

    def verify_girth_fast(n, adj, min_girth):
        """
        Fast girth verification using BFS from each vertex.
        """
        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    d_min = (g - 1) // 2
    
    for d in range(d_min, 11):
        if time.time() - t0 > deadline - 5.0:
            break
            
        rule = build_rule(d)
        try:
            n, edges, adj = build_graph(d, rule)
        except MemoryError:
            break
        except Exception:
            continue

        if verify_girth_fast(n, adj, g):
            return n, edges
    
    return 0, []
