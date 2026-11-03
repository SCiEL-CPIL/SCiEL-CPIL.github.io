import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0  # 5s margin

    # Moore bound for k-regular graph with girth g (odd g=2r+1)
    r = (g - 1) // 2
    moore_bound = 1 + k * sum((k-1)**i for i in range(r))

    def seed_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def exact_girth_check(n, adj, g):
        half_g = g // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= half_g:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # For k=5, g=9, we can try the generalized hexagon of order (4,4)
    # This is a known 5-regular graph with girth 12 and 682 vertices
    # It's smaller than the LUW construction (1116 vertices)
    
    if k == 5 and g <= 12:
        # Try to construct the generalized hexagon H(4)
        # This is the incidence graph of the unique generalized hexagon of order (4,4)
        # It has 682 vertices and girth 12
        
        # The construction uses the projective line over GF(4)
        # GF(4) = {0, 1, a, a^2} where a^2 = a + 1
        
        # Points: (x, y) where x, y in GF(4)^3, not both zero, up to scalar
        # Actually, for H(4), we can use a simpler construction
        
        # Alternative: Use the known explicit construction
        # H(4) can be constructed using the following method:
        # Vertices are pairs (i, j) where i is in {0, 1, ..., 681}
        # But we need the actual edge structure
        
        # Let's use the algebraic construction with GF(4)
        # GF(4) elements: 0, 1, 2, 3 where 2^2 = 2+1 = 3, 3^2 = 2^2+2+1 = 3+2+1 = 0 (mod 2)
        # Wait, let me be more careful with GF(4)
        
        # GF(4) = F2[x]/(x^2+x+1)
        # Elements: 0, 1, x, x+1
        # Let's represent them as: 0=0, 1=1, a=2, a^2=3
        # a^2 = a+1 => 3 = 2+1 = 3 ✓
        # a^3 = a*a^2 = a(a+1) = a^2+a = (a+1)+a = 1 ✓
        
        # Addition table (XOR):
        # 0+0=0, 0+1=1, 0+2=2, 0+3=3
        # 1+0=1, 1+1=0, 1+2=3, 1+3=2
        # 2+0=2, 2+1=3, 2+2=0, 2+3=1
        # 3+0=3, 3+1=2, 3+2=1, 3+3=0
        
        # Multiplication table:
        # 0*x=0
        # 1*x=x
        # 2*0=0, 2*1=2, 2*2=3, 2*3=1
        # 3*0=0, 3*1=3, 3*2=1, 3*3=2
        
        gf4_add = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
        gf4_mul = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        
        def gf4_add(x, y):
            return gf4_add[x][y]
        
        def gf4_mul(x, y):
            return gf4_mul[x][y]
        
        # The generalized hexagon H(4) can be constructed as follows:
        # It's a bipartite graph with two parts of size 341 each
        # Total vertices: 682
        
        # One construction uses the following:
        # Part 1: All triples (x, y, z) in GF(4)^3 with x^2 + y^2 + z^2 = 0, up to scalar
        # Part 2: All lines (a, b, c) in GF(4)^3, up to scalar
        # Edge if ax + by + cz = 0
        
        # But this is complex. Let me use a known explicit edge list approach.
        
        # Actually, let me try a different approach: use the fact that H(4) is the 
        # point-line incidence graph of the unique generalized hexagon of order (4,4).
        
        # For now, let's just return the seed construction if we can't easily construct H(4)
        # But let me try one more thing: a simpler algebraic construction
        
        # The vertices can be indexed as follows:
        # Type 0 vertices: (0, x, y) where x, y in GF(4) - that's 16 vertices
        # Type 1 vertices: (1, x, y, z) where x, y, z in GF(4) with x^2 + y^2 + z^2 = 0
        # This is getting complicated.
        
        # Let me just use the seed construction for now and add a note that H(4) could be better
        pass

    seed_n, seed_edges = seed_construct(k, g)
    
    def try_circulant(k, g, n_max, time_limit):
        """
        Search for a circulant k-regular graph with girth >= g and n < n_max.
        Returns (n, edges) if found, else None.
        """
        if k > 10:
            return None
            
        # Determine search range
        # Start from Moore bound
        # If n_max is small, search is fast
        
        # For efficiency, we only run this if k is small enough to allow search
        
        # For k=5, g=9, n is likely > 1000.
        # Random search for a,b in Z_n.
        
        # We will try a few random samples for each n
        # Or better, iterate n and do a limited random search for S
        
        # Since we don't know the exact structure, random search is a good heuristic.
        
        # However, if n is large (e.g., 1116), the number of pairs is ~150k.
        # Checking girth for each pair is O(n * g * k).
        # 150k * 1116 * 9 * 5 ~ 700 million ops. Too slow in Python.
        
        # We need a faster girth check for circulant graphs.
        # Girth of circulant graph C_n(S) is the length of the shortest cycle.
        # This can be found by BFS on the Cayley graph, but we can prune.
        
        # Alternative: Use the fact that we are looking for girth >= g.
        # This is equivalent to saying there are no relations of length < g.
        
        # Let's stick to a very limited search:
        # Only try n close to the Moore bound?
        # If the seed is 1116, and Moore is 426, searching 426..1115 is 700 values.
        # For each n, try 100 random pairs.
        # 700 * 100 * (girth check cost).
        
        # Girth check for one graph:
        # BFS from 0 to depth g/2.
        # If no cycle found, it's not sufficient (need all vertices).
        # But for circulant graphs, if it's vertex-transitive, checking from 0 might be enough?
        # No, girth is the minimum over all vertices. But in a vertex-transitive graph, the local structure is the same.
        # So if there is a short cycle, it can be moved to include vertex 0?
        # Yes, in a vertex-transitive graph, if there is a cycle of length L, there is one containing any given vertex.
        # So we only need to check for cycles containing vertex 0.
        
        def check_girth_circulant(n, S, g):
            # S is a set of generators
            # BFS from 0
            # We want to find the shortest cycle containing 0.
            # A cycle containing 0 is a path from a neighbor of 0 back to 0.
            # So we do BFS from all neighbors of 0, and look for a common node or return to 0?
            # No, a cycle 0 -> v1 -> ... -> vk -> 0.
            # This is equivalent to finding the shortest path from v1 to vk in the graph where 0 is removed?
            # Or simply: BFS from 0, and if we find an edge (u,v) where dist[u] + dist[v] + 1 < g, and u,v are not parent-child in a way that avoids 0?
            # Actually, standard BFS from 0:
            # If we encounter an edge (u,v) such that dist[u] + dist[v] + 1 < g, and the path from 0 to u and 0 to v are disjoint (except 0), then we have a cycle.
            # In a tree, dist[u] + dist[v] + 1 is the cycle length if u,v are connected and not ancestor-descendant.
            
            dist = [-1] * n
            parent = [-1] * n
            dist[0] = 0
            q = deque([0])
            
            # We stop if we find a cycle < g
            # We only need to search up to depth (g-1)//2
            max_depth = (g - 1) // 2
            
            while q:
                u = q.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in S:
                    w = (u + v) % n
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        q.append(w)
                    else:
                        # Found a connection to a visited node
                        # Check if it's a cycle
                        # The cycle length is dist[u] + dist[w] + 1
                        # We must ensure the cycle goes through 0?
                        # Not necessarily. But if the graph is vertex transitive, a cycle anywhere implies a cycle at 0.
                        # However, the BFS tree from 0 might not capture all cycles.
                        # But if there is a cycle of length L < g, there is one containing 0.
                        # So if we find ANY cycle in the BFS from 0, is it guaranteed to be the shortest?
                        # No. But if we find a cycle of length < g, we fail.
                        # If we don't find any, do we pass?
                        # We must ensure we explored enough.
                        # If the BFS tree has no cycles < g, does the graph have girth >= g?
                        # Yes, because any cycle can be "rotated" to contain 0.
                        # So checking for cycles in the BFS from 0 is sufficient for vertex-transitive graphs.
                        
                        # Calculate cycle length
                        # If w is parent of u, it's the tree edge, ignore.
                        if w == parent[u]:
                            continue
                        # Also if u is parent of w, it's the tree edge (but w is visited, so u is current, w is visited. If w is parent of u, we skip. If u is parent of w, w would be unvisited when we processed u? No, w is visited. If u is parent of w, then dist[w] = dist[u]+1. But we are at u, so dist[u] < dist[w]. This case is handled by "w is unvisited" branch. So here dist[w] <= dist[u] or dist[w] > dist[u] but not parent-child).
                        
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False
            return True

        # Search loop
        # We want n < seed_n
        # Start from moore_bound
        # If k is odd, n must be even
        # If k is even, n can be any
        
        # To save time, we only try a few random S for each n
        # And we stop if time is up
        
        # Determine step for n
        if k % 2 == 1:
            start_n = moore_bound
            if start_n % 2 != 0:
                start_n += 1
            # Ensure n is large enough to have k distinct generators
            # Min n for k=5 is 6? No, n/2 + 2 > k/2?
            # S = {n/2, a, -a, b, -b}. Need n/2, a, b distinct.
            # So n/2 >= 3 => n >= 6.
            
            for n in range(start_n, min(seed_n, start_n + 200)): # Limit search range
                if time.time() > time_limit:
                    break
                if n % 2 != 0:
                    continue
                
                # Try 20 random pairs (a, b)
                # a, b in [1, n/2 - 1]
                # a != b
                # a != n/2, b != n/2
                # Also a != -b mod n => a != n-b => a+b != n
                # Since a,b < n/2, a+b < n, so a+b != n is always true.
                
                # Random selection
                half_n = n // 2
                candidates = list(range(1, half_n))
                
                # If candidates < 2, skip
                if len(candidates) < 2:
                    continue
                
                # Try 50 random samples
                for _ in range(50):
                    if time.time() > time_limit:
                        break
                    # Pick a, b
                    a, b = random.sample(candidates, 2)
                    S = [half_n, a, -a % n, b, -b % n]
                    # Ensure S has 5 distinct elements
                    if len(set(S)) != 5:
                        continue
                    
                    if check_girth_circulant(n, S, g):
                        # Found a valid graph!
                        edges = []
                        for i in range(n):
                            for s in S:
                                j = (i + s) % n
                                if i < j:
                                    edges.append((i, j))
                        return n, edges
        else:
            # Even k
            # S = {a1, -a1, ..., ak/2, -ak/2}
            # n can be any
            pass
            
        return None

    # Try circulant search
    # Only if seed_n is not already minimal (we don't know, but try)
    # If seed_n is small, search is fast
    result = try_circulant(k, g, seed_n, deadline)
    
    if result is not None:
        n, edges = result
        adj = build_adj(edges, n)
        # Verify with exact check just in case
        if exact_girth_check(n, adj, g):
            return n, edges

    # Fallback to seed
    return seed_n, seed_edges
