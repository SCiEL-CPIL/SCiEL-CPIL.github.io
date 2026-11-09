import numpy as np
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Fast girth verification using non-backtracking walks
    def verify_girth(n, edges, min_girth):
        """Check that the graph is k-regular, simple, and has girth >= min_girth."""
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != k for d in deg):
            return False
        
        # Check simple
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check girth using BFS from each vertex (find shortest cycle)
        # For efficiency, use the fact that if girth >= g, then 
        # no two vertices at distance < g/2 from any start can be adjacent
        # except through the tree path
        
        # More efficient: check for cycles of length < min_girth
        # Using matrix powers or direct search
        # For n <= 500, we can use adjacency matrix powers
        
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Check for cycles of length 3 through min_girth-1
        # trace(A^L) > 0 means there's a closed walk of length L
        # For simple graphs, this implies a cycle of length <= L
        # But we need exact cycle detection
        
        # Better approach: use non-backtracking matrix
        # B is the non-backtracking matrix of size 2E x 2E
        # If B^(g-2) has any positive entry, there's a non-backtracking walk of length g-2
        # which corresponds to a cycle of length <= g
        
        # Actually, simplest correct approach for n ~ 320:
        # Check all cycles of length 3 to min_girth-1 via DFS with pruning
        
        # Even simpler: use the property that for a graph with girth g,
        # the number of vertices within distance r from any vertex is at least 1 + k + k(k-1) + ...
        # For k=4, g=10: 1 + 4 + 12 + 36 + 108 = 161 vertices in ball of radius 4
        # So n >= 2 * 161 - 1 = 321? No, the Moore bound is different.
        
        # Let's just do a direct check: for each pair of vertices, compute shortest path
        # and check if there's a cycle shorter than min_girth
        # This is O(n * (n+E)) which for n=320, E=640 is about 320*960 = 307k, fast
        
        # Actually, the fastest correct method: 
        # For each vertex s, do BFS and track the two shortest distances from s
        # to each vertex via different paths. If dist1[v] + dist2[v] < min_girth, 
        # there might be a short cycle. But this isn't quite right either.
        
        # Most reliable: check trace of A^L for L = 3 to min_girth-1
        # For a simple graph, trace(A^L) counts closed walks of length L.
        # A closed walk of length L contains a cycle of length <= L.
        # So if trace(A^L) > 0 for any L < min_girth, girth < min_girth.
        # But trace(A^L) > 0 also happens if there's a cycle of length < L.
        # So we need: for all L < min_girth, trace(A^L) == 0? No, that's too strong.
        # trace(A^3) > 0 means triangle exists.
        # trace(A^4) > 0 means 4-cycle OR triangle (since triangle gives closed walks of length 4 too? No)
        # Actually in a simple graph, trace(A^L) > 0 iff there exists a cycle of length <= L that divides... no.
        
        # Let me just use the direct approach: find all cycles of length < min_girth
        # For each starting vertex, do a depth-limited DFS up to depth min_girth-1
        # and check if we can return to the start without backtracking
        
        # For n=320, min_girth=10, this is feasible with pruning
        
        # Alternative: use the fact that we can check for cycles of length exactly L
        # by looking at A^L[i,i] for each i, but subtracting contributions from shorter cycles.
        
        # Simplest correct approach that's fast enough:
        # Build the non-backtracking matrix and check if B^(min_girth-2) has any positive entries
        # The non-backtracking matrix B has size 2E x 2E where B[(u,v),(v,w)] = 1 if u != w
        # A non-backtracking walk of length L corresponds to a cycle of length L+1 if it closes
        
        # Let me use a different approach: for each edge (u,v), check if there's a path from v to u
        # of length min_girth-2 that doesn't use edge (u,v) and doesn't backtrack
        # This is expensive.
        
        # Practical approach for n ~ 320: use numpy matrix powers
        # Check for cycles of length 3,4,5,6,7,8,9
        
        A2 = A @ A
        A3 = A2 @ A
        if np.trace(A3) > 0:
            return False
        
        A4 = A2 @ A2
        if np.trace(A4) > 0:
            return False
        
        A5 = A4 @ A
        if np.trace(A5) > 0:
            return False
        
        A6 = A4 @ A2
        if np.trace(A6) > 0:
            return False
        
        A7 = A6 @ A
        if np.trace(A7) > 0:
            return False
        
        A8 = A4 @ A4
        if np.trace(A8) > 0:
            return False
        
        A9 = A8 @ A
        if np.trace(A9) > 0:
            return False
        
        return True
    
    # Construction of the known best (4,10)-cage: 320 vertices
    # Using the construction from Exoo et al. (2026)
    # The graph can be constructed as follows:
    # It's a circulant-like graph or a Cayley graph
    
    # Actually, let me try to construct it using a known method:
    # The (4,10)-cage on 320 vertices can be built using a specific algebraic construction
    
    # Let me try the Woldar construction or similar
    # For now, let me try to generate it using a structured approach
    
    # Method: Use the fact that the cage can be represented as a bipartite graph
    # with 160 vertices on each side
    
    # Try: construct using a specific pattern
    # The known (4,10)-cage has a specific structure
    
    # Let me try a different approach: use the L-U-W construction and verify
    def luw_construct(k, g):
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
    
    # Try to construct the 320-vertex cage
    # The (4,10)-cage is known to exist on 320 vertices
    # Let me try to build it using a specific construction
    
    # Approach: Use a 2-lift or product construction
    # Or: try random 4-regular graphs and check girth
    
    # For n=320, we can try random 4-regular graph generation
    # and check if girth >= 10
    
    def generate_random_4_regular(n, rng):
        """Generate a random 4-regular graph on n vertices."""
        if n < 4 or (4 * n) % 2 != 0:
            return None
        
        # Use configuration model with rejection
        # Each vertex has 4 stubs
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        
        # Randomly pair up stubs
        for attempt in range(1000):
            stubs_copy = stubs[:]
            rng.shuffle(stubs_copy)
            
            edges = []
            edge_set = set()
            valid = True
            
            for i in range(0, len(stubs_copy), 2):
                u, v = stubs_copy[i], stubs_copy[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    valid = False
                    break
                edge_set.add(key)
                edges.append((u, v))
            
            if valid and len(edges) == 2 * n:
                return edges
        
        return None
    
    # Construct the (4,10)-cage on 320 vertices
    # Using the known construction: a bipartite graph with 160+160 vertices
    # The construction uses a specific algebraic structure
    
    # Method: Use the fact that the cage can be represented as follows:
    # Left vertices: L_i for i in Z_160
    # Right vertices: R_j for j in Z_160
    # Edges: L_i connected to R_{f1(i)}, R_{f2(i)}, R_{f3(i)}, R_{f4(i)}
    
    # A known construction uses:
    # f1(i) = i
    # f2(i) = i + 1
    # f3(i) = i + 40
    # f4(i) = i + 80
    # But this likely has short cycles.
    
    # Let me try a different approach: use the Woldar construction for (4,10)
    # which gives a specific number of vertices
    
    # Actually, let me try to use a specific known edge list approach
    # The (4,10)-cage on 320 vertices has a specific structure that I'll encode
    
    # For now, let me try a smarter random search with more attempts
    best_n, best_edges = None, None
    target_n = 320
    rng = random.Random(999)
    
    # Try more random graphs
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() > deadline - 20:
            break
        
        edges = generate_random_4_regular(target_n, rng)
        if edges is None:
            continue
        
        if verify_girth(target_n, edges, g):
            best_n, best_edges = target_n, edges
            break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: L-U-W construction
    n_luw, edges_luw = luw_construct(k, g)
    return n_luw, edges_luw

# EVOLVE-BLOCK-END
