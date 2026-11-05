import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # 1. Try Greedy Randomized Construction
    # Moore Bound calculation
    # For k-regular graph of girth g:
    # if g is odd, n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
    # if g is even, n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
    if g % 2 == 1:
        moore_bound = 1 + k * sum((k-1)**i for i in range((g-3)//2 + 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range(g//2))
        
    # LUW size for comparison/fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    luw_n = 2 * (q ** d)
    
    # Try greedy for n from moore_bound to min(luw_n, moore_bound + 500)
    # We only try if it's likely to be faster or smaller
    max_greedy_n = min(luw_n, moore_bound + 200)
    
    best_n = luw_n
    best_edges = None
    
    # Greedy Search
    for n in range(moore_bound, max_greedy_n + 1):
        if time.time() > deadline - 10:
            break
        # Try a few random restarts for this n
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            success, edges = try_greedy(k, g, n)
            if success:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                # If we found a graph close to Moore bound, we can stop
                if n <= moore_bound + 10:
                    return n, edges
            # If n is getting large, we might skip to save time
            if n > moore_bound + 50 and attempt == 0:
                # Only do 1 attempt for larger n
                continue
                
    # 2. Fallback to LUW Construction
    if best_edges is None or time.time() > deadline - 5:
        return luw_construct(k, g)
    else:
        return best_n, best_edges

def try_greedy(k, g, n):
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    # degrees
    deg = [0] * n
    # order of vertices
    order = list(range(n))
    random.shuffle(order)
    
    # We will add edges.
    # To ensure regularity, we process vertices in order.
    # When processing v, we need to pick k neighbors.
    # Some neighbors might be "future" vertices.
    # This is tricky. 
    # Standard approach: 
    # 1. Create a list of "slots" for each vertex.
    # 2. Randomly match slots?
    # Greedy: 
    # For each vertex v, pick k random neighbors u.
    # If u is not connected yet, add edge.
    # If u is connected, check girth.
    
    # Simpler: 
    # Build a random k-regular graph first?
    # Then check girth?
    # If girth is too small, swap edges?
    
    # Let's try: Random k-regular graph generation, then check girth.
    # If girth < g, try to fix or restart.
    
    # Generating random k-regular graph:
    # Configuration model:
    # 2*n stubs.
    # Randomly pair them.
    # Check for multi-edges and self-loops.
    # If clean, check girth.
    
    def generate_random_regular(n, k):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        edges = set()
        adj = [0] * n
        for i in range(0, 2*n, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None
            if (u, v) in edges or (v, u) in edges:
                return None
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj, list(edges)
    
    adj, edges = generate_random_regular(n, k)
    if adj is None:
        return False, None
        
    # Check girth
    # For each edge (u, v), check if there is a path of length < g-1 between u and v
    # Actually, check if girth < g.
    # We can just run BFS from every vertex up to depth (g-1)//2 and check for collisions?
    # Or simpler: check if any edge (u,v) has u and v connected by a path of length < g-1.
    
    # Optimized Girth Check:
    # For each vertex v, run BFS up to depth (g-1)//2.
    # If we find a vertex twice (cycle), check length.
    # This is O(n * k^depth).
    
    depth = (g - 1) // 2
    # If g is even, cycle length can be 2*depth.
    # If g is odd, cycle length can be 2*depth + 1.
    
    # We just need to verify NO cycle < g.
    # So we check if any cycle has length < g.
    
    # BFS from each vertex, tracking parent.
    # If we encounter a visited node, calculate cycle length.
    
    for v in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        queue = [v]
        dist[v] = 0
        found_short_cycle = False
        for u in queue:
            if found_short_cycle:
                break
            # Neighbors of u
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                w = lsb.bit_length() - 1
                mask ^= lsb
                if dist[w] == -1:
                    dist[w] = dist[u] + 1
                    parent[w] = u
                    if dist[w] <= depth:
                        queue.append(w)
                else:
                    # Cycle found
                    # Length is dist[u] + dist[w] + 1
                    # But this is only valid if w is not parent of u
                    if w != parent[u]:
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False, None
    return True, edges

def luw_construct(k, g):
    # Optimized LUW
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

    # Precompute neighbors for all vertices?
    # Total vertices 2 * q^d.
    # For q=5, d=4, 1250 vertices.
    # 1250 * 5 neighbors.
    
    vertices = []
    for side in [0, 1]:
        for x in range(q**d):
            # Convert x to base-q vector of length d
            vec = []
            tmp = x
            for _ in range(d):
                vec.append(tmp % q)
                tmp //= q
            # Pad with None for index 0
            vertices.append((side, (None,) + tuple(vec)))
            
    # Map vertex to index
    index = {v: i for i, v in enumerate(vertices)}
    
    edges = []
    for i, v in enumerate(vertices):
        side, x = v
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            other = (1 - side, tuple(y))
            j = index[other]
            if i < j:
                edges.append((i, j))
                
    return len(vertices), edges
