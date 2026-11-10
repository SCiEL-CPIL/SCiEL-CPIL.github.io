import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses randomized local search (rewiring) starting from random regular graphs.
    Falls back to Lazebnik-Ustimenko-Woldar construction if no valid graph is found.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Fallback: LUW construction from seed
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

    # Girth Checker
    def check_girth(adj, min_girth):
        n = len(adj)
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([(start, 0, -1)])
            while q:
                u, d, parent = q.popleft()
                if d >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if dist[v] == -1:
                        dist[v] = d + 1
                        q.append((v, d + 1, u))
                    else:
                        # Cycle found
                        cycle_len = d + 1 + dist[v]
                        if cycle_len < min_girth:
                            return False
        return True

    # Generate random k-regular graph
    def generate_random_k_regular(n, k):
        # Configuration model
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                return None # Loop
            if v in adj[u]:
                return None # Multi-edge
            adj[u].append(v)
            adj[v].append(u)
        return adj

    # Local Search
    def local_search(adj, n, k, g, time_limit):
        current_girth = 100 # Assume high
        # Initial girth check
        if not check_girth(adj, g):
             # Find actual girth to guide search?
             # For simplicity, we just try to improve.
             pass
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Pick two random non-adjacent edges
            # To be efficient, we pick random vertices and their edges
            # Simple approach: pick random u, v, x, y
            # Better: pick random edge (u,v) and random edge (x,y)
            
            # Collect all edges
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            
            if len(edges) < 2:
                break
                
            # Pick two random edges
            e1 = random.choice(edges)
            e2 = random.choice(edges)
            
            # Ensure they are disjoint (no shared vertices)
            if len(set(e1) | set(e2)) < 4:
                continue
                
            u, v = e1
            x, y = e2
            
            # Check if (u,x) and (v,y) are already edges or loops
            if x in adj[u] or y in adj[v]:
                continue
                
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[x].remove(y)
            adj[y].remove(x)
            
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            
            # Check girth
            if check_girth(adj, g):
                # Valid!
                final_edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            final_edges.append((i, j))
                return n, final_edges
            
            # If not valid, we could revert or keep.
            # Keeping might lead to better girth later, but might get stuck in bad local minima.
            # For now, always revert to stay in "validity space" of 3-regular?
            # Actually, we don't require intermediate graphs to be valid, just 3-regular.
            # But if we don't revert, we might drift.
            # Let's revert if it doesn't improve girth?
            # We don't track girth value easily.
            # Let's just revert to maintain stability, or use simulated annealing.
            # Given time constraints, simple hill climbing on "is valid" is binary.
            # Let's revert to avoid getting stuck in a "bad" configuration that is hard to escape.
            adj[u].remove(x)
            adj[x].remove(u)
            adj[v].remove(y)
            adj[y].remove(v)
            
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            
            if time.time() > deadline:
                break
                
        return None

    # Main Loop
    # Try n from Moore bound upwards
    moore_bound = 2 * (2**(g//2) - 1) if g % 2 == 0 else 2 * (2**(g//2) - 1) + 1
    # For k=3, g=16, moore_bound = 510.
    
    n = moore_bound
    # Ensure n is even for bipartite? Not strictly required, but 3-regular graphs with odd n have odd number of edges?
    # 3n must be even, so n must be even.
    if n % 2 != 0:
        n += 1
        
    max_n = 2000 # Don't go too high
    
    while n <= max_n and time.time() < deadline:
        # Try to generate and optimize for this n
        best_result = None
        attempts = 0
        while time.time() < deadline and attempts < 5:
            adj = generate_random_k_regular(n, k)
            if adj is None:
                attempts += 1
                continue
                
            # Run local search with a small time budget
            time_budget = min(10, deadline - time.time())
            if time_budget < 1:
                break
                
            result = local_search(adj, n, k, g, time_budget)
            if result:
                best_result = result
                break
                
            attempts += 1
            
        if best_result:
            return best_result
            
        n += 2 # Increase n by 2 to keep it even
        
    # Fallback
    return luw_construct(k, g)
