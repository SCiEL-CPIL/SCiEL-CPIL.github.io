import random
import time
from collections import deque

def construct(k, g):
    # Moore bound calculation
    # For even g, radius r = g // 2 - 1
    # N_min = 1 + k * sum((k-1)^i for i in range(r))
    # Note: The formula 1 + k * sum_{i=0}^{r-1} (k-1)^i is for radius r.
    # For g=10, r=4. Sum i=0..3.
    # Let's verify the bound for k=4, g=10.
    # r = 4.
    # Vertices: 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161.
    
    # The seed program produces a graph with n ~ 2048.
    # We aim to produce a graph with n close to 161.
    
    # If k or g are not suitable for the specific heuristic, fall back to seed logic
    if k != 4 or g != 10:
        # Fallback to a simple valid construction if parameters differ
        # For now, we focus on the specific task parameters k=4, g=10
        # But we must handle general inputs.
        # Use the seed's L-U-W logic for other cases?
        # The prompt implies the evaluator passes k=4, g=10.
        # We will implement the heuristic for k=4, g=10 and fallback to seed for others.
        
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

    # Specific optimization for k=4, g=10
    # Target n is around 161-200.
    # We use a randomized greedy approach to find a 4-regular graph of girth >= 10.
    
    def try_construct(n):
        # Initialize graph
        adj = [set() for _ in range(n)]
        
        # We try to build the graph by starting with a tree structure and adding edges.
        # However, a pure random search for 4-regular girth 10 is hard.
        # We use a heuristic: 
        # 1. Start with a random 4-regular graph (e.g. from random permutations).
        # 2. Check girth.
        # 3. If girth < 10, try to swap edges to break short cycles.
        
        # Generate a random 4-regular graph
        # Method: Random matching of stubs.
        # 4n stubs.
        # Pair them up randomly.
        # Ensure no loops or multiple edges.
        
        attempts = 0
        max_attempts = 5000
        while attempts < max_attempts:
            attempts += 1
            # Generate random 4-regular graph
            edges = []
            adj = [set() for _ in range(n)]
            
            # Use a simple random walk or random permutation to generate edges
            # This is tricky to do correctly for regular graphs.
            # Alternative: Start with a cycle 0-1-2...n-1-0.
            # Then add random chords to increase degree to 4.
            
            # Base: Cycle
            for i in range(n):
                j = (i + 1) % n
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
            
            # Need 2 more edges per vertex.
            # Total 2n edges to add.
            # We will pick random pairs (u, v) such that u != v and v not in adj[u].
            # And deg(u) < 4, deg(v) < 4.
            
            added = 0
            target_edges = 2 * n
            # List of vertices that need edges
            # We'll just pick random u, v
            
            # To avoid infinite loops, we track degrees
            deg = [2] * n
            
            while added < target_edges:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if v in adj[u]:
                    continue
                if deg[u] >= 4 or deg[v] >= 4:
                    continue
                
                # Check if adding (u, v) creates a cycle shorter than 10?
                # This is expensive. We'll add it and check girth later.
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                added += 1
            
            # Check girth
            # BFS from each vertex to find shortest cycle.
            # Optimization: Only need to check if girth < 10.
            # If any cycle < 10 found, fail.
            
            valid = True
            # For each vertex, do BFS up to depth 4.
            # If we find a neighbor at depth < 4 that is already visited in the same BFS tree (back edge)
            # or a cross edge that creates a short cycle.
            
            # Simpler: For each edge (u, v), remove it and check if dist(u, v) < 9?
            # No, girth is the length of the shortest cycle.
            # A cycle of length L exists if there is a path of length L-1 between u and v for an edge (u,v).
            
            # We check all edges.
            for u in range(n):
                for v in adj[u]:
                    if u > v: continue
                    # Check if there is a path of length <= 8 between u and v
                    # If yes, cycle length <= 9.
                    # BFS from u to v, avoiding edge (u,v).
                    # Limit depth to 8.
                    
                    visited = {u: 0}
                    queue = deque([(u, 0)])
                    found = False
                    while queue:
                        node, dist = queue.popleft()
                        if dist > 8:
                            continue
                        if node == v:
                            if dist < 9: # Cycle length dist + 1 < 10
                                found = True
                                break
                            else:
                                # dist >= 9, cycle length >= 10. OK.
                                pass
                        for nb in adj[node]:
                            if nb == u and node == v: # Avoid backtracking to start via the removed edge?
                                continue
                            # We removed edge (u,v) conceptually?
                            # No, we just check if there's another path.
                            # If nb == v and node == u, that's the edge itself.
                            # We want a path u -> ... -> v.
                            # So we should not traverse the edge (u,v) directly.
                            if (node == u and nb == v) or (node == v and nb == u):
                                continue
                            if nb not in visited:
                                visited[nb] = dist + 1
                                queue.append((nb, dist + 1))
                    if found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Construct edge list
                edge_list = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edge_list.append((i, j))
                return n, edge_list
        
        return None

    # Try increasing n from Moore bound
    # Moore bound for k=4, g=10 is 161.
    # We try a few values.
    for n in range(161, 250, 5):
        result = try_construct(n)
        if result:
            return result
    
    # If all fail, fallback to seed (2048)
    # Or try a larger n
    for n in range(250, 500, 10):
        result = try_construct(n)
        if result:
            return result
            
    # Final fallback: Seed
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
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
