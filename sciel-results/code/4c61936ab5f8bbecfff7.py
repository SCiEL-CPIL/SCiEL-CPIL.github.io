import random
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Attempt to find a smaller graph using Circulant Search
    # Target girth g, degree k
    start_time = time.time()
    timeout = 10  # seconds
    
    # Moore bound
    if g % 2 == 0:
        r = g // 2
        moore = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
        
    # Try circulant graphs
    # A 4-regular circulant graph on N vertices has steps s1, s2.
    # Edges: i ~ i+s1, i~i-s1, i~i+s2, i~i-s2.
    # Must ensure s1 != s2, s1 != -s2, etc.
    
    found = False
    best_n = None
    best_edges = None
    
    # We search for N from moore to moore + 100
    # If we find one, we return it.
    for n in range(moore, moore + 100):
        if time.time() - start_time > timeout:
            break
            
        # Try a few random circulant graphs
        for _ in range(1000):
            if time.time() - start_time > timeout:
                break
                
            # Generate random steps
            s1 = random.randint(1, n//2)
            s2 = random.randint(1, n//2)
            
            # Check validity of steps
            if s1 == s2 or s1 == n - s2 or s2 == n - s1 or s1 == 0 or s2 == 0:
                continue
            if s1 == n - s1 or s2 == n - s2: # self loop if n is even and s=n/2
                continue
                
            # Build adjacency and check girth
            # Use BFS from node 0 to find shortest cycle?
            # Girth is the shortest cycle in the graph.
            # For circulant graphs, we can check girth by BFS.
            
            # Optimization: Check if any cycle of length < g exists.
            # We can do BFS from 0 up to depth g//2 - 1.
            # If we find a back edge or a visited node at distance < g//2, we have a short cycle?
            # Actually, girth is the shortest cycle.
            # We can just run BFS from 0 and check distances.
            # But we need to check ALL vertices?
            # For a regular graph, if it's vertex-transitive (circulants are), checking from 0 is sufficient for girth?
            # No, girth is global. But for circulants, the local structure is similar.
            # However, a cycle might not pass through 0.
            # But if the graph is vertex-transitive, the girth is the same for all vertices.
            # So checking from 0 is sufficient to determine the girth?
            # Yes, if the graph is vertex-transitive, the shortest cycle through 0 is the girth.
            
            # Let's verify: In a vertex-transitive graph, every vertex is on a cycle of length equal to the girth.
            # So BFS from 0 finding a cycle of length L implies girth <= L.
            # If no cycle of length < g is found from 0, is girth >= g?
            # Yes, because if there was a cycle of length < g, it would be isomorphic to a cycle through 0 (by vertex transitivity).
            
            # So we just need to check if there is a cycle of length < g through 0.
            
            # BFS from 0
            visited = [-1] * n
            parent = [-1] * n
            visited[0] = 0
            queue = deque([0])
            girth_found = n # infinity
            
            while queue:
                u = queue.popleft()
                if visited[u] >= g//2: # We only care about cycles < g
                    # If we go deeper, we might find longer cycles, but we stop if we exceed g//2
                    # Actually, we stop expanding if depth >= g//2
                    pass
                
                # Neighbors of u in circulant graph
                for s in [s1, -s1, s2, -s2]:
                    v = (u + s) % n
                    if visited[v] == -1:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Back edge
                        # Cycle length = visited[u] + visited[v] + 1
                        cycle_len = visited[u] + visited[v] + 1
                        if cycle_len < girth_found:
                            girth_found = cycle_len
                        if girth_found < g:
                            break
                if girth_found < g:
                    break
            
            if girth_found >= g:
                # Construct edges
                edges = []
                for i in range(n):
                    edges.append((i, (i + s1) % n))
                    edges.append((i, (i + s2) % n))
                # Remove duplicates and sort?
                # The problem asks for a list of edges.
                # Simple graph: no repeated edges.
                # (i, j) and (j, i) are the same.
                # We should output unique undirected edges.
                
                unique_edges = set()
                for i in range(n):
                    for j in [(i + s1) % n, (i + s2) % n]:
                        if i < j:
                            unique_edges.add((i, j))
                        else:
                            unique_edges.add((j, i))
                
                # Verify degree
                # It should be 4-regular.
                
                return n, list(unique_edges)

    # If not found, fallback to LUW
    return luw_construct(k, g)
