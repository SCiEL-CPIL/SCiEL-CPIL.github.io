import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Target: 4-regular graph of girth 10 with minimal vertices.
    # Known optimal: 320 vertices.
    
    def verify_girth(n, edges, target_girth):
        """
        Verify that the graph has no cycle shorter than target_girth.
        Uses BFS from each vertex to find shortest cycle.
        """
        if n == 0:
            return True
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        # Check for loops
        for u, v in edges:
            if u == v:
                return False
        
        # Check for repeated edges
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return False
            edge_set.add((u, v))
            
        # Check girth using BFS from each vertex
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            min_cycle = float('inf')
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] * 2 >= min_cycle:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if parent[u] != v and parent[v] != u:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
            if min_cycle < target_girth:
                return False
                
        return True

    # Search for a Circulant Graph C_320(a, b) with girth >= 10
    n = 320
    if k == 4 and g == 10:
        # Steps must be in 1..159
        # We need 4-regular, so 2 steps a, b
        # a != b, a != -b (mod n)
        # Since n=320, -a = 320-a.
        # We just need distinct values in the set {a, -a, b, -b} mod n
        # And a, b != 0.
        
        # Optimization: Only check a < b
        for a in range(1, n // 2):
            if time.time() > deadline - 20:
                break
            for b in range(a + 1, n // 2):
                # Check if steps are distinct
                # a, b, n-a, n-b must be distinct
                if a == b or a == n - b or b == n - a:
                    continue
                # Also a != n-a (a != n/2) and b != n-b
                # Since a, b < n/2, this is true.
                
                # Generate edges
                edges = []
                for i in range(n):
                    j1 = (i + a) % n
                    j2 = (i + b) % n
                    if i < j1:
                        edges.append((i, j1))
                    if i < j2:
                        edges.append((i, j2))
                
                # Verify
                if verify_girth(n, edges, g):
                    return n, edges

    # Fallback: Voltage Graph on K_5 (4-regular) with Z_64
    # K_5 has 5 vertices, 10 edges.
    # Lift with Z_64 -> 320 vertices, 640 edges.
    # Degree 4.
    if k == 4 and g == 10 and time.time() < deadline - 10:
        base_n = 5
        group_n = 64
        total_n = base_n * group_n
        
        # Edges of K_5
        base_edges = []
        for i in range(base_n):
            for j in range(i + 1, base_n):
                base_edges.append((i, j))
                
        # We need to assign voltages (elements of Z_64) to each edge.
        # There are 10 edges. 64^10 is too large to search exhaustively.
        # We can try random assignments or structured ones.
        # However, finding a valid girth 10 lift is hard.
        
        # Let's stick to LUW as a guaranteed valid (though large) fallback if time permits.
        pass

    # Fallback: LUW Construction with d=4 (2048 vertices)
    if time.time() < deadline - 10:
        q = k
        d = 4
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges_luw = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges_luw.append((index[vertex], index[other]))
        
        n_luw = len(order)
        if verify_girth(n_luw, edges_luw, g):
            return n_luw, edges_luw

    # Final Fallback: Seed LUW with d=5 (8192 vertices)
    q = k
    d = 5
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    
    return len(order), edges_luw
