import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
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

    if k == 3 and g == 16:
        # Try Random Lift of Petersen Graph
        # Petersen has 10 vertices, girth 5.
        # Lift size m. Total vertices 10*m.
        # Girth approx max(5, 2*log2(m)).
        # We want girth 16, so 2*log2(m) >= 16 => m >= 256.
        
        petersen_edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (2, 7), (3, 8), (3, 9), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 4)]
        # Note: Standard Petersen edges might vary by labeling, but structure is what matters.
        # Let's use a standard Petersen adjacency.
        # Vertices 0-9.
        # Outer pentagon: 0-1-2-3-4-0
        # Inner pentagram: 5-7-9-6-8-5
        # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        
        petersen_adj = [[] for _ in range(10)]
        for u, v in petersen_edges:
            petersen_adj[u].append(v)
            petersen_adj[v].append(u)
            
        # Sort neighbors for consistent lifting
        for i in range(10):
            petersen_adj[i].sort()

        def generate_lift(m):
            # Each base vertex i has 3 neighbors.
            # We assign a permutation pi_{i,j} in S_m for each edge (i,j) in base.
            # The lift vertex (i, k) is connected to (j, pi_{i,j}(k)).
            # To ensure symmetry and undirectedness, we need pi_{j,i} = pi_{i,j}^{-1}.
            
            perms = {}
            # Generate random permutations for each directed edge?
            # Better: Generate random permutations for each undirected edge, and use inverse for the other direction.
            
            edges_base = set()
            for u in range(10):
                for v in petersen_adj[u]:
                    if u < v:
                        edges_base.add((u, v))
            
            # Random permutations for each edge
            for u, v in edges_base:
                # Random permutation of 0..m-1
                p = list(range(m))
                random.shuffle(p)
                # p maps k in u to k' in v.
                # We need the inverse for v to u.
                inv_p = [0] * m
                for idx, val in enumerate(p):
                    inv_p[val] = idx
                perms[(u, v)] = p
                perms[(v, u)] = inv_p
            
            # Construct edges of the lift
            lift_edges = []
            seen = set()
            for u in range(10):
                for k in range(m):
                    u_lift = u * m + k
                    for v in petersen_adj[u]:
                        # Get permutation for (u, v)
                        p = perms[(u, v)]
                        k_prime = p[k]
                        v_lift = v * m + k_prime
                        if u_lift < v_lift:
                            edge = (u_lift, v_lift)
                        else:
                            edge = (v_lift, u_lift)
                        if edge not in seen:
                            seen.add(edge)
                            lift_edges.append(edge)
            return 10 * m, lift_edges

        def check_girth_fast(edges, n, min_girth):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            max_depth = (min_girth - 1) // 2
            for start in range(n):
                visited = {start: 0}
                queue = [(start, 0, -1)]
                while queue:
                    node, dist, parent = queue.pop(0)
                    if dist > max_depth:
                        continue
                    for neighbor in adj[node]:
                        if neighbor == parent:
                            continue
                        if neighbor in visited:
                            cycle_len = dist + 1 + visited[neighbor]
                            if cycle_len < min_girth:
                                return False
                        else:
                            visited[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1, node))
            return True

        # Try m = 256, 512, 1024
        for m in [256, 512, 1024]:
            if time.time() > deadline - 10:
                break
            # Try a few random instances
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                n, edges = generate_lift(m)
                if check_girth_fast(edges, n, g):
                    return n, edges
                    
        # Fallback to LUW
        return luw_construct(k, g)
    else:
        return luw_construct(k, g)
