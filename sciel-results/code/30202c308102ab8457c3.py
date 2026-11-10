import random
import time
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    # Fallback: LUW construction
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

    # Moore bound check
    # For k=5, g=9, Moore bound is 428.
    # Tree of depth 4 has 1 + 5 + 25 + 125 + 625 = 781 nodes.
    # If n >= 781, girth >= 9 iff BFS from every node visits exactly 781 nodes (if n >= 781).
    # Actually, if n > 781, the BFS ball of radius 4 is a tree.
    
    def is_girth_at_least_g(adj, n, g):
        if g < 9:
             # Standard check
             pass
        else:
            # For g=9, radius = 4
            radius = g // 2
            # Number of nodes in tree of depth radius
            # 1 + k + k(k-1) + ...
            # For k=5, radius=4: 1 + 5 + 25 + 125 + 625 = 781
            target_size = 1
            for i in range(1, radius + 1):
                target_size += k * (k - 1) ** (i - 1)
            
            if n < target_size:
                return False
            
            # Check if BFS from each node visits exactly target_size nodes
            # Optimization: Check a subset if n is large?
            # But n is ~1000, so checking all is fine.
            for start in range(n):
                visited = set([start])
                queue = collections.deque([(start, 0)])
                while queue:
                    node, depth = queue.popleft()
                    if depth == radius:
                        continue
                    for neighbor in adj[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
                if len(visited) != target_size:
                    return False
            return True

    # Try to find a smaller graph
    best_n, best_edges = None, None
    
    # Sizes to try
    # The known minimum is 1116.
    # We try sizes close to 1116.
    sizes_to_try = [1116, 1100, 1050, 1000, 1200, 1300]
    
    # Sort by size to find the smallest valid one
    # But we want the smallest n.
    # So we should try smaller n first.
    # However, if n is too small, it's impossible.
    # Moore bound is 428.
    # We try a range.
    
    # Actually, just try 1116 first as it's the known record.
    # If we find a valid graph at 1116, we can return it.
    # If we find one at 1000, even better.
    
    # Let's try a few random graphs for each size.
    
    for n in sorted(sizes_to_try):
        if time.time() > deadline - 5:
            break
            
        # Try a few random graphs
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Generate random k-regular graph
            # Configuration model
            half_edges = [i for i in range(n) for _ in range(k)]
            random.shuffle(half_edges)
            edges = []
            valid = True
            edge_set = set()
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edge_set or (v, u) in edge_set:
                    valid = False
                    break
                edges.append((u, v))
                edge_set.add((u, v))
                edge_set.add((v, u))
            
            if not valid:
                continue
                
            # Check connectivity
            adj = {i: set() for i in range(n)}
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # BFS from 0 to check connectivity
            visited = {0}
            queue = collections.deque([0])
            while queue:
                node = queue.popleft()
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            if len(visited) != n:
                continue
                
            # Check girth
            if is_girth_at_least_g(adj, n, g):
                return n, edges

    # If no random graph worked, return LUW
    return luw_construct(k, g)
