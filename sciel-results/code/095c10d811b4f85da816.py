import random
import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0 # 120s limit, keep margin
    
    # Fallback: LUW construction (from seed)
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

    # Try to find a smaller graph via random search + 2-switches
    # Moore bound for k=3, g=16 is 512.
    # We try n = 512, 514, ...
    
    def find_shortest_cycle(adj, n):
        # BFS from each node
        # Returns (length, path) or (inf, None)
        min_len = float('inf')
        min_path = None
        for i in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [(i, 0)]
            visited[i] = 0
            while queue:
                u, dist = queue.pop(0)
                if dist >= min_len:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = dist + 1
                        parent[v] = u
                        queue.append((v, dist + 1))
                    elif v != parent[u]:
                        # Found a cycle
                        # Reconstruct path
                        path = []
                        curr = u
                        while curr != v:
                            path.append(curr)
                            curr = parent[curr]
                        path.append(v)
                        path.reverse()
                        # The cycle is path[0]...path[-1] and back to path[0]?
                        # Actually, the cycle is v -> ... -> u -> v
                        # Length is dist + 1
                        if dist + 1 < min_len:
                            min_len = dist + 1
                            min_path = path
            if min_len == 3: # Can't get better for simple graph (no multi-edges)
                break
        return min_len, min_path

    def random_3_reg(n):
        # Generate random 3-regular graph
        # Use configuration model
        edges = []
        half_edges = []
        for i in range(n):
            for _ in range(3):
                half_edges.append(i)
        random.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        used = [False] * (3 * n)
        # Simple matching
        # This is a simplified version, might produce multi-edges or loops
        # We will fix them later or reject
        pairs = []
        for i in range(0, 3 * n, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                continue # Loop
            pairs.append((u, v))
        
        # Check for multi-edges
        edge_set = set()
        for u, v in pairs:
            if u > v: u, v = v, u
            if (u, v) in edge_set:
                return None # Multi-edge
            edge_set.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != 3:
                return None
                
        return adj

    def break_cycle(adj, n, path):
        # path is a list of vertices forming the cycle
        # Pick two non-adjacent edges in the cycle to swap
        # Cycle: v0-v1-...-vk-v0
        # Edges: (v0,v1), (v1,v2), ...
        # Swap (v0,v1) and (vi, vi+1)?
        # We need to pick two edges that are not adjacent in the cycle?
        # Actually, any two edges on the cycle can be swapped to break it,
        # provided we don't create multi-edges.
        
        # Let's pick edge (path[0], path[1]) and (path[k//2], path[k//2+1])
        # Remove them
        u1, v1 = path[0], path[1]
        u2, v2 = path[len(path)//2], path[len(path)//2 + 1]
        
        # Check if new edges (u1, v2) and (u2, v1) are valid
        # They must not be existing edges
        new_e1 = (min(u1, v2), max(u1, v2))
        new_e2 = (min(u2, v1), max(u2, v1))
        
        if new_e1 == new_e2:
            return False
        if u1 == v2 or u2 == v1:
            return False
            
        # Check if new_e1 exists
        if v2 in adj[u1]:
            return False
        if v1 in adj[u2]:
            return False
            
        # Perform swap
        adj[u1].remove(v1)
        adj[u2].remove(v2)
        adj[u1].append(v2)
        adj[u2].append(v1)
        
        return True

    best_n, best_edges = None, None
    
    # Try sizes
    for n in range(512, 1000, 2):
        if time.time() > deadline:
            break
        # Try a few random graphs for this n
        for _ in range(50):
            if time.time() > deadline:
                break
            adj = random_3_reg(n)
            if not adj:
                continue
            
            # Iteratively break cycles
            success = False
            for _ in range(1000): # Max iterations
                if time.time() > deadline:
                    break
                min_len, path = find_shortest_cycle(adj, n)
                if min_len >= g:
                    success = True
                    break
                if path is None:
                    break
                # Try to break
                if not break_cycle(adj, n, path):
                    # If we can't break, restart this graph
                    break
            
            if success:
                # Convert adj to edge list
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_n = n
                best_edges = edges
                return best_n, best_edges

    # Fallback to LUW
    return luw_construct(k, g)
