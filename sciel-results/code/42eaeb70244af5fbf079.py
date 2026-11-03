import time
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 150.0
    
    def check_time():
        if time.time() > deadline - 5.0:
            return True
        return False

    # LUW construction parameters
    # We want girth >= g.
    # LUW graph CD(d, q) has girth at least d+5 (or similar).
    # We try d = g-5 first, then d = g-4, etc.
    
    q = k
    
    # Helper to build LUW graph
    def build_luw(d, q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule for LUW
        # c ranges from 2 to d
        rule = {}
        if d >= 2:
            rule[2] = (1, 1)
        if d >= 3:
            rule[3] = (2, 1)
        if d >= 4:
            rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            # The indices are 1-based in the paper, 0-based in code?
            # Seed code used:
            # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Let's stick to the seed's rule definition to be safe.
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1, x[0] is None, x[1..d] are values
            for first in range(q):
                # y[0] is None, y[1] is first
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # point to line
                        # l[c] - p[c] = l[a] * p[b]
                        # y[c] = x[c] + y[a] * x[b]
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        # line to point
                        # p[c] = l[c] - l[a] * p[b] ? No.
                        # Seed: sub(x[c], mul(x[a], y[b]))
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    def verify_girth(n, edges, min_girth):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles shorter than min_girth using BFS from each vertex
        # This is O(n * m) which might be slow for n=1250, m=3125.
        # 1250 * 3125 ~ 4 million operations. Should be fine in Python if optimized.
        
        # To speed up, we can just check up to min_girth-1.
        # BFS from each node, tracking depth and parent.
        # If we find a node at depth < min_girth-1 that is already visited 
        # and is not the parent, we have a cycle.
        
        # Actually, for bipartite graphs, we can be smarter, but let's stick to simple BFS.
        
        for i in range(n):
            visited = [-1] * n
            parent = [-1] * n
            queue = [(i, 0)]
            visited[i] = 0
            while queue:
                u, depth = queue.pop(0)
                if depth >= min_girth - 1:
                    continue
                for v in adj[u]:
                    if visited[v] == -1:
                        visited[v] = depth + 1
                        parent[v] = u
                        queue.append((v, depth + 1))
                    elif v != parent[u]:
                        # Found a cycle
                        # Length is depth + 1 + (distance from v to root?)
                        # Actually, if we meet a visited node, the cycle length is 
                        # depth + 1 + dist(v, root) - dist(lca, root)?
                        # Simpler: if we see a visited node that is not parent, 
                        # the cycle length is at most depth + 1 + depth? 
                        # No, we need the exact length.
                        # For small girth, we can just check if depth + 1 < min_girth 
                        # and if it forms a cycle.
                        # Actually, if we find any edge (u,v) where visited[v] != -1 and v != parent[u],
                        # and depth + 1 < min_girth, we need to check if it's a valid cycle.
                        # In a tree, the only visited neighbors are parents.
                        # If we find a cross edge, it's a cycle.
                        # The length is depth(u) + 1 + depth(v) - 2*depth(lca).
                        # Since we don't track LCA easily, we can just say 
                        # if we find such an edge, and depth(u)+1 < min_girth, 
                        # it's likely a short cycle.
                        # To be safe, we can return False if we find any such edge 
                        # where depth(u) + 1 < min_girth.
                        # Wait, if depth(u) + 1 >= min_girth, the cycle is long enough.
                        if depth + 1 < min_girth:
                            # Potential short cycle
                            # We should verify the length.
                            # But for now, let's assume if we hit a visited node 
                            # at depth < min_girth, it's a problem.
                            # Actually, we need to ensure the cycle length is < min_girth.
                            # The cycle length is at least depth(u) + 1.
                            # If depth(u) + 1 < min_girth, it's a candidate.
                            # Let's just return False to be conservative.
                            return False
        return True

    best_n = float('inf')
    best_edges = []
    
    # Try d = g-5 first
    # For g=9, d=4.
    candidates = [g - 5, g - 4, g - 3]
    for d in candidates:
        if d < 3:
            continue
        if check_time():
            break
        try:
            n, edges = build_luw(d, q)
            if n < best_n:
                # Verify girth
                if verify_girth(n, edges, g):
                    best_n = n
                    best_edges = edges
                    # If we found a valid graph, we might still try smaller d?
                    # But d is increasing, so n is increasing.
                    # So the first valid one is the best.
                    break
        except Exception:
            pass

    if best_n == float('inf'):
        # Fallback to seed's logic if something went wrong
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges = build_luw(d, q)
        return n, edges

    return best_n, best_edges
# EVOLVE-BLOCK-END
