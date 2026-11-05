import random
import time
from collections import deque

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
    def construct_lw(k, g):
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

    # Local Search for (4,10) cage
    def has_girth(adj, n, min_girth):
        # Check if graph has no cycle shorter than min_girth
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Found a cycle
                        # Length is dist[u] + dist[v] + 1
                        # We need to ensure this is not a back edge in tree
                        # Actually, in BFS, if u and v are both visited and not parents,
                        # the cycle length is dist[u] + dist[v] + 1
                        # But we must be careful with even/odd.
                        # If we find any edge (u,v) where dist[u] + dist[v] + 1 < min_girth,
                        # and u, v are not parent-child in the BFS tree, it's a cycle.
                        # However, BFS trees don't have cross edges in undirected graphs?
                        # Yes, in undirected graphs, BFS only has tree edges and back edges to parent.
                        # Wait, in undirected graphs, if (u,v) is an edge and u != parent[v],
                        # then it must be that v is an ancestor of u or u is an ancestor of v?
                        # No, in undirected graphs, BFS partitions edges into tree edges and non-tree edges.
                        # Non-tree edges connect vertices at the same level or adjacent levels.
                        # If they connect same level, cycle length is 2*dist[u] + 1.
                        # If they connect dist[u] and dist[u]+1, it's a tree edge.
                        # So we just check for edges connecting same level or levels differing by > 1?
                        # In undirected graph, |dist[u] - dist[v]| <= 1.
                        # So if dist[u] == dist[v], cycle length is 2*dist[u] + 1.
                        # If dist[u] == dist[v] + 1, it's a tree edge (if parent) or a "chord"?
                        # No, if dist[v] = dist[u] - 1 and v is not parent[u], then it's a cycle.
                        # Length: dist[u] + dist[v] + 1 = dist[u] + dist[u] - 1 + 1 = 2*dist[u].
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
            # If we finish BFS without finding short cycle, continue
        return True

    def random_regular_graph(n, k):
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u != v:
                adj[u].append(v)
                adj[v].append(u)
            else:
                # Self loop, retry
                return random_regular_graph(n, k)
        # Check for multi-edges
        for i in range(n):
            if len(adj[i]) != k or len(set(adj[i])) != k:
                return random_regular_graph(n, k)
        return adj

    def swap_edges(adj, n, k):
        # Pick two edges (a,b) and (c,d) and swap to (a,c) and (b,d) or (a,d) and (b,c)
        # Ensure no loops or multi-edges
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        if len(edges) < 2:
            return False
        idx1, idx2 = random.sample(range(len(edges)), 2)
        a, b = edges[idx1]
        c, d = edges[idx2]
        # Remove
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        # Try swap 1: (a,c) and (b,d)
        if a != c and b != d and c not in adj[a] and d not in adj[b]:
            adj[a].append(c)
            adj[c].append(a)
            adj[b].append(d)
            adj[d].append(b)
            return True
        
        # Try swap 2: (a,d) and (b,c)
        if a != d and b != c and d not in adj[a] and c not in adj[b]:
            adj[a].append(d)
            adj[d].append(a)
            adj[b].append(c)
            adj[c].append(b)
            return True
        
        # Revert
        adj[a].append(b)
        adj[b].append(a)
        adj[c].append(d)
        adj[d].append(c)
        return False

    # Main search
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Try small n
    for n in range(320, 400, 2):
        if time.time() > deadline:
            break
        best_adj = None
        best_score = float('inf')
        
        # Initialize
        adj = random_regular_graph(n, k)
        
        # Local search
        temp = 1.0
        for step in range(10000):
            if time.time() > deadline:
                break
            if has_girth(adj, n, g):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # Calculate score (number of short cycles approx)
            # Just use has_girth result? No, we need a gradient.
            # For now, just random swaps and check girth.
            # This is slow.
            # Better: count 4-cycles.
            
            # If we can't find it quickly, skip to next n
            if step % 100 == 0:
                if has_girth(adj, n, g):
                    return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            swap_edges(adj, n, k)
            
        # If not found, try next n

    # Fallback
    return construct_lw(k, g)

# EVOLVE-BLOCK-END
