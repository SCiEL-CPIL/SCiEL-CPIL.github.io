import time
import random
import math
from collections import deque

def construct(k, g):
    # Handle the specific case k=3, g=16 with a search algorithm
    # Fallback to L-U-W for other cases or if search fails
    if k == 3 and g == 16:
        start_time = time.time()
        deadline = start_time + 110  # 10s buffer
        
        # Moore bound for (3, 16) is 512
        # Known optimal is 936.
        # We will search for N in increasing order.
        
        def is_valid_girth(n, edges, g):
            # Build adjacency list
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check for short cycles
            # For each edge (u, v), check if there is a path from u to v of length < g-1
            # in the graph without (u, v).
            # We can just do BFS from u to v with limit g-2.
            # To optimize, we can just check if any cycle < g exists.
            # A faster check: BFS from every node, track parent.
            # If we see a visited node, cycle length = dist[u] + dist[v] + 1.
            # But this only finds cycles passing through the root? No, BFS finds shortest path from root.
            # If we do BFS from every node, we will find all shortest paths.
            # The girth is the minimum over all pairs (u,v) of dist(u,v) + 1? No.
            # Girth is the length of the shortest cycle.
            # We can find it by checking for each edge (u,v) if dist(u,v) < g-1 in G-(u,v).
            
            # Optimization: Only check edges that might be part of short cycles.
            # But we don't know which ones.
            # For N~1000, O(N * E * 2^(g/2)) is too slow.
            # Let's use a heuristic: Randomly sample edges to check?
            # Or, use the fact that if girth is small, it's likely to be found by BFS from a random node.
            
            # Let's try a different check:
            # Iterate all nodes. BFS.
            # If we find a back-edge to a node at depth d, cycle length is 2*d + 1 (if odd) or similar.
            # Actually, if we do BFS from u, and we visit v which is already visited at depth d_v,
            # and current depth is d_u, then cycle length is d_u + d_v + 1?
            # No, if v is in the tree, the cycle is closed.
            
            # Let's stick to the edge-removal check but optimize it.
            # For each edge (u,v), do BFS from u to v with max_depth = g-2.
            # If found, return False.
            
            # To speed up, we can break early.
            # Also, we can shuffle edges to check random ones first?
            # No, we must check all.
            
            # However, for N=1000, E=1500.
            # 1500 BFS of depth 14.
            # 1500 * (1 + 2 + 4 + ... + 2^13) ~ 1500 * 16384 ~ 24 million steps.
            # In Python, 24M steps is ~ 2-5 seconds.
            # This is acceptable for a few checks per second.
            
            # We can optimize by using a single BFS per node?
            # No, the edge-removal is necessary.
            
            # Let's implement the check.
            for u, v in edges:
                # BFS from u to v, avoiding edge (u,v)
                # We can just do BFS from u, and stop if we reach v.
                # But we must not use the direct edge (u,v).
                # So we start BFS from u's neighbors (excluding v).
                
                visited = {u: 0}
                queue = deque()
                for nei in adj[u]:
                    if nei != v:
                        queue.append((nei, 1))
                        visited[nei] = 1
                        
                while queue:
                    node, dist = queue.popleft()
                    if dist >= g - 1:
                        continue
                    for nei in adj[node]:
                        if nei == u:
                            continue
                        if nei == v:
                            # Found path of length dist + 1
                            # Cycle length = dist + 1 + 1 = dist + 2
                            if dist + 2 < g:
                                return False
                        if nei not in visited:
                            visited[nei] = dist + 1
                            queue.append((nei, dist + 1))
            return True
            
        def generate_random_3reg(n):
            # Configuration model
            while True:
                half_edges = [i for i in range(n) for _ in range(3)]
                random.shuffle(half_edges)
                edges = set()
                valid = True
                for i in range(0, len(half_edges), 2):
                    u = half_edges[i]
                    v = half_edges[i+1]
                    if u == v:
                        valid = False
                        break
                    if (u, v) in edges or (v, u) in edges:
                        valid = False
                        break
                    edges.add((u, v))
                if valid:
                    return list(edges)
                    
        def try_find_graph(n, time_limit):
            # Try to find a valid graph of size n
            # 1. Generate random
            # 2. Check girth
            # 3. If bad, do edge swaps
            
            edges = generate_random_3reg(n)
            
            if is_valid_girth(n, edges, g):
                return edges
            
            # Edge swapping
            # We need a list of edges
            # We will try to break short cycles.
            
            # To make swapping fast, we keep a set of edges for O(1) lookup
            edge_set = set(edges)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
                
            # We will try swaps for a while
            start = time.time()
            attempts = 0
            while time.time() - start < time_limit and attempts < 500:
                attempts += 1
                # Pick a random edge to swap
                # Ideally pick an edge in a short cycle.
                # For simplicity, pick any edge.
                u, v = random.choice(edges)
                
                # Find another edge (x, y) to swap with
                # We want to replace (u,v) and (x,y) with (u,x) and (v,y)
                # or (u,y) and (v,x)
                
                # Pick a random vertex x != u, v
                # and y such that (x,y) is an edge.
                
                # To ensure we don't create loops or multi-edges:
                # New edges: (u, x) and (v, y)
                # Conditions:
                # x != u, x != v
                # y != u, y != v
                # x != y
                # (u, x) not in edge_set
                # (v, y) not in edge_set
                
                # Let's pick x from neighbors of some random node?
                # Or just pick a random edge (x, y) from the list.
                
                x, y = random.choice(edges)
                
                if x == u or x == v or y == u or y == v:
                    continue
                if x == y:
                    continue
                
                # Try swap 1: (u,x) and (v,y)
                if (u, x) not in edge_set and (v, y) not in edge_set and (x, u) not in edge_set and (y, v) not in edge_set:
                    # Perform swap
                    edge_set.remove((u, v))
                    edge_set.remove((x, y))
                    edge_set.add((u, x))
                    edge_set.add((v, y))
                    
                    # Update adjacency
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    # Check girth
                    if is_valid_girth(n, list(edge_set), g):
                        return list(edge_set)
                        
                    # If not valid, we can keep the swap if it didn't make things worse?
                    # Or revert?
                    # For now, keep it and hope it leads to a solution.
                    # But we need to update the `edges` list for the next iteration's random choice?
                    # `edges` is a list. We should update it.
                    edges = list(edge_set)
                    
            return None

        # Search loop
        best_edges = None
        # Start from Moore bound
        # Step size 8 (multiple of 2? 3-regular requires even number of vertices? No, 3N must be even, so N must be even)
        for n in range(512, 2000, 8):
            if time.time() > deadline:
                break
                
            # Try for a short time for this N
            # If we find it, return immediately
            # We give more time to smaller N? Or fixed time?
            # Let's give 5 seconds per N.
            time_slice = 5.0
            if n > 1000:
                time_slice = 2.0
                
            result = try_find_graph(n, time_slice)
            if result:
                return n, result
                
        # If we didn't find one in the search, fallback to L-U-W
        # This ensures we always return a valid graph
        
    # Fallback: L-U-W Construction
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
