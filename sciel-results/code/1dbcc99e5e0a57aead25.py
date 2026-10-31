import time
import sys
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s limit

    # Moore bound: minimum vertices for k-regular girth g
    # For k=3, g=16: 1 + 3*(1 + 2^0 + ... + 2^7) = 1 + 3*255 = 766? 
    # Wait, Moore bound for girth g (even) is 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # For k=3, g=16: 1 + 3 * sum_{i=0}^{7} 2^i = 1 + 3 * 255 = 766
    # But the task says best lower bound is 512. Let me re-read.
    # "The smallest known such graph has 936 vertices ... and the best lower bound is 512."
    # So Moore bound might be different or they use a different bound.
    # Actually for girth 16, the radius is 8 (since girth is even, g/2 = 8).
    # Number of vertices at distance i from a vertex: k*(k-1)^(i-1) for i>=1, 1 for i=0.
    # Sum for i=0 to 7: 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 = 382.
    # Wait, that's for a tree of depth 7 (radius 7, diameter 14).
    # For girth 16, the tree of radius 8 (diameter 16) would have:
    # 1 + 3 + 6 + 12 + 24 + 48 + 96 + 192 + 384 = 966.
    # But the Moore bound is 1 + k * sum_{i=0}^{g/2-1} (k-1)^i.
    # For g=16, g/2 = 8, so sum i=0 to 7: 1+2+4+8+16+32+64+128 = 255.
    # 1 + 3*255 = 766.
    # The task says lower bound 512. This might be a different bound or I'm misremembering.
    # Anyway, we'll start from 512 as hinted.

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

    def generate_random_3_regular(n, seed=None):
        if seed is not None:
            random.seed(seed)
        # Configuration model
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        seen = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                valid = False
                break
            seen.add((u, v))
            edges.append((u, v))
        if not valid:
            return None
        return edges

    def build_adjacency(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def check_girth_nonbacktracking(adj, n, min_girth):
        # Check that no cycle shorter than min_girth exists
        # Using non-backtracking walks from every vertex
        # A cycle of length L < min_girth exists iff there's a non-backtracking walk of length L that returns to start
        # We can BFS in the non-backtracking state space (node, prev)
        # State: (current, previous), start: (start, -1) for each start
        # We want to find if any state (v, u) is reached where v == start and the path length < min_girth
        # But we need to be careful: the walk must be non-backtracking, so we can't go back to the node we came from.
        
        max_len = min_girth - 1  # we check up to this length
        # For efficiency, we can do BFS from each start, tracking (node, prev, depth)
        # But that's O(n * n) states which is too much for n=512+.
        
        # Alternative: for each start, do BFS in the non-backtracking space
        # State: (node, prev), dist
        # Start: (start, -1), dist=0
        # Transition: from (u, prev), go to v in adj[u] where v != prev
        # If v == start and dist+1 < min_girth, we found a cycle.
        # But we also need dist+1 >= 3 (minimum cycle length)
        
        for start in range(n):
            # BFS in non-backtracking space
            # dist[node][prev] = shortest non-backtracking walk from (start,-1) to (node, prev)
            # We only care about walks that return to start
            # Use a queue: (node, prev, dist)
            visited = set()
            queue = deque()
            queue.append((start, -1, 0))
            visited.add((start, -1))
            
            while queue:
                node, prev, dist = queue.popleft()
                if dist >= max_len:
                    continue
                for neighbor in adj[node]:
                    if neighbor == prev:
                        continue
                    new_state = (neighbor, node)
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    new_dist = dist + 1
                    if neighbor == start and new_dist >= 3 and new_dist < min_girth:
                        return False
                    queue.append((neighbor, node, new_dist))
        return True

    def check_girth_bfs(adj, n, min_girth):
        # Simpler BFS: check for any cycle of length < min_girth
        # For each start, BFS up to depth (min_girth-1)//2
        # If we find a back-edge to an already visited node at depth d1 from current depth d2,
        # cycle length = d1 + d2 + 1
        max_depth = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if dist[node] > max_depth:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node] and dist[node] != -1:
                        # Found a cycle
                        # The cycle length is dist[node] + dist[neighbor] + 1
                        # But this might not be the shortest cycle through this edge
                        # For a proper check, we need to verify
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Try to find a smaller graph than LUW
    luw_n, luw_edges = luw_construct(k, g)
    
    if k == 3 and g == 16:
        # Start from Moore bound or lower bound
        # Task says lower bound is 512
        start_n = 512
        # n must be even for 3-regular graph
        if start_n % 2 != 0:
            start_n += 1
        
        best_n = luw_n
        best_edges = luw_edges
        
        # Try random graphs with edge swaps to improve girth
        # But first, try pure random generation for small n
        
        for n in range(start_n, min(luw_n, start_n + 500), 2):
            if time.time() > deadline - 10:
                break
            
            # Try a few random graphs
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                edges = generate_random_3_regular(n)
                if edges is None:
                    continue
                
                adj = build_adjacency(edges, n)
                
                # Quick girth check
                if check_girth_bfs(adj, n, g):
                    return n, edges
        
        # If we didn't find one, try edge swaps on a random graph at luw_n or nearby
        # Actually, let's try a different approach: start with a random graph at n=luw_n
        # and try to reduce n? No, that's hard.
        
        # Alternative: try random graphs at n = luw_n - 4, luw_n - 8, etc.
        # But we already tried from 512 up to luw_n.
        
        return best_n, best_edges
    else:
        return luw_n, luw_edges
