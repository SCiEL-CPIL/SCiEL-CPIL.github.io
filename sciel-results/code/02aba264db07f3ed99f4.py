import time
import math
from collections import deque

def construct(k, g):
    # Fallback to LUW if search fails or takes too long
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Moore Bound for k=5, g=9
    # n >= 1 + k * sum((k-1)^i for i in 0..(g-3)/2 - 1)
    # g=9 -> r=4. Sum 1+4+16+64 = 85. 1 + 5*85 = 426.
    n_min = 426
    
    # We search for n from n_min upwards
    # We use the circulant form with steps {d1, -d1, d2, -d2, n/2}
    # n must be even.
    
    def luw_fallback():
        # Re-implementation of the seed program logic for safety
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

    def check_circulant(n, d1, d2):
        # Steps: d1, -d1, d2, -d2, n//2
        H = n // 2
        steps = [d1, n - d1, d2, n - d2, H]
        
        # Check for immediate small cycles (length 3, 4)
        # 3-cycles:
        # a + b + c = 0 mod n
        # Possibilities with steps from S:
        # 1. d1 + d1 + d1 = 3d1 = 0 -> n divides 3d1. Since d1 < n/2, 3d1 < 1.5n.
        #    So 3d1 = n. d1 = n/3.
        # 2. d1 + d1 + d2 = 2d1 + d2 = 0 -> 2d1 + d2 = n.
        # 3. d1 + d1 + d1' (where d1' = n-d1) -> d1 + d1 + n - d1 = d1 + n = 0? No.
        #    d1 + (n-d1) + d2 = d2 = 0? No.
        #    d1 + (n-d1) + (n-d1) = n - d1 = 0? No.
        #    d1 + d2 + (n-d2) = d1 = 0? No.
        #    d1 + d2 + (n-d1) = d2 + n = 0? No.
        #    d1 + d2 + d2 = d1 + 2d2 = n.
        #    d1 + d2 + H = 0 -> d1 + d2 + n/2 = n -> d1 + d2 = n/2.
        #    d1 + (n-d2) + H = d1 - d2 + n/2 = 0 -> 2(d1 - d2) = -n -> 2(d2 - d1) = n.
        #       d2 - d1 = n/2. Since d1, d2 < n/2, impossible.
        #    H + H + d1 = n + d1 = 0? No.
        #    H + d1 + d1 = H + 2d1 = n -> 2d1 = n/2 -> d1 = n/4.
        #    H + d1 + d2 = H + d1 + d2 = n -> d1 + d2 = n/2. (Already checked)
        #    H + d1 + (n-d2) = H + d1 - d2 = 0 -> d2 - d1 = H = n/2. Impossible.
        #    H + d2 + d1 = same.
        #    H + d2 + (n-d1) = H + d2 - d1 = 0 -> d1 - d2 = n/2. Impossible.
        #    H + (n-d1) + (n-d2) = H - d1 - d2 = 0 -> d1 + d2 = H = n/2. (Already checked)
        
        if d1 * 3 % n == 0: return False
        if (2 * d1 + d2) % n == 0: return False
        if (d1 + 2 * d2) % n == 0: return False
        if (d1 + d2) % n == H: return False
        if (2 * d1) % n == H: return False
        if (2 * d2) % n == H: return False
        
        # 4-cycles:
        # a + b + c + d = 0
        # Common ones:
        # d1 + d1 + d1 + d1 = 4d1 = 0 -> d1 = n/4 (covered by 3-cycle check? No, 3-cycle was 2d1=H=n/2 -> 4d1=n. Same).
        # d1 + d1 + d2 + d2 = 2d1 + 2d2 = 0 -> d1 + d2 = n/2 (covered).
        # d1 + d1 + d1 + d2 = 3d1 + d2 = n.
        # d1 + d2 + d2 + d1 = 2d1 + 2d2 = n (covered).
        # d1 + d1 + d2 + H = 2d1 + d2 + H = 0 -> 2d1 + d2 = n/2.
        # d1 + d2 + H + H = d1 + d2 + n = 0 -> d1 + d2 = 0 (No).
        # d1 + (n-d1) + d2 + (n-d2) = n (No, sum is n, not 0 mod n? n=0 mod n. Yes.)
        #   Path: 0 -> d1 -> 0 -> d2 -> 0? No, non-backtracking.
        #   0 -> d1 -> 0 is backtracking.
        #   So this specific combination is invalid for simple cycle.
        
        if (3 * d1 + d2) % n == 0: return False
        if (d1 + 3 * d2) % n == 0: return False
        if (2 * d1 + d2) % n == H: return False
        if (d1 + 2 * d2) % n == H: return False
        
        # BFS for girth >= 9
        # Neighbors of 0
        n0 = [d1, n-d1, d2, n-d2, H]
        # Remove duplicates if any (e.g., d1 == d2)
        # But we assume d1 != d2.
        
        # State: (vertex, prev_vertex)
        # Start states: (u, 0) for u in n0
        # We want to see if we can reach (0, v) with dist < 8
        # Dist 0: (u, 0)
        # Dist 1: (w, u) where w in N(u), w != 0
        # ...
        
        visited = set()
        queue = deque()
        
        for u in n0:
            state = (u, 0)
            visited.add(state)
            queue.append((u, 0, 0))
            
        while queue:
            v, prev, dist = queue.popleft()
            
            # If we reached 0, we found a cycle
            if v == 0:
                # Cycle length is dist + 1
                # We started at dist 0 with (u, 0).
                # If we are at (0, v) at dist d, the path is 0->...->v->0.
                # Length is d+1.
                if dist + 1 < g:
                    return False
                # We can stop checking this path
                continue
                
            # Expand
            # Neighbors of v
            # Steps are d1, -d1, d2, -d2, H
            # v + d1, v - d1, v + d2, v - d2, v + H
            nexts = [
                (v + d1) % n,
                (v - d1) % n,
                (v + d2) % n,
                (v - d2) % n,
                (v + H) % n
            ]
            
            for w in nexts:
                if w == prev:
                    continue
                state = (w, v)
                if state not in visited:
                    visited.add(state)
                    queue.append((w, v, dist + 1))
                    
        return True

    # Search
    # We scan n from 426 to 1116
    # If we find a valid one, we return it
    # To ensure we return a valid graph, we construct the component
    
    best_result = None
    
    for n in range(n_min, 1117):
        if n % 2 != 0:
            continue
        H = n // 2
        # d1, d2 in 1..H-1
        # d1 != d2
        # d1 + d2 != H
        # d1 != H/2 if H even? (2d1=H check handles this)
        
        for d1 in range(1, H):
            if time.time() > deadline:
                break
            for d2 in range(d1 + 1, H):
                if (d1 + d2) == H:
                    continue
                # Quick checks
                if check_circulant(n, d1, d2):
                    # Found a valid graph!
                    # Construct the component
                    # Since it's a circulant, it might be disconnected.
                    # We return the component of 0.
                    
                    # Build adjacency
                    adj = [[] for _ in range(n)]
                    steps = [d1, n-d1, d2, n-d2, H]
                    for i in range(n):
                        for s in steps:
                            j = (i + s) % n
                            adj[i].append(j)
                    
                    # BFS from 0 to get component
                    comp = [0]
                    visited = [False] * n
                    visited[0] = True
                    q = deque([0])
                    while q:
                        u = q.popleft()
                        for v in adj[u]:
                            if not visited[v]:
                                visited[v] = True
                                comp.append(v)
                                q.append(v)
                                
                    if len(comp) == n:
                        # Connected
                        edges = []
                        for u in comp:
                            for v in adj[u]:
                                if u < v:
                                    edges.append((u, v))
                        return len(comp), edges
                    else:
                        # Disconnected. Return component.
                        # But wait, if it's disconnected, the "cage" is usually the whole graph?
                        # The task says "on as few vertices as possible".
                        # If I return the component, n is smaller.
                        # Is the component 5-regular? Yes.
                        # Does it have girth >= 9? Yes.
                        # So this is a valid solution with smaller n.
                        # However, the component might be smaller than the current best.
                        # We should return it.
                        
                        # Remap vertices to 0..len(comp)-1
                        map_idx = {v: i for i, v in enumerate(comp)}
                        edges = []
                        for u in comp:
                            for v in adj[u]:
                                if u < v:
                                    edges.append((map_idx[u], map_idx[v]))
                        return len(comp), edges
        if time.time() > deadline:
            break
            
    # If not found, fallback
    return luw_fallback()

