import time
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # 10s margin
    
    # Fallback: LUW graph
    def luw_construct():
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

    # Try Circulant Search
    # Moore bound for (4,10) is 243.
    # We search n from 243 to 400.
    for n in range(243, 401):
        if time.time() > deadline:
            break
        
        # Steps a, b
        # a < b
        # a != 0, b != 0
        # a != n-a (loop), b != n-b
        # a != b, a != n-b, b != n-a
        for a in range(1, n // 2 + 1):
            if a == n - a: continue # Loop
            for b in range(a + 1, n // 2 + 1):
                if b == n - b: continue # Loop
                if a == b or a == n - b or b == n - a: continue # Multi-edge
                
                # Check Girth < 10
                # We check for non-backtracking closed walks of length L < 10.
                # We can do a BFS/DP on states (node, last_step_index).
                # Steps: 0:+a, 1:-a, 2:+b, 3:-b
                steps = [a, -a, b, -b]
                
                # DP[L][node][last_idx]
                # We only need to check up to L=9.
                # We can early terminate if we find a cycle.
                
                # Optimization: Check L=3 analytically first
                # Sums of 3 steps.
                # Non-backtracking means we can't go i, j, i.
                # So sequences like (0,1,0) are invalid.
                # Valid L=3 sequences:
                # (0,0,0) -> 3a
                # (0,0,1) -> 2a-a = a (wait, 0 is +a, 1 is -a. +a, +a, -a = a. Not 0 unless a=0)
                # Actually, just check if any sum of 3 steps is 0 mod n.
                # The steps are from {a, -a, b, -b}.
                # Possible sums:
                # 3a, 3b, 2a+b, a+2b, 2a-b, 2b-a, a+b (if we allow +a, +b, -a? Sum=b. Not 0).
                # We just need to check if any combination sums to 0.
                # There are 4^3 = 64 combinations.
                # We can precompute the set of sums for a,b,n? No, n changes.
                # Just check:
                found_short = False
                for s1 in steps:
                    for s2 in steps:
                        if s2 == -s1: continue # Backtracking
                        for s3 in steps:
                            if s3 == -s2: continue # Backtracking
                            if (s1 + s2 + s3) % n == 0:
                                found_short = True
                                break
                        if found_short: break
                    if found_short: break
                if found_short:
                    continue
                
                # Check L=4..9 using DP
                # State: (node, last_step_idx)
                # We start from node 0.
                # We consider walks of length L.
                # We want to see if we can return to 0.
                
                # dp[node][last_idx] = True if reachable
                # We iterate L from 1 to 9.
                # Actually, we can just do a BFS from all states?
                # No, we just propagate.
                
                # Initialize L=1
                # From 0, we can go to a (idx 0), -a (idx 1), b (idx 2), -b (idx 3).
                # But we need to track the path to check if it's a cycle?
                # No, if we reach 0 at L < 10, it's a cycle.
                
                # We need to ensure the walk is non-backtracking.
                # The DP state (node, last_idx) ensures the next step is not -last_idx.
                
                # Let's use a set of active states: (node, last_idx)
                active = set()
                for i, s in enumerate(steps):
                    active.add((s % n, i))
                
                # If any active state is (0, i), we have a cycle of length 1?
                # No, s != 0.
                
                for L in range(2, 10):
                    next_active = set()
                    for node, last_idx in active:
                        for next_idx, s in enumerate(steps):
                            if next_idx == (last_idx ^ 1): # 0<->1, 2<->3
                                continue
                            new_node = (node + s) % n
                            next_active.add((new_node, next_idx))
                            if new_node == 0:
                                found_short = True
                                break
                        if found_short:
                            break
                    if found_short:
                        break
                    active = next_active
                    if not active:
                        break
                
                if not found_short:
                    # Girth >= 10
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i + b) % n))
                    return n, edges

    # Fallback
    return luw_construct()
